import { CallToolRequest, CallToolResultSchema, ListToolsResult, ListToolsResultSchema } from '@modelcontextprotocol/sdk/types.js'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { StdioClientTransport, StdioServerParameters } from '@modelcontextprotocol/sdk/client/stdio.js'
import { BaseToolkit, tool, Tool } from '@langchain/core/tools'
import { z } from 'zod'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js'
import { callMcpRunner } from './mcpBridge'

const MCP_IDLE_CLOSE_MS = 120_000
// A dead Streamable HTTP endpoint otherwise sits until the SDK's 60s request timeout
// and every other request waiting on this toolkit waits with it.
const MCP_TRANSPORT_PROBE_TIMEOUT_MS = 8_000

export class MCPToolkit extends BaseToolkit {
    tools: Tool[] = []
    _tools: ListToolsResult | null = null
    model_config: any
    serverParams: StdioServerParameters | any
    transportType: 'stdio' | 'sse'
    private client: Client | null = null
    private connectPromise: Promise<Client> | null = null
    private closing: Promise<void> | null = null
    private idleTimer: ReturnType<typeof setTimeout> | null = null
    private inflight = 0
    private initPromise: Promise<void> | null = null
    private requestQueue: Promise<void> = Promise.resolve()
    private preferredRemoteTransport: 'streamable' | 'sse' | null = null

    constructor(serverParams: StdioServerParameters | any, transportType: 'stdio' | 'sse') {
        super()
        this.serverParams = serverParams
        this.transportType = transportType
    }

    private newClient(): Client {
        return new Client(
            {
                name: 'flowise-client',
                version: '1.0.0'
            },
            {
                capabilities: {}
            }
        )
    }

    // Method to create a new client with transport
    async createClient(): Promise<Client> {
        if (this.transportType === 'stdio') {
            const client = this.newClient()

            // "inherit" and an undrained "pipe" both stall the child once the stderr
            // buffer fills. The tool call then waits until timeout, and a blocked
            // stderr write stalls this process too.
            const params = {
                ...this.serverParams,
                stderr: this.serverParams.stderr ?? 'ignore',
                env: {
                    ...(this.serverParams.env || {}),
                    PATH: process.env.PATH
                }
            }

            const transport = new StdioClientTransport(params as StdioServerParameters)
            try {
                await client.connect(transport)
                return client
            } catch (error) {
                await client.close().catch(() => undefined)
                throw error
            }
        }

        if (this.serverParams.url === undefined) {
            throw new Error('URL is required for SSE transport')
        }

        const baseUrl = new URL(this.serverParams.url)
        const headers = this.serverParams.headers

        if (this.preferredRemoteTransport === 'sse') {
            return this.connectSse(baseUrl, headers)
        }
        if (this.preferredRemoteTransport === 'streamable') {
            return this.connectStreamable(baseUrl, headers)
        }

        const configuredTimeout = Number(this.serverParams.options?.timeout)
        const probeTimeout =
            Number.isFinite(configuredTimeout) && configuredTimeout > 0
                ? Math.min(configuredTimeout, MCP_TRANSPORT_PROBE_TIMEOUT_MS)
                : MCP_TRANSPORT_PROBE_TIMEOUT_MS

        try {
            const client = await this.connectStreamable(baseUrl, headers, { timeout: probeTimeout })
            this.preferredRemoteTransport = 'streamable'
            return client
        } catch (streamableError: any) {
            try {
                const client = await this.connectSse(baseUrl, headers)
                this.preferredRemoteTransport = 'sse'
                return client
            } catch (sseError) {
                throw new Error(
                    `Could not connect to MCP server at ${baseUrl.toString()}. Streamable HTTP failed with "${
                        streamableError?.message ?? streamableError
                    }", SSE failed with "${(sseError as any)?.message ?? sseError}"`
                )
            }
        }
    }

    private async connectStreamable(baseUrl: URL, headers: Record<string, string> | undefined, requestOptions?: { timeout: number }) {
        const client = this.newClient()
        try {
            const transport = headers
                ? new StreamableHTTPClientTransport(baseUrl, { requestInit: { headers } })
                : new StreamableHTTPClientTransport(baseUrl)
            await client.connect(transport, requestOptions)
            return client
        } catch (error) {
            await client.close().catch(() => undefined)
            throw error
        }
    }

    private async connectSse(baseUrl: URL, headers: Record<string, string> | undefined) {
        const client = this.newClient()
        try {
            const transport = headers
                ? new SSEClientTransport(baseUrl, {
                      requestInit: {
                          headers
                      },
                      eventSourceInit: {
                          fetch: (url, init) => fetch(url, { ...init, headers })
                      }
                  })
                : new SSEClientTransport(baseUrl)
            await client.connect(transport)
            return client
        } catch (error) {
            await client.close().catch(() => undefined)
            throw error
        }
    }

    private enqueue<T>(fn: () => Promise<T>): Promise<T> {
        const run = this.requestQueue.then(fn, fn)
        this.requestQueue = run.then(
            () => undefined,
            () => undefined
        )
        return run
    }

    /**
     * One live connection per toolkit. Spawning a stdio process (or opening an HTTP/SSE
     * session) per tools/list and per tools/call stalls the server under load.
     */
    private async getConnectedClient(): Promise<Client> {
        if (this.closing) {
            await this.closing
        }
        if (this.client) {
            return this.client
        }
        if (!this.connectPromise) {
            this.connectPromise = this.createClient()
                .then((client) => {
                    this.client = client
                    client.onclose = () => {
                        if (this.client === client) {
                            this.client = null
                            this.connectPromise = null
                        }
                    }
                    return client
                })
                .catch((error) => {
                    this.connectPromise = null
                    throw error
                })
        }
        return this.connectPromise
    }

    private scheduleIdleClose() {
        if (this.inflight > 0 || !this.client) return
        if (this.idleTimer) clearTimeout(this.idleTimer)
        this.idleTimer = setTimeout(() => {
            this.idleTimer = null
            if (this.inflight === 0) {
                void this.close()
            }
        }, MCP_IDLE_CLOSE_MS)
    }

    private async withClient<T>(fn: (client: Client) => Promise<T>): Promise<T> {
        this.inflight += 1
        if (this.idleTimer) {
            clearTimeout(this.idleTimer)
            this.idleTimer = null
        }
        try {
            const client = await this.getConnectedClient()
            return await fn(client)
        } finally {
            this.inflight -= 1
            if (this.inflight === 0) {
                this.scheduleIdleClose()
            }
        }
    }

    private isConnectionError(error: unknown): boolean {
        const message = error instanceof Error ? error.message : String(error)
        return message.includes('Connection closed') || message.includes('Not connected') || message.includes('Request timed out')
    }

    get busy(): boolean {
        return this.inflight > 0
    }

    async forceClose(): Promise<void> {
        if (this.idleTimer) {
            clearTimeout(this.idleTimer)
            this.idleTimer = null
        }
        const client = this.client
        this.client = null
        this.connectPromise = null
        if (!client) return
        const closing = client.close().then(
            () => undefined,
            () => undefined
        )
        this.closing = closing
        try {
            await closing
        } finally {
            if (this.closing === closing) {
                this.closing = null
            }
        }
    }

    async close(): Promise<void> {
        if (this.inflight > 0) return
        await this.forceClose()
    }

    async callTool(name: string, args: Record<string, unknown>): Promise<string> {
        // The runner process does the actual I/O. Doing it here stalls every other request for the whole tool call.
        if (process.env.FLOWISE_MCP_RUNNER !== '1') {
            return callMcpRunner({
                type: 'call',
                serverParams: this.serverParams,
                transportType: this.transportType,
                name,
                args
            })
        }

        const req: CallToolRequest = { method: 'tools/call', params: { name, arguments: args } }
        const requestOptions = this.serverParams.options

        // Remote sessions that stay open keep an SSE body alive. The next call then
        // waits until timeout, and the open fetch holds a connection the rest of the
        // process needs. Open a client for this call only.
        if (this.transportType !== 'stdio') {
            const client = await this.createClient()
            try {
                const res = await client.request(req, CallToolResultSchema, requestOptions)
                return JSON.stringify(res.content)
            } finally {
                await client.close().catch(() => undefined)
            }
        }

        // One stdio process, one request at a time. Overlapped writes deadlock the
        // child (it stops answering) and the chat request sits until timeout.
        return this.enqueue(async () => {
            try {
                const res = await this.withClient((client) => client.request(req, CallToolResultSchema, requestOptions))
                return JSON.stringify(res.content)
            } catch (error) {
                if (this.isConnectionError(error)) {
                    await this.forceClose()
                }
                throw error
            }
        })
    }

    async initialize() {
        if (process.env.FLOWISE_MCP_RUNNER !== '1') {
            if (this._tools !== null) return
            if (!this.initPromise) {
                this.initPromise = callMcpRunner({
                    type: 'list',
                    serverParams: this.serverParams,
                    transportType: this.transportType
                })
                    .then(async (toolsResult) => {
                        this._tools = toolsResult
                        this.tools = await this.get_tools()
                    })
                    .catch(async (error) => {
                        this.initPromise = null
                        throw error
                    })
            }
            return this.initPromise
        }

        if (this._tools !== null) return
        if (!this.initPromise) {
            const load = async (client: Client) => {
                if (this._tools !== null) return
                const requestOptions = this.serverParams.options
                this._tools = await client.request({ method: 'tools/list' }, ListToolsResultSchema, requestOptions)
                this.tools = await this.get_tools()
            }
            this.initPromise = (
                this.transportType === 'stdio'
                    ? this.enqueue(() => this.withClient(load))
                    : this.createClient().then(async (client) => {
                          try {
                              await load(client)
                          } finally {
                              await client.close().catch(() => undefined)
                          }
                      })
            ).catch(async (error) => {
                this.initPromise = null
                await this.forceClose()
                throw error
            })
        }
        return this.initPromise
    }

    async get_tools(): Promise<Tool[]> {
        if (this._tools === null) {
            throw new Error('Must initialize the toolkit first')
        }
        const toolsPromises = this._tools.tools.map(async (tool: any) => {
            return await MCPTool({
                toolkit: this,
                name: tool.name,
                description: tool.description || '',
                argsSchema: createSchemaModel(tool.inputSchema)
            })
        })
        const res = await Promise.allSettled(toolsPromises)
        const errors = res.filter((r) => r.status === 'rejected')
        if (errors.length !== 0) {
            console.error('MCP Tools failed to be resolved', errors)
        }
        const successes = res.filter((r) => r.status === 'fulfilled').map((r) => r.value)
        return successes
    }
}

export async function MCPTool({
    toolkit,
    name,
    description,
    argsSchema
}: {
    toolkit: MCPToolkit
    name: string
    description: string
    argsSchema: any
}): Promise<Tool> {
    return tool(
        async (input): Promise<string> => {
            return toolkit.callTool(name, (input ?? {}) as Record<string, unknown>)
        },
        {
            name: name,
            description: description,
            schema: argsSchema
        }
    )
}
function createSchemaModel(
    inputSchema: {
        type: 'object'
        properties?: Record<string, any>
        required?: string[]
        additionalProperties?: boolean | object
    } & { [k: string]: unknown }
): any {
    if (inputSchema.type !== 'object' || !inputSchema.properties) {
        throw new Error('Invalid schema type or missing properties')
    }

    function createPropertySchema(schema: any): z.ZodTypeAny {
        switch (schema.type) {
            case 'string':
                if (Array.isArray(schema.enum) && schema.enum.length > 0) {
                    return z.enum(schema.enum as [string, ...string[]]).describe(schema.description || '')
                }
                return z.string().describe(schema.description || '')
            case 'number':
                if (Array.isArray(schema.enum) && schema.enum.length > 0) {
                    const enumLiterals = (schema.enum as number[]).map((v: number) => z.literal(v))
                    if (enumLiterals.length === 1) {
                        return enumLiterals[0].describe(schema.description || '')
                    }
                    return z.union([...enumLiterals] as [any, any, ...any[]]).describe(schema.description || '')
                }
                return z.number().describe(schema.description || '')
            case 'boolean':
                return z.boolean().describe(schema.description || '')
            case 'array':
                if (schema.items) {
                    return z.array(createPropertySchema(schema.items)).describe(schema.description || '')
                }
                return z.array(z.any()).describe(schema.description || '')
            case 'object':
                if (schema.properties) {
                    // Recursively build the object schema
                    const properties = Object.entries(schema.properties).reduce((acc, [key, propSchema]) => {
                        acc[key] = createPropertySchema(propSchema)
                        return acc
                    }, {} as Record<string, z.ZodTypeAny>)
                    // Mark non-required fields as optional
                    const required = schema.required || []
                    for (const key of Object.keys(properties)) {
                        if (!required.includes(key)) {
                            properties[key] = properties[key].optional()
                        }
                    }
                    return z.object(properties).describe(schema.description || '')
                } else if (schema.additionalProperties) {
                    if (schema.additionalProperties === true) {
                        return z.record(z.any()).describe(schema.description || '')
                    } else if (typeof schema.additionalProperties === 'object') {
                        return z.record(createPropertySchema(schema.additionalProperties)).describe(schema.description || '')
                    }
                }
                return z.record(z.any()).describe(schema.description || '')
            default:
                return z.any().describe(schema.description || '')
        }
    }

    // Build the root object schema
    const properties = Object.entries(inputSchema.properties).reduce((acc, [key, propSchema]) => {
        acc[key] = createPropertySchema(propSchema)
        return acc
    }, {} as Record<string, z.ZodTypeAny>)

    // Mark non-required fields as optional
    const required = inputSchema.required || []
    for (const key of Object.keys(properties)) {
        if (!required.includes(key)) {
            properties[key] = properties[key].optional()
        }
    }

    let baseSchema = z.object(properties)

    // Handle additionalProperties at the root level
    if (inputSchema.additionalProperties) {
        if (inputSchema.additionalProperties === true) {
            baseSchema = baseSchema.catchall(z.any())
        } else if (typeof inputSchema.additionalProperties === 'object') {
            baseSchema = baseSchema.catchall(createPropertySchema(inputSchema.additionalProperties))
        }
    }

    return baseSchema
}

export const validateArgsForLocalFileAccess = (args: string[]): void => {
    const dangerousPatterns = [
        // Absolute paths
        /^\/[^/]/, // Unix absolute paths starting with /
        /^[a-zA-Z]:\\/, // Windows absolute paths like C:\

        // Relative paths that could escape current directory
        /\.\.\//, // Parent directory traversal with ../
        /\.\.\\/, // Parent directory traversal with ..\
        /^\.\./, // Starting with ..

        // Local file access patterns
        /^\.\//, // Current directory with ./
        /^~\//, // Home directory with ~/
        /^file:\/\//, // File protocol

        // Common file extensions that shouldn't be accessed
        /\.(exe|bat|cmd|sh|ps1|vbs|scr|com|pif|dll|sys)$/i,

        // File flags and options that could access local files
        /^--?(?:file|input|output|config|load|save|import|export|read|write)=/i,
        /^--?(?:file|input|output|config|load|save|import|export|read|write)$/i
    ]

    for (const arg of args) {
        if (typeof arg !== 'string') continue

        // Check for dangerous patterns
        for (const pattern of dangerousPatterns) {
            if (pattern.test(arg)) {
                throw new Error(`Argument contains potential local file access: "${arg}"`)
            }
        }

        // Check for null bytes
        if (arg.includes('\0')) {
            throw new Error(`Argument contains null byte: "${arg}"`)
        }

        // Check for very long paths that might be used for buffer overflow attacks
        if (arg.length > 1000) {
            throw new Error(`Argument is suspiciously long (${arg.length} characters): "${arg.substring(0, 100)}..."`)
        }
    }
}

export const validateCommandInjection = (args: string[]): void => {
    const dangerousPatterns = [
        // Shell metacharacters
        /[;&|`$(){}[\]<>]/,
        // Command chaining
        /&&|\|\||;;/,
        // Redirections
        />>|<<|>/,
        // Backticks and command substitution
        /`|\$\(/,
        // Process substitution
        /<\(|>\(/
    ]

    for (const arg of args) {
        if (typeof arg !== 'string') continue

        for (const pattern of dangerousPatterns) {
            if (pattern.test(arg)) {
                throw new Error(`Argument contains potentially dangerous characters: "${arg}"`)
            }
        }
    }
}

export const validateEnvironmentVariables = (env: Record<string, any>): void => {
    const dangerousEnvVars = ['PATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH']

    for (const [key, value] of Object.entries(env)) {
        if (dangerousEnvVars.includes(key)) {
            throw new Error(`Environment variable '${key}' modification is not allowed`)
        }

        if (typeof value === 'string' && value.includes('\0')) {
            throw new Error(`Environment variable '${key}' contains null byte`)
        }
    }
}

export const validateMCPServerConfig = (serverParams: any): void => {
    // Validate the entire server configuration
    if (!serverParams || typeof serverParams !== 'object') {
        throw new Error('Invalid server configuration')
    }

    // Command allowlist - only allow specific safe commands
    const allowedCommands = ['node', 'npx', 'python', 'python3', 'docker']

    if (serverParams.command && !allowedCommands.includes(serverParams.command)) {
        throw new Error(`Command '${serverParams.command}' is not allowed. Allowed commands: ${allowedCommands.join(', ')}`)
    }

    // Validate arguments if present
    if (serverParams.args && Array.isArray(serverParams.args)) {
        validateArgsForLocalFileAccess(serverParams.args)
        validateCommandInjection(serverParams.args)
    }

    // Validate environment variables
    if (serverParams.env) {
        validateEnvironmentVariables(serverParams.env)
    }
}
