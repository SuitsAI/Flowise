import { CallToolRequest, CallToolResultSchema, ListToolsResult, ListToolsResultSchema } from '@modelcontextprotocol/sdk/types.js'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { StdioClientTransport, StdioServerParameters } from '@modelcontextprotocol/sdk/client/stdio.js'
import { BaseToolkit, tool, Tool } from '@langchain/core/tools'
import { z } from 'zod'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js'

export class MCPToolkit extends BaseToolkit {
    tools: Tool[] = []
    _tools: ListToolsResult | null = null
    model_config: any
    serverParams: StdioServerParameters | any
    transportType: 'stdio' | 'sse'
    constructor(serverParams: StdioServerParameters | any, transportType: 'stdio' | 'sse') {
        super()
        this.serverParams = serverParams
        this.transportType = transportType
    }

    // Method to create a new client with transport
    async createClient(): Promise<Client> {
        const client = new Client(
            {
                name: 'flowise-client',
                version: '1.0.0'
            },
            {
                capabilities: {}
            }
        )

        let transport: StdioClientTransport | SSEClientTransport | StreamableHTTPClientTransport

        if (this.transportType === 'stdio') {
            // Compatible with overridden PATH configuration
            const params = {
                ...this.serverParams,
                env: {
                    ...(this.serverParams.env || {}),
                    PATH: process.env.PATH
                }
            }

            transport = new StdioClientTransport(params as StdioServerParameters)
            await client.connect(transport)
        } else {
            if (this.serverParams.url === undefined) {
                throw new Error('URL is required for SSE transport')
            }

            const baseUrl = new URL(this.serverParams.url)
            try {
                if (this.serverParams.headers) {
                    transport = new StreamableHTTPClientTransport(baseUrl, {
                        requestInit: {
                            headers: this.serverParams.headers
                        }
                    })
                } else {
                    transport = new StreamableHTTPClientTransport(baseUrl)
                }
                if (this.serverParams.headers) {
                    transport = new StreamableHTTPClientTransport(baseUrl, {
                        requestInit: {
                            headers: this.serverParams.headers
                        }
                    })
                } else {
                    transport = new StreamableHTTPClientTransport(baseUrl)
                }
                await client.connect(transport)
            } catch (error) {
                if (this.serverParams.headers) {
                    transport = new SSEClientTransport(baseUrl, {
                        requestInit: {
                            headers: this.serverParams.headers
                        },
                        eventSourceInit: {
                            fetch: (url, init) => fetch(url, { ...init, headers: this.serverParams.headers })
                        }
                    })
                } else {
                    transport = new SSEClientTransport(baseUrl)
                }
                if (this.serverParams.headers) {
                    transport = new SSEClientTransport(baseUrl, {
                        requestInit: {
                            headers: this.serverParams.headers
                        },
                        eventSourceInit: {
                            fetch: (url, init) => fetch(url, { ...init, headers: this.serverParams.headers })
                        }
                    })
                } else {
                    transport = new SSEClientTransport(baseUrl)
                }
                await client.connect(transport)
            }
        }

        return client
    }

    async initialize() {
        if (this._tools === null) {
            let client: Client | null = null
            try{
                client = await this.createClient()

                // Pass timeout options to the request
                const requestOptions = this.serverParams.options
                this._tools = await client.request({ method: 'tools/list' }, ListToolsResultSchema, requestOptions)

                this.tools = await this.get_tools()
            }
            finally{
                // Close the initial client after initialization
                if (client) {
                    await client.close()
                }
            }
        }
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
            console.error('MCP Tools falied to be resolved', errors)
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
            // Create a new client for this request
            let client: Client | null = null

            try {
                client = await toolkit.createClient()
                const req: CallToolRequest = { method: 'tools/call', params: { name: name, arguments: input as any } }
                
                // Pass timeout options to the request
                const requestOptions = toolkit.serverParams.options
                const res = await client.request(req, CallToolResultSchema, requestOptions)
                
                const content = res.content
                const contentString = JSON.stringify(content)
                return contentString
            } finally {
                // Always close the client after the request completes
                if (client) {
                    await client.close()
                }
            }
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
