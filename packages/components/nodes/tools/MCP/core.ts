import { CallToolRequest, CallToolResultSchema, ListToolsResult, ListToolsResultSchema } from '@modelcontextprotocol/sdk/types.js'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { StdioClientTransport, StdioServerParameters } from '@modelcontextprotocol/sdk/client/stdio.js'
import { BaseToolkit, tool, Tool } from '@langchain/core/tools'
import { z } from 'zod'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js'

async function initializeClient(serverParams: StdioServerParameters | any, transportType: 'stdio' | 'sse'): Promise<Client> {
    const client = new Client(
        {
            name: 'flowise-client',
            version: '1.0.0'
        },
        {
            capabilities: {}
        }
    )
    if (transportType === 'stdio') {
        // Compatible with overridden PATH configuration
        serverParams.env = {
            ...(serverParams.env || {}),
            PATH: process.env.PATH
        }

        const transport = new StdioClientTransport(serverParams as StdioServerParameters)
        await client.connect(transport)
    } else {
        if (serverParams.url === undefined) {
            throw new Error('URL is required for SSE transport')
        }

        const baseUrl = new URL(serverParams.url)
        try {
            const transport = new StreamableHTTPClientTransport(baseUrl)
            await client.connect(transport)
        } catch (error) {
            const transport = new SSEClientTransport(baseUrl)
            await client.connect(transport)
        }
    }
    return client
}

export class MCPToolkit extends BaseToolkit {
    tools: Tool[] = []
    _tools: ListToolsResult | null = null
    model_config: any
    transport: StdioClientTransport | SSEClientTransport | StreamableHTTPClientTransport | null = null
    client: Client | null = null
    serverParams: StdioServerParameters | any
    transportType: 'stdio' | 'sse'
    constructor(serverParams: StdioServerParameters | any, transportType: 'stdio' | 'sse') {
        super()
        this.serverParams = serverParams
        this.transportType = transportType
    }
    async initialize() {
        if (this._tools === null) {
            this.client = await initializeClient(this.serverParams, this.transportType)

            this._tools = await this.client.request({ method: 'tools/list' }, ListToolsResultSchema)

            this.tools = await this.get_tools()

            await this.client.close()
        }
    }

    async get_tools(): Promise<Tool[]> {
        if (this._tools === null || this.client === null) {
            throw new Error('Must initialize the toolkit first')
        }
        const toolsPromises = this._tools.tools.map(async (tool: any) => {
            if (this.client === null) {
                throw new Error('Client is not initialized')
            }
            return await MCPTool({
                name: tool.name,
                description: tool.description || '',
                argsSchema: createSchemaModel(tool.inputSchema),
                serverParams: this.serverParams,
                transportType: this.transportType
            })
        })
        return Promise.all(toolsPromises)
    }
}

export async function MCPTool({
    name,
    description,
    argsSchema,
    serverParams,
    transportType
}: {
    name: string
    description: string
    argsSchema: any
    serverParams: StdioServerParameters | any
    transportType: 'stdio' | 'sse'
}): Promise<Tool> {
    return tool(
        async (input): Promise<string> => {
            const req: CallToolRequest = { method: 'tools/call', params: { name: name, arguments: input } }
            const client = await initializeClient(serverParams, transportType)
            const res = await client.request(req, CallToolResultSchema)
            await client.close()
            const content = res.content
            const contentString = JSON.stringify(content)
            return contentString
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
        properties?: import('zod').objectOutputType<{}, import('zod').ZodTypeAny, 'passthrough'> | undefined
    } & { [k: string]: unknown }
): any {
    if (inputSchema.type !== 'object' || !inputSchema.properties) {
        throw new Error('Invalid schema type or missing properties')
    }

    // 
    return jsonSchemaToZod(inputSchema)
}

function jsonSchemaToZod(schema: any): z.ZodType<any> {
  if (!schema) {
    return z.any();
  }

  switch (schema.type) {
    case 'string':
      if (schema.enum) {
        return z.enum(schema.enum as [string, ...string[]]);
      }
      return z.string();
    
    case 'number':
      return z.number();
    
    case 'boolean':
      return z.boolean();
    
    case 'object':
      if (schema.properties) {
        const shape: Record<string, z.ZodType<any>> = {};
        
        // Process each property
        for (const [key, propSchema] of Object.entries(schema.properties)) {
          const isRequired = schema.required?.includes(key);
          const zodType = jsonSchemaToZod(propSchema);
          shape[key] = isRequired ? zodType : zodType.optional();
        }

        // Create the object schema
        const objSchema = z.object(shape);

        // Handle additionalProperties
        return schema.additionalProperties === false ? objSchema.strict() : objSchema;
      }
      return z.record(z.any());
    
    case 'array':
      if (schema.items) {
        return z.array(jsonSchemaToZod(schema.items));
      }
      return z.array(z.any());
    
    default:
      return z.any();
  }
}
