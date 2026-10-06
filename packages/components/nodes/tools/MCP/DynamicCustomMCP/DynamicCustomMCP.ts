import { Tool, tool } from '@langchain/core/tools'
import { z } from 'zod'
import hash from 'object-hash'
import { ICommonObject, INode, INodeData, INodeOptionsValue, INodeParams } from '../../../../src/Interface'
import { MCPToolkit } from '../core'

// One toolkit per resolved server config, shared by every chat that uses it.
// Idle connections are closed by MCPToolkit; this cap only bounds memory if
// mcpConfigValues change on every run (each value is its own process).
const MAX_CACHED_MCP_TOOLKITS = 64
const mcpToolkitCache = new Map<string, { tools: Tool[]; toolkit: MCPToolkit }>()
const mcpToolkitInflight = new Map<string, Promise<Tool[]>>()

function rememberMCPToolkit(cacheKey: string, entry: { tools: Tool[]; toolkit: MCPToolkit }) {
    mcpToolkitCache.delete(cacheKey)
    mcpToolkitCache.set(cacheKey, entry)
    for (const [key, cached] of mcpToolkitCache) {
        if (mcpToolkitCache.size <= MAX_CACHED_MCP_TOOLKITS) break
        if (key === cacheKey || cached.toolkit.busy) continue
        mcpToolkitCache.delete(key)
        void cached.toolkit.close()
    }
}

const RESPONSE_BATCH_SIZE = 50_000

const mcpServerConfig = `{
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
}`

class DynamicCustom_MCP implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    documentation: string
    credential: INodeParams
    inputs: INodeParams[]

    constructor() {
        this.label = 'Dynamic Custom MCP'
        this.name = 'dynamicCustomMCP'
        this.version = 1.0
        this.type = 'Dynamic Custom MCP Tool'
        this.icon = 'dynamicCustomMCP.png'
        this.category = 'Tools (MCP)'
        this.description = 'Custom MCP Config'
        this.documentation = 'https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search'
        this.inputs = [
            {
                label: 'MCP Server Config',
                name: 'mcpServerConfig',
                type: 'code',
                hideCodeExecute: true,
                placeholder: mcpServerConfig
            },
            {
                label: 'MCP Config Values',
                name: 'mcpConfigValues',
                type: 'json',
                optional: true,
                acceptVariable: true,
                list: true
            }
        ]
        this.baseClasses = ['Tool']
    }

    //@ts-ignore
    loadMethods = {
        listActions: async (nodeData: INodeData, options: ICommonObject): Promise<INodeOptionsValue[]> => {
            try {
                const toolset = await this.getTools(nodeData, options)
                toolset.sort((a: any, b: any) => a.name.localeCompare(b.name))

                return toolset.map(({ name, ...rest }) => ({
                    label: name.toUpperCase(),
                    name: name,
                    description: rest.description || name
                }))
            } catch (error) {
                return [
                    {
                        label: 'No Available Actions',
                        name: 'error',
                        description: 'No available actions, please check your API key and refresh'
                    }
                ]
            }
        }
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const tools = await this.getTools(nodeData, options)
        return tools
    }

    async getTools(nodeData: INodeData, _options?: ICommonObject): Promise<Tool[]> {
        const mcpConfigValuesStr = nodeData.inputs?.mcpConfigValues
        let mcpConfigValues: ICommonObject = {}
        if (mcpConfigValuesStr) {
            try {
                mcpConfigValues = typeof mcpConfigValuesStr === 'object' ? mcpConfigValuesStr : JSON.parse(mcpConfigValuesStr)
            } catch (exception) {
                return []
                //throw new Error("Invalid JSON in the DynamicCustomMCP's mcpConfigValues: " + exception)
            }
        }

        const mcpServerConfig = nodeData.inputs?.mcpServerConfig as string

        if (!mcpServerConfig) {
            return []
            //throw new Error('MCP Server Config is required')
        }

        try {
            let serverParams
            if (typeof mcpServerConfig === 'object') {
                serverParams = mcpServerConfig
            } else if (typeof mcpServerConfig === 'string') {
                const serverParamsString = convertToValidJSONString(mcpServerConfig, mcpConfigValues)
                serverParams = JSON.parse(serverParamsString)
            }

            let tmpServerParams = JSON.stringify(serverParams)
            for (const key in mcpConfigValues) {
                tmpServerParams = tmpServerParams.replace(new RegExp(`{${key}}`, 'g'), mcpConfigValues[key])
            }
            serverParams = JSON.parse(tmpServerParams)

            const cacheKey = hash(serverParams)

            const cached = mcpToolkitCache.get(cacheKey)
            if (cached?.tools.length) {
                rememberMCPToolkit(cacheKey, cached)
                return cached.tools
            }

            const pending = mcpToolkitInflight.get(cacheKey)
            if (pending) {
                return pending
            }

            const loading = this.connectToolkit(serverParams, cacheKey)
            mcpToolkitInflight.set(cacheKey, loading)
            try {
                return await loading
            } finally {
                mcpToolkitInflight.delete(cacheKey)
            }
        } catch (error) {
            console.error('DynamicCustomMCP failed to load tools:', error)
            return []
        }
    }

    private async connectToolkit(serverParams: any, cacheKey: string): Promise<Tool[]> {
        const toolkit = new MCPToolkit(serverParams, serverParams?.command === undefined ? 'sse' : 'stdio')
        try {
            await toolkit.initialize()
            const tools = (toolkit.tools ?? []).map((t) => wrapToolWithResponseBatching(t as Tool))
            if (!tools.length) {
                await toolkit.close()
                return []
            }
            rememberMCPToolkit(cacheKey, { tools, toolkit })
            return tools
        } catch (error) {
            await toolkit.close()
            throw error
        }
    }
}

function getZodObjectSchema(schema: unknown): z.ZodObject<z.ZodRawShape> {
    if (schema instanceof z.ZodObject) {
        return schema
    }
    if (schema instanceof z.ZodEffects) {
        return getZodObjectSchema(schema._def.schema)
    }
    return z.object({})
}

function wrapToolWithResponseBatching(originalTool: Tool): Tool {
    const originalSchema = getZodObjectSchema(originalTool.schema)
    const extendedSchema = originalSchema.extend({
        responseBatchNumber: z
            .number()
            .int()
            .min(0)
            .optional()
            .default(0)
            .describe(
                `Optional 0-based batch index of the tool response. Large responses are split into ${RESPONSE_BATCH_SIZE}-character batches. Default is 0. Call again with the same arguments and a higher batch number to get the next batch (tool re-executes; response is not cached).`
            )
    })

    return tool(
        async (input): Promise<string> => {
            const { responseBatchNumber = 0, ...toolArgs } = (input ?? {}) as Record<string, any>
            const fullResponse = await originalTool.invoke(toolArgs)
            const responseString = typeof fullResponse === 'string' ? fullResponse : JSON.stringify(fullResponse)

            const totalBatches = Math.max(1, Math.ceil(responseString.length / RESPONSE_BATCH_SIZE))
            const batchNumber = Math.max(0, Math.floor(Number(responseBatchNumber) || 0))

            if (batchNumber >= totalBatches) {
                return JSON.stringify({
                    batchNumber,
                    totalBatches,
                    totalLength: responseString.length,
                    hasMore: false,
                    content: '',
                    message: `Batch ${batchNumber} is out of range. Valid batches are 0 to ${totalBatches - 1}.`
                })
            }

            const start = batchNumber * RESPONSE_BATCH_SIZE
            const content = responseString.slice(start, start + RESPONSE_BATCH_SIZE)

            if (totalBatches === 1) {
                return content
            }

            return JSON.stringify({
                batchNumber,
                totalBatches,
                totalLength: responseString.length,
                hasMore: batchNumber < totalBatches - 1,
                content
            })
        },
        {
            name: originalTool.name,
            description: `${originalTool.description}\n\nNote: Large responses are split into batches of ${RESPONSE_BATCH_SIZE} characters. Use responseBatchNumber (default 0) to request a specific batch. Re-call with the same args and the next batch number for subsequent batches (not cached).`,
            schema: extendedSchema
        }
    ) as unknown as Tool
}

function convertToValidJSONString(inputString: string, mcpConfigValues: ICommonObject) {
    try {
        let tmpServerParams = inputString
        for (const key in mcpConfigValues) {
            tmpServerParams = tmpServerParams.replace(new RegExp(`{${key}}`, 'g'), mcpConfigValues[key])
        }
        const jsObject = Function('return ' + tmpServerParams)()
        return JSON.stringify(jsObject, null, 2)
    } catch (error) {
        console.error('Error converting to JSON:', error)
        return ''
    }
}

module.exports = { nodeClass: DynamicCustom_MCP }
