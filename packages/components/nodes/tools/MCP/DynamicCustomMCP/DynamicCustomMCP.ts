import { Tool } from '@langchain/core/tools'
import { ICommonObject, INode, INodeData, INodeOptionsValue, INodeParams } from '../../../../src/Interface'
import { MCPToolkit } from '../core'

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
        listActions: async (nodeData: INodeData): Promise<INodeOptionsValue[]> => {
            try {
                const toolset = await this.getTools(nodeData)
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

    async init(nodeData: INodeData): Promise<any> {
        const tools = await this.getTools(nodeData)
        return tools
    }

    async getTools(nodeData: INodeData): Promise<Tool[]> {

        const mcpConfigValuesStr = nodeData.inputs?.mcpConfigValues
        let mcpConfigValues: ICommonObject = {}
        if (mcpConfigValuesStr) {
            try {
                mcpConfigValues = typeof mcpConfigValuesStr === 'object' ? mcpConfigValuesStr : JSON.parse(mcpConfigValuesStr)
            } catch (exception) {
                throw new Error("Invalid JSON in the DynamicCustomMCP's mcpConfigValues: " + exception)
            }
        }

        const mcpServerConfig = nodeData.inputs?.mcpServerConfig as string

        if (!mcpServerConfig) {
            throw new Error('MCP Server Config is required')
        }

        console.log('mcpConfigValues', mcpConfigValues)

        try {
            let serverParams
            if (typeof mcpServerConfig === 'object') {
                serverParams = mcpServerConfig
            } else if (typeof mcpServerConfig === 'string') {
                const serverParamsString = convertToValidJSONString(mcpServerConfig)
                serverParams = JSON.parse(serverParamsString)
            }

            let tmpServerParams = JSON.stringify(serverParams);
            for (const key in mcpConfigValues) {
                tmpServerParams = tmpServerParams.replace(new RegExp(`{${key}}`, 'g'), mcpConfigValues[key])
            }
            serverParams = JSON.parse(tmpServerParams)

            console.log('serverParams', serverParams)

            // Compatible with stdio and SSE
            let toolkit: MCPToolkit
            if (serverParams?.command === undefined) {
                toolkit = new MCPToolkit(serverParams, 'sse')
            } else {
                toolkit = new MCPToolkit(serverParams, 'stdio')
            }

            await toolkit.initialize()

            const tools = toolkit.tools ?? []

            return tools as Tool[]
        } catch (error) {
            throw new Error(`Invalid MCP Server Config: ${error}`)
        }
    }
}

function convertToValidJSONString(inputString: string) {
    try {
        const jsObject = Function('return ' + inputString)()
        return JSON.stringify(jsObject, null, 2)
    } catch (error) {
        console.error('Error converting to JSON:', error)
        return ''
    }
}

module.exports = { nodeClass: DynamicCustom_MCP }