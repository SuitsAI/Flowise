import { Tool } from '@langchain/core/tools'
import { INode, INodeData, INodeOptionsValue, INodeParams } from '../../../../src/Interface'
import { MCPToolkit } from '../core'

const mcpServerConfig = `{
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
}`

class Custom_MCP implements INode {
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
    toolkit: MCPToolkit | null = null

    constructor() {
        this.label = 'Custom MCP'
        this.name = 'customMCP'
        this.version = 1.0
        this.type = 'Custom MCP Tool'
        this.icon = 'customMCP.png'
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
                label: 'Available Actions',
                name: 'mcpActions',
                type: 'asyncMultiOptions',
                loadMethod: 'listActions',
                refresh: true
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
        // Clean up previous toolkit instance if it exists
        if (this.toolkit) {
            if (this.toolkit.client) {
                try {
                    // Close the client connection if possible
                    if (typeof this.toolkit.client.close === 'function') {
                        await this.toolkit.client.close()
                    }
                } catch (error) {
                    console.error('Error closing MCP client:', error)
                }
            }
            // Clear the toolkit properties
            this.toolkit = null
        }

        const toolkit = await this.createToolkit(nodeData)
        this.toolkit = toolkit

        const tools = await toolkit.getTools()

        const _mcpActions = nodeData.inputs?.mcpActions
        let mcpActions = []
        if (_mcpActions) {
            try {
                mcpActions = typeof _mcpActions === 'string' ? JSON.parse(_mcpActions) : _mcpActions
            } catch (error) {
                mcpActions = (_mcpActions || []).split(',').map((action: string) => action.trim())
            }
        }

        return tools.filter((tool: any) => mcpActions.includes(tool.name))
    }

    async getTools(nodeData: INodeData): Promise<Tool[]> {
        const toolkit = await this.createToolkit(nodeData)
        // Don't store this toolkit since it's just for listing actions
        const tools = toolkit.tools ?? []
        
        // Clean up the temporary toolkit after use
        if (toolkit.client) {
            try {
                if (typeof toolkit.client.close === 'function') {
                    await toolkit.client.close()
                }
            } catch (error) {
                console.error('Error closing temporary MCP client:', error)
            }
        }
        
        return tools as Tool[]
    }

    async createToolkit(nodeData: INodeData): Promise<MCPToolkit> {
        const mcpServerConfig = nodeData.inputs?.mcpServerConfig as string

        if (!mcpServerConfig) {
            throw new Error('MCP Server Config is required')
        }

        try {
            let serverParams
            if (typeof mcpServerConfig === 'object') {
                serverParams = mcpServerConfig
            } else if (typeof mcpServerConfig === 'string') {
                const serverParamsString = convertToValidJSONString(mcpServerConfig)
                serverParams = JSON.parse(serverParamsString)
            }

            // Compatible with stdio and SSE
            let toolkit: MCPToolkit
            if (serverParams?.command === undefined) {
                toolkit = new MCPToolkit(serverParams, 'sse')
            } else {
                toolkit = new MCPToolkit(serverParams, 'stdio')
            }

            await toolkit.initialize()
            return toolkit
        } catch (error) {
            throw new Error(`Invalid MCP Server Config: ${error}`)
        }
    }

    async cleanup(): Promise<void> {
        // Clean up toolkit when this node is no longer needed
        if (this.toolkit) {
            if (this.toolkit.client) {
                try {
                    // Close the client connection if possible
                    if (typeof this.toolkit.client.close === 'function') {
                        await this.toolkit.client.close()
                    }
                } catch (error) {
                    console.error('Error closing MCP client during cleanup:', error)
                }
            }
            this.toolkit = null
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

module.exports = { nodeClass: Custom_MCP }
