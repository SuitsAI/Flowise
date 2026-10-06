import os from 'os'
import { MCPToolkit } from './core'

// Separate process so a multi-minute MCP tool cannot block Flowise's event loop.
// Lower priority than the server: on a busy CPU the API keeps getting time.
try {
    os.setPriority(process.pid, 10)
} catch {
    // Not supported in this environment.
}

type RunnerMessage = {
    id: number
    type: 'list' | 'call'
    serverParams: any
    transportType: 'stdio' | 'sse'
    name?: string
    args?: Record<string, unknown>
}

const toolkits = new Map<string, MCPToolkit>()
const inflight = new Map<string, Promise<MCPToolkit>>()

function cacheKey(serverParams: any, transportType: string): string {
    return JSON.stringify({ transportType, serverParams })
}

async function getToolkit(serverParams: any, transportType: 'stdio' | 'sse'): Promise<MCPToolkit> {
    const key = cacheKey(serverParams, transportType)
    const cached = toolkits.get(key)
    if (cached) return cached
    const pending = inflight.get(key)
    if (pending) return pending
    const loading = (async () => {
        const toolkit = new MCPToolkit(serverParams, transportType)
        await toolkit.initialize()
        toolkits.set(key, toolkit)
        return toolkit
    })()
    inflight.set(key, loading)
    try {
        return await loading
    } finally {
        inflight.delete(key)
    }
}

process.on('message', (message: RunnerMessage) => {
    void (async () => {
        try {
            const toolkit = await getToolkit(message.serverParams, message.transportType)
            if (message.type === 'list') {
                process.send?.({ id: message.id, result: JSON.parse(JSON.stringify(toolkit._tools)) })
                return
            }
            const result = await toolkit.callTool(message.name || '', message.args || {})
            process.send?.({ id: message.id, result })
        } catch (error) {
            process.send?.({
                id: message.id,
                error: error instanceof Error ? error.message : String(error)
            })
        }
    })()
})

process.on('disconnect', () => {
    process.exit(0)
})
