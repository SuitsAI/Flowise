import { ChildProcess, fork } from 'child_process'
import fs from 'fs'
import path from 'path'

type Pending = {
    resolve: (value: any) => void
    reject: (error: Error) => void
}

let child: ChildProcess | null = null
let nextId = 0
let exitHooked = false
const pending = new Map<number, Pending>()

function runnerLaunch(): { runner: string; execArgv: string[] } {
    const jsPath = path.join(__dirname, 'mcpRunner.js')
    if (fs.existsSync(jsPath)) {
        return { runner: jsPath, execArgv: [] }
    }
    let register = 'ts-node/register/transpile-only'
    try {
        register = require.resolve('ts-node/register/transpile-only')
    } catch {
        // Fall back to the bare specifier so Node can resolve it from the repo.
    }
    return { runner: path.join(__dirname, 'mcpRunner.ts'), execArgv: ['-r', register] }
}

function failPending(error: Error) {
    for (const waiter of pending.values()) {
        waiter.reject(error)
    }
    pending.clear()
}

function ensureChild(): ChildProcess {
    if (child && child.connected) return child
    const { runner, execArgv } = runnerLaunch()
    child = fork(runner, [], {
        execArgv,
        // ignore keeps a chatty runner from filling this process's stderr pipe and stalling every request
        stdio: ['ignore', 'ignore', 'ignore', 'ipc'],
        env: { ...process.env, FLOWISE_MCP_RUNNER: '1' }
    })
    child.on('message', (message: { id?: number; result?: any; error?: string }) => {
        if (message?.id == null) return
        const waiter = pending.get(message.id)
        if (!waiter) return
        pending.delete(message.id)
        if (message.error) waiter.reject(new Error(message.error))
        else waiter.resolve(message.result)
    })
    child.on('exit', (code, signal) => {
        child = null
        failPending(new Error(`MCP runner exited (${signal || code || 'unknown'})`))
    })
    child.on('error', (error) => {
        child = null
        failPending(error instanceof Error ? error : new Error(String(error)))
    })
    // The IPC channel must not keep the server running after it has stopped.
    child.unref()
    child.channel?.unref()
    if (!exitHooked) {
        exitHooked = true
        process.once('exit', () => {
            if (child?.connected) child.kill()
        })
    }
    return child
}

export function callMcpRunner(request: Record<string, unknown>): Promise<any> {
    const proc = ensureChild()
    const id = ++nextId
    return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject })
        proc.send({ id, ...request }, (error) => {
            if (!error) return
            pending.delete(id)
            reject(error instanceof Error ? error : new Error(String(error)))
        })
    })
}
