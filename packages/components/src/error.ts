type ErrorWithMessage = {
    message: string
}

const MAX_DEPTH = 6
const MAX_TEXT = 2000

const clip = (text: string): string => {
    if (text.length <= MAX_TEXT) return text
    return `${text.slice(0, MAX_TEXT)}…`
}

const readString = (value: unknown): string => (typeof value === 'string' ? value : '')

const apiErrorText = (value: unknown, depth = 0): string => {
    if (value == null || depth > 4) return ''
    if (typeof value === 'string') return clip(value)
    if (typeof value !== 'object') return ''

    const obj = value as Record<string, unknown>
    const type = readString(obj.type)
    const code = obj.code != null && obj.code !== '' && typeof obj.code !== 'object' ? String(obj.code) : ''
    const message = readString(obj.message)
    const nested = apiErrorText(obj.error, depth + 1)
    const self = [type && type !== 'error' ? type : '', code ? `code=${code}` : '', message].filter(Boolean).join(' ')

    if (nested && nested !== message && !self.includes(nested)) {
        return clip([self, nested].filter(Boolean).join(' | '))
    }
    return clip(self || nested)
}

const summarizeError = (error: unknown, depth: number, seen: WeakSet<object>): string => {
    if (error == null || depth > MAX_DEPTH) return ''
    if (typeof error === 'string') return clip(error)
    if (typeof error !== 'object') return clip(String(error))
    if (seen.has(error)) return ''
    seen.add(error)

    const record = error as Record<string, unknown>
    const name = readString(record.name)
    const message = readString(record.message)
    const meta: string[] = []
    const status = typeof record.status === 'number' ? record.status : typeof record.statusCode === 'number' ? record.statusCode : undefined
    if (status != null) meta.push(`status=${status}`)
    if (record.code != null && record.code !== '' && typeof record.code !== 'object') meta.push(`code=${String(record.code)}`)
    if (typeof record.errno === 'number' || typeof record.errno === 'string') meta.push(`errno=${String(record.errno)}`)
    if (typeof record.syscall === 'string') meta.push(`syscall=${record.syscall}`)
    if (typeof record.hostname === 'string') meta.push(`hostname=${record.hostname}`)
    if (typeof record.address === 'string') meta.push(`address=${record.address}`)
    if (typeof record.port === 'number') meta.push(`port=${record.port}`)
    const requestId = readString(record.requestID) || readString(record.request_id) || readString(record.requestId)
    if (requestId) meta.push(`requestId=${requestId}`)

    const api = apiErrorText(record.error)
    let responseBody = ''
    if (record.response && typeof record.response === 'object') {
        const response = record.response as Record<string, unknown>
        if (status == null && typeof response.status === 'number') meta.push(`status=${response.status}`)
        responseBody = apiErrorText(response.data ?? response.body)
    }
    const headCore = [name && name !== 'Error' ? name : '', message].filter(Boolean).join(': ')
    let head = headCore || 'Unknown error'
    if (meta.length) head += ` (${meta.join(', ')})`
    if (api && api !== message && !head.includes(api)) head += ` api=${api}`
    if (responseBody && responseBody !== message && responseBody !== api && !head.includes(responseBody)) {
        head += ` response=${responseBody}`
    }

    const parts = [clip(head)]
    if (record.cause) {
        const cause = summarizeError(record.cause, depth + 1, seen)
        if (cause) parts.push(cause)
    }
    if (Array.isArray(record.errors)) {
        for (const nested of record.errors) {
            const text = summarizeError(nested, depth + 1, seen)
            if (text) parts.push(text)
        }
    }
    return parts.join(' | caused by: ')
}

const topLevelMessage = (error: unknown): string => {
    if (typeof error === 'string') return error
    if (typeof error === 'object' && error !== null && 'message' in error && typeof (error as ErrorWithMessage).message === 'string') {
        return (error as ErrorWithMessage).message
    }
    return ''
}

export const getErrorMessage = (error: unknown): string => {
    try {
        return summarizeError(error, 0, new WeakSet()) || 'Unknown error'
    } catch {
        return topLevelMessage(error) || String(error)
    }
}

/** Message, cause chain, status/code, and stack for server logs. */
export const formatErrorForLog = (error: unknown): string => {
    const summary = getErrorMessage(error)
    const stack = error instanceof Error && typeof error.stack === 'string' ? error.stack.split('\n').slice(0, 40).join('\n') : ''
    if (!stack) return summary
    if (summary && stack.includes(summary)) return stack
    return `${summary}\n${stack}`
}

/** Log an LLM failure with the cause chain. Falls back to console when no logger is on the node options. */
export const logLLMError = (logger: { error?: (message: string) => void } | undefined, context: string, error: unknown): void => {
    const line = `[llm] ${context}\n${formatErrorForLog(error)}`
    if (logger && typeof logger.error === 'function') {
        logger.error(line)
    } else {
        console.error(line)
    }
}

/** True when an AbortController / LangChain / OpenAI abort cancelled the run */
export const isAbortError = (error: unknown, signal?: AbortSignal): boolean => {
    if (signal?.aborted) return true
    if (!error) return false

    const name =
        typeof error === 'object' && error !== null && 'name' in error ? String((error as { name?: unknown }).name) : ''
    const ctor =
        typeof error === 'object' && error !== null && 'constructor' in error
            ? String((error as { constructor?: { name?: unknown } }).constructor?.name ?? '')
            : ''

    // OpenAI: APIUserAbortError (name is often just "Error"), DOM: AbortError
    if (name === 'AbortError' || ctor === 'AbortError' || ctor === 'APIUserAbortError') return true

    const message = topLevelMessage(error).toLowerCase()
    return message.includes('aborted') || message.includes('abort error')
}
