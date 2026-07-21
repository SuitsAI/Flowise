type ErrorWithMessage = {
    message: string
}

const isErrorWithMessage = (error: unknown): error is ErrorWithMessage => {
    return (
        typeof error === 'object' && error !== null && 'message' in error && typeof (error as Record<string, unknown>).message === 'string'
    )
}

const toErrorWithMessage = (maybeError: unknown): ErrorWithMessage => {
    if (isErrorWithMessage(maybeError)) return maybeError

    try {
        return new Error(JSON.stringify(maybeError))
    } catch {
        // fallback in case there's an error stringifying the maybeError
        // like with circular references for example.
        return new Error(String(maybeError))
    }
}

export const getErrorMessage = (error: unknown) => {
    return toErrorWithMessage(error).message
}

/** True when an AbortController / LangChain abort cancelled the run */
export const isAbortError = (error: unknown): boolean => {
    if (!error) return false
    const name = typeof error === 'object' && error !== null && 'name' in error ? String((error as { name?: unknown }).name) : ''
    if (name === 'AbortError') return true
    return getErrorMessage(error).includes('Aborted')
}
