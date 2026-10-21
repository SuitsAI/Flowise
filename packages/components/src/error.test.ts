import { formatErrorForLog, getErrorMessage, isAbortError } from './error'

describe('getErrorMessage', () => {
    it('includes the undici fetch cause instead of only "fetch failed"', () => {
        const cause = Object.assign(new Error('Connect Timeout Error'), {
            name: 'ConnectTimeoutError',
            code: 'UND_ERR_CONNECT_TIMEOUT',
            hostname: 'api.anthropic.com'
        })
        const error = new TypeError('fetch failed')
        ;(error as TypeError & { cause?: unknown }).cause = cause

        const message = getErrorMessage(error)

        expect(message).toContain('fetch failed')
        expect(message).toContain('ConnectTimeoutError')
        expect(message).toContain('UND_ERR_CONNECT_TIMEOUT')
        expect(message).toContain('api.anthropic.com')
        expect(formatErrorForLog(error)).toContain('fetch failed')
    })

    it('includes API status and error body', () => {
        const error = Object.assign(new Error('invalid model'), {
            status: 400,
            error: { type: 'invalid_request_error', message: 'model not found' },
            requestID: 'req_123'
        })

        const message = getErrorMessage(error)

        expect(message).toContain('status=400')
        expect(message).toContain('invalid_request_error')
        expect(message).toContain('model not found')
        expect(message).toContain('requestId=req_123')
    })

    it('does not treat a fetch failure as an abort', () => {
        const error = new TypeError('fetch failed')
        ;(error as TypeError & { cause?: unknown }).cause = new Error('socket hang up')
        expect(isAbortError(error)).toBe(false)
    })

    it('still detects abort errors from the top-level message', () => {
        expect(isAbortError(new Error('Aborted'))).toBe(true)
        const abortError = new Error('The operation was aborted')
        abortError.name = 'AbortError'
        expect(isAbortError(abortError)).toBe(true)
    })
})
