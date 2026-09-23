import {
    buildThinkingConfig,
    rejectsSamplingParams,
    requiresAdaptiveThinkingApi,
    requiresAlwaysOnThinking,
    stripSamplingParams,
    supportsEffort
} from './anthropicModelCompat'

describe('anthropicModelCompat', () => {
    describe('rejectsSamplingParams', () => {
        it.each([
            'claude-sonnet-5',
            'claude-opus-5-5',
            'claude-opus-5',
            'claude-opus-4-8',
            'claude-opus-4-7',
            'claude-opus-4-7-20251101'
        ])('returns true for %s', (model) => {
            expect(rejectsSamplingParams(model)).toBe(true)
        })

        it.each([
            'claude-sonnet-4-6',
            'claude-sonnet-4-5',
            'claude-opus-4-6',
            'claude-3-haiku',
            'claude-3-5-sonnet-latest'
        ])('returns false for %s', (model) => {
            expect(rejectsSamplingParams(model)).toBe(false)
        })
    })

    describe('buildThinkingConfig', () => {
        it('uses adaptive/disabled for Sonnet 5', () => {
            expect(buildThinkingConfig('claude-sonnet-5', true, '1024')).toEqual({
                type: 'adaptive',
                display: 'summarized'
            })
            expect(buildThinkingConfig('claude-sonnet-5', false, '1024')).toEqual({ type: 'disabled' })
        })

        it('uses adaptive/disabled for Opus 5', () => {
            expect(buildThinkingConfig('claude-opus-5', true, '1024')).toEqual({
                type: 'adaptive',
                display: 'summarized'
            })
            expect(buildThinkingConfig('claude-opus-5', false, '1024')).toEqual({ type: 'disabled' })
        })

        it('never disables thinking on Opus 5.5', () => {
            expect(buildThinkingConfig('claude-opus-5-5', true, '1024')).toEqual({
                type: 'adaptive',
                display: 'summarized'
            })
            expect(buildThinkingConfig('claude-opus-5-5', false, '1024')).toBeUndefined()
        })

        it('uses manual extended thinking for older Sonnet models when enabled', () => {
            expect(buildThinkingConfig('claude-sonnet-4-6', true, '2048')).toEqual({
                type: 'enabled',
                budget_tokens: 2048
            })
            expect(buildThinkingConfig('claude-sonnet-4-6', false, '2048')).toBeUndefined()
        })
    })

    describe('stripSamplingParams', () => {
        it('removes temperature, top_p, and top_k', () => {
            const params = { temperature: 0.9, top_p: 0.5, top_k: 40, max_tokens: 1024 }
            stripSamplingParams(params)
            expect(params).toEqual({ max_tokens: 1024 })
        })
    })

    it('requiresAdaptiveThinkingApi matches rejectsSamplingParams', () => {
        expect(requiresAdaptiveThinkingApi('claude-sonnet-5')).toBe(true)
        expect(requiresAdaptiveThinkingApi('claude-opus-5-5')).toBe(true)
        expect(requiresAdaptiveThinkingApi('claude-opus-5')).toBe(true)
        expect(requiresAdaptiveThinkingApi('claude-sonnet-4-6')).toBe(false)
    })

    it('requiresAlwaysOnThinking is true only for Opus 5.5', () => {
        expect(requiresAlwaysOnThinking('claude-opus-5-5')).toBe(true)
        expect(requiresAlwaysOnThinking('claude-opus-5')).toBe(false)
        expect(requiresAlwaysOnThinking('claude-sonnet-5')).toBe(false)
    })

    describe('supportsEffort', () => {
        it.each([
            'claude-sonnet-5',
            'claude-sonnet-4-6',
            'claude-opus-5-5',
            'claude-opus-5',
            'claude-opus-4-8',
            'claude-opus-4-7',
            'claude-opus-4-6',
            'claude-opus-4-5',
            'claude-opus-4-7-20251101'
        ])('returns true for %s', (model) => {
            expect(supportsEffort(model)).toBe(true)
        })

        it.each(['claude-sonnet-4-5', 'claude-opus-4-1', 'claude-3-haiku', 'claude-3-5-sonnet-latest'])(
            'returns false for %s',
            (model) => {
                expect(supportsEffort(model)).toBe(false)
            }
        )
    })
})
