/**
 * Anthropic API compatibility helpers for models with breaking behavior changes
 * (Claude Sonnet 5, Opus 4.7+, etc.).
 * @see https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5
 */

/** Models that reject non-default temperature, top_p, and top_k. */
export function rejectsSamplingParams(modelName: string): boolean {
    const normalized = modelName.trim().toLowerCase()
    if (normalized === 'claude-sonnet-5') return true
    if (normalized === 'claude-opus-4-8' || normalized.startsWith('claude-opus-4-8')) return true
    if (normalized === 'claude-opus-4-7' || normalized.startsWith('claude-opus-4-7')) return true
    return false
}

/** Models that removed manual extended thinking (`budget_tokens`); use adaptive/disabled instead. */
export function requiresAdaptiveThinkingApi(modelName: string): boolean {
    return rejectsSamplingParams(modelName)
}

export function stripSamplingParams(params: Record<string, unknown>): void {
    delete params['temperature']
    delete params['top_p']
    delete params['top_k']
}

/** Valid values for the `effort` parameter, ordered from least to most token/latency spend. */
export const EFFORT_VALUES = ['low', 'medium', 'high', 'xhigh', 'max'] as const
export type AnthropicEffort = (typeof EFFORT_VALUES)[number]

/**
 * Models that support the top-level `effort` parameter (tunes intelligence vs. token spend).
 * @see https://platform.claude.com/docs/en/build-with-claude/effort
 */
export function supportsEffort(modelName: string): boolean {
    const normalized = modelName.trim().toLowerCase()
    if (normalized === 'claude-sonnet-5' || normalized.startsWith('claude-sonnet-5')) return true
    if (normalized === 'claude-sonnet-4-6' || normalized.startsWith('claude-sonnet-4-6')) return true
    if (normalized === 'claude-opus-4-8' || normalized.startsWith('claude-opus-4-8')) return true
    if (normalized === 'claude-opus-4-7' || normalized.startsWith('claude-opus-4-7')) return true
    if (normalized === 'claude-opus-4-6' || normalized.startsWith('claude-opus-4-6')) return true
    if (normalized === 'claude-opus-4-5' || normalized.startsWith('claude-opus-4-5')) return true
    return false
}

export type AnthropicThinkingConfig =
    | { type: 'enabled'; budget_tokens: number }
    | { type: 'adaptive'; display: 'summarized' }
    | { type: 'disabled' }

/** Build the `thinking` payload for a model + Extended Thinking toggle. */
export function buildThinkingConfig(modelName: string, extendedThinking: boolean, budgetTokens: string): AnthropicThinkingConfig | undefined {
    if (requiresAdaptiveThinkingApi(modelName)) {
        if (extendedThinking) {
            // Sonnet 5 / Opus 4.7+ default to display:"omitted" (no readable thinking in stream).
            // Flowise Extended Thinking toggle expects llmReasoning SSE — opt in to summarized text.
            return { type: 'adaptive', display: 'summarized' }
        }
        return { type: 'disabled' }
    }
    if (!extendedThinking) return undefined
    return {
        type: 'enabled',
        budget_tokens: parseInt(budgetTokens, 10) || 1024
    }
}
