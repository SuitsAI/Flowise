/**
 * LangChain's ChatOpenAI delegates requests to inner `responses` and `completions` objects.
 * Each calls its own `_getReasoningParams`, which only allows /^o\\d/ models — so GPT-5 `reasoning`
 * never reaches `responses.create` / chat completions even when Flowise sets `fields.reasoning`.
 * We patch those instances after construction so traces and HTTP payloads match node configuration.
 */

export function openAIReasoningParamsApplyToModel(model: string | undefined): boolean {
    if (!model) return false
    if (/^o\d/i.test(model)) return true
    return /^gpt-5/i.test(model)
}

/* eslint-disable @typescript-eslint/no-explicit-any */
export function flowiseMergeOpenAIReasoningParams(
    instance: { model?: string; reasoning?: any },
    options?: any
): Record<string, unknown> | undefined {
    if (!openAIReasoningParamsApplyToModel(instance.model)) {
        return undefined
    }
    let reasoning: Record<string, unknown> | undefined
    if (instance.reasoning !== undefined) {
        reasoning = {
            ...reasoning,
            ...(instance.reasoning as Record<string, unknown>)
        }
    }
    if (options?.reasoning !== undefined) {
        reasoning = {
            ...reasoning,
            ...options.reasoning
        }
    }
    return reasoning
}
/* eslint-enable @typescript-eslint/no-explicit-any */

/** Mutates nested LangChain delegates created in ChatOpenAI / AzureChatOpenAI constructors. */
export function patchInnerOpenAIReasoningDelegates(llm: object): void {
    for (const key of ['responses', 'completions'] as const) {
        const inner = (llm as Record<string, unknown>)[key]
        if (!inner || typeof inner !== 'object') continue
        const bearer = inner as { model?: string; reasoning?: unknown; _getReasoningParams?: (opts: unknown) => unknown }
        bearer._getReasoningParams = function flowisePatchedReasoningParams(this: { model?: string; reasoning?: unknown }, options?: unknown) {
            return flowiseMergeOpenAIReasoningParams(this, options)
        }
    }
}
