import type { AIMessageChunk } from '@langchain/core/messages'

function isReasoningBlockType(type: unknown): boolean {
    return (
        type === 'reasoning' ||
        type === 'thinking' ||
        type === 'thinking_delta' ||
        type === 'reasoning_delta' ||
        type === 'redacted_thinking'
    )
}

function reasoningTextFromBlock(block: Record<string, unknown>): string {
    if (typeof block.reasoning === 'string') return block.reasoning
    if (typeof block.thinking === 'string') return block.thinking
    if (typeof block.summary === 'string') return block.summary
    if (Array.isArray(block.summary)) {
        const summaryText = (block.summary as Record<string, unknown>[])
            .map((s) => (typeof s?.text === 'string' ? s.text : ''))
            .join('')
        if (summaryText) return summaryText
    }
    // Anthropic/LC sometimes keep the API delta subtype on the block before normalization
    if (typeof (block as { thinking_delta?: string }).thinking_delta === 'string')
        return (block as { thinking_delta: string }).thinking_delta
    if (typeof block.text === 'string') return block.text
    return ''
}

function appendFromContentBlocks(messageChunk: AIMessageChunk | undefined, reasoningDelta: string, textDelta: string): { r: string; t: string } {
    let r = reasoningDelta
    let t = textDelta
    const blocks = (messageChunk as { contentBlocks?: unknown })?.contentBlocks
    if (!Array.isArray(blocks)) return { r, t }
    for (const block of blocks as Record<string, unknown>[]) {
        if (!block || typeof block !== 'object') continue
        const type = block.type
        if (isReasoningBlockType(type)) {
            const piece = reasoningTextFromBlock(block)
            if (piece) r += piece
        } else if (type === 'text') {
            if (typeof block.text === 'string') t += block.text
        }
    }
    return { r, t }
}

/**
 * Splits streamed model output into internal reasoning vs user-visible answer text.
 * Covers OpenAI (reasoning_content), Anthropic extended thinking (thinking / thinking_delta blocks
 * when LangChain uses array content — see @langchain/anthropic _streamResponseChunks), and LC contentBlocks.
 */
export function extractLLMStreamDeltas(
    messageChunk: AIMessageChunk | undefined,
    token: string
): { reasoningDelta: string; textDelta: string } {
    let reasoningDelta = ''
    let textDelta = ''

    if (messageChunk) {
        const kwargs = messageChunk.additional_kwargs as Record<string, unknown> | undefined
        const rc = kwargs?.reasoning_content
        if (typeof rc === 'string' && rc.length > 0) {
            reasoningDelta += rc
        }
        const reasoningKwargs = kwargs?.reasoning as Record<string, unknown> | undefined
        if (typeof reasoningKwargs?.summary === 'string') {
            reasoningDelta += reasoningKwargs.summary
        } else if (Array.isArray(reasoningKwargs?.summary)) {
            reasoningDelta += (reasoningKwargs.summary as Record<string, unknown>[])
                .map((s) => (typeof s?.text === 'string' ? s.text : ''))
                .join('')
        }

        const c = messageChunk.content
        if (Array.isArray(c)) {
            for (const block of c as Record<string, unknown>[]) {
                if (!block || typeof block !== 'object') continue
                const type = block.type
                if (isReasoningBlockType(type)) {
                    // Skip signature-only chunks from Anthropic (opaque, not human-readable "thinking")
                    if (typeof block.signature === 'string' && !reasoningTextFromBlock(block)) {
                        continue
                    }
                    reasoningDelta += reasoningTextFromBlock(block)
                } else if (type === 'text') {
                    textDelta += typeof block.text === 'string' ? block.text : ''
                }
            }
        } else if (typeof c === 'string' && c.length > 0) {
            textDelta += c
        }

        // Prefer `content` when LangChain fills it (Anthropic extended thinking uses array blocks).
        // Only fall back to `contentBlocks` when content is empty or not an array (some adapters).
        if (!Array.isArray(c) || c.length === 0) {
            const merged = appendFromContentBlocks(messageChunk, reasoningDelta, textDelta)
            reasoningDelta = merged.r
            textDelta = merged.t
        }
    }

    if (!textDelta && token) {
        textDelta = token
    }

    return { reasoningDelta, textDelta }
}

/** Collapses multimodal content to answer text only (excludes reasoning/thinking blocks). */
export function getAnswerTextFromMessageContent(content: unknown): string {
    if (typeof content === 'string') {
        return content
    }
    if (!Array.isArray(content)) {
        return ''
    }
    const parts: string[] = []
    for (const block of content as Record<string, unknown>[]) {
        if (!block || typeof block !== 'object') continue
        if (block.type === 'text' && typeof block.text === 'string') {
            parts.push(block.text)
        }
    }
    return parts.join('')
}
