import { AnthropicInput, ChatAnthropic as LangchainChatAnthropic } from '@langchain/anthropic'
import { type BaseChatModelParams } from '@langchain/core/language_models/chat_models'
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import { type BaseMessage, type MessageContentComplex } from '@langchain/core/messages'
import { type ChatResult, type ChatGenerationChunk } from '@langchain/core/outputs'
import { IVisionChatModal, IMultiModalOption } from '../../../src'

const DEFAULT_IMAGE_MODEL = 'claude-3-5-haiku-latest'
const DEFAULT_IMAGE_MAX_TOKEN = 2048
const CACHE_CONTROL = { type: 'ephemeral' as const, ttl: '1h' as const }
/** Put this in your system prompt to mark the cache boundary: only the text *before* it is cached. Put dynamic content (e.g. date) *after* it. */
const CACHE_BOUNDARY_MARKER = '\n---\n'
/** Anthropic requires non-empty content for all messages (except optional final assistant). Use a single space to avoid 400. */
const EMPTY_CONTENT_PLACEHOLDER = ' '

export class ChatAnthropic extends LangchainChatAnthropic implements IVisionChatModal {
    configuredModel: string
    configuredMaxToken: number
    multiModalOption: IMultiModalOption
    id: string
    promptCaching: boolean

    constructor(id: string, fields?: Partial<AnthropicInput> & BaseChatModelParams, promptCaching?: boolean) {
        // @ts-ignore
        super(fields ?? {})
        this.id = id
        this.configuredModel = fields?.modelName || ''
        this.configuredMaxToken = fields?.maxTokens ?? 2048
        this.promptCaching = promptCaching ?? false
    }

    revertToOriginalModel(): void {
        this.modelName = this.configuredModel
        this.maxTokens = this.configuredMaxToken
    }

    setMultiModalOption(multiModalOption: IMultiModalOption): void {
        this.multiModalOption = multiModalOption
    }

    setVisionModel(): void {
        if (!this.modelName.startsWith('claude-3')) {
            this.modelName = DEFAULT_IMAGE_MODEL
            this.maxTokens = this.configuredMaxToken ? this.configuredMaxToken : DEFAULT_IMAGE_MAX_TOKEN
        }
    }

    /**
     * Override invocationParams to sanitize Langchain's sentinel defaults before sending to the API.
     *
     * Langchain's ChatAnthropic defaults topP and topK to -1 as "unset" sentinels, but the
     * Anthropic API rejects -1 on newer models (e.g. claude-sonnet-4-6). Additionally, some
     * models reject requests where both temperature and top_p are present simultaneously.
     *
     * Rules applied here:
     *  - Strip top_p / top_k when they are -1 (Langchain's unset sentinel).
     *  - If a valid top_p (0–1) is being sent, also strip temperature (mutually exclusive on newer models).
     */
    // @ts-ignore – return type intentionally widened; base class uses a complex intersection type that varies across langchain versions
    invocationParams(options?: this['ParsedCallOptions']): Record<string, unknown> {
        const params = super.invocationParams(options) as Record<string, unknown>

        const topP = params['top_p']
        const topK = params['top_k']

        // Remove sentinel -1 values that Langchain uses to mean "not set"
        if (topP === -1 || topP === undefined) delete params['top_p']
        if (topK === -1 || topK === undefined) delete params['top_k']

        // If a valid top_p is being sent, temperature must be omitted (mutually exclusive on newer models)
        if (typeof params['top_p'] === 'number') delete params['temperature']

        // Langchain drops context_management from invocationParams when extended thinking is enabled.
        // Re-add it so compaction still works when thinking and compaction are both turned on.
        if (params['context_management'] === undefined && (this as any).contextManagement) {
            params['context_management'] = (this as any).contextManagement
        }

        return params
    }

    async _generate(
        messages: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun
    ): Promise<ChatResult> {
        if (!this.promptCaching) {
            return super._generate(messages, options, runManager)
        }

        const modifiedMessages = this._injectCacheControl(messages)
        const result = await super._generate(modifiedMessages, options, runManager)
        this._logCacheDebug(modifiedMessages, result)
        this._enrichUsageWithCacheBreakdown(result)
        return result
    }

    /**
     * Streaming path: when the client requests streaming, the chain calls _streamResponseChunks
     * instead of _generate. We must inject cache_control here too so prompt caching works when streaming.
     */
    async *_streamResponseChunks(
        messages: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun
    ): AsyncGenerator<ChatGenerationChunk> {
        if (!this.promptCaching) {
            yield* super._streamResponseChunks(messages, options, runManager)
            return
        }
        const modifiedMessages = this._injectCacheControl(messages)
        yield* super._streamResponseChunks(modifiedMessages, options, runManager)
    }

    /** Logs cache-related state when prompt caching is enabled (visible in Flowise server logs). */
    private _logCacheDebug(modifiedMessages: BaseMessage[], result: ChatResult): void {
        const systemMsg = modifiedMessages.find((m) => (m as any)._getType?.() === 'system' || (m as any).constructor?.name === 'SystemMessage')
        const systemContent = systemMsg?.content
        const hasCacheControlOnSystem =
            Array.isArray(systemContent) &&
            (systemContent as any[]).some((b: any) => b?.cache_control?.type === 'ephemeral')
        const lastMsg = modifiedMessages[modifiedMessages.length - 1]
        const lastContent = lastMsg?.content
        const hasCacheControlOnLast =
            Array.isArray(lastContent) && (lastContent as any[]).some((b: any) => b?.cache_control?.type === 'ephemeral')
        const gen = result.generations?.[0]?.message as any
        const usage = gen?.response_metadata?.usage ?? gen?.usage_metadata
        const cacheRead = usage?.cache_read_input_tokens ?? usage?.input_token_details?.cache_read ?? 0
        const cacheCreation = usage?.cache_creation_input_tokens ?? usage?.input_token_details?.cache_creation ?? 0
        const payload = {
            cache_control_on_system: hasCacheControlOnSystem,
            cache_control_on_last_message: hasCacheControlOnLast,
            response_has_usage: !!usage,
            response_has_response_metadata_usage: !!(gen?.response_metadata?.usage),
            cache_read_input_tokens: cacheRead,
            cache_creation_input_tokens: cacheCreation,
            input_tokens: usage?.input_tokens
        }
        // Log to stdout so it appears in Flowise server output (no env var needed)
        // eslint-disable-next-line no-console
        console.log('[Flowise Anthropic cache]', JSON.stringify(payload))
    }

    /**
     * Merges Anthropic's cache token counts from response_metadata.usage into
     * message.usage_metadata so LangSmith (and other tracers) can show cache reads/creation
     * and apply correct pricing.
     */
    private _enrichUsageWithCacheBreakdown(result: ChatResult): void {
        for (const gen of result.generations) {
            const msg = gen.message
            const raw = (msg as any).response_metadata?.usage as
                | {
                      input_tokens?: number
                      output_tokens?: number
                      cache_read_input_tokens?: number | null
                      cache_creation_input_tokens?: number | null
                  }
                | undefined
            if (!raw) continue

            const existing = (msg as any).usage_metadata ?? {}
            const cacheRead = raw.cache_read_input_tokens ?? 0
            const cacheCreation = raw.cache_creation_input_tokens ?? 0

            ;(msg as any).usage_metadata = {
                ...existing,
                input_tokens: existing.input_tokens ?? raw.input_tokens,
                output_tokens: existing.output_tokens ?? raw.output_tokens,
                total_tokens:
                    existing.total_tokens ??
                    (existing.input_tokens ?? raw.input_tokens) + (existing.output_tokens ?? raw.output_tokens),
                cache_read_input_tokens: cacheRead,
                cache_creation_input_tokens: cacheCreation,
                input_token_details: {
                    ...(existing.input_token_details ?? {}),
                    cache_read: cacheRead,
                    cache_creation: cacheCreation
                }
            }
        }
    }

    /**
     * Injects cache_control breakpoints into messages for Anthropic prompt caching.
     * - Normalizes empty message content so Anthropic never receives empty content (avoids 400).
     * - System message: cache_control on the *first* content block only.
     * - Last message: cache_control on the last block (caches conversation history up to that point).
     */
    private _injectCacheControl(messages: BaseMessage[]): BaseMessage[] {
        if (messages.length === 0) return messages

        // Clone messages to avoid mutating originals
        const cloned = messages.map((msg) => {
            const clone = msg.constructor
                // @ts-ignore - BaseMessage subclasses accept fields in constructor
                ? new (msg.constructor as any)({ ...msg, content: this._cloneContent(msg.content) })
                : msg
            return clone as BaseMessage
        })

        // Anthropic: "all messages must have non-empty content except for the optional final assistant message"
        this._normalizeEmptyContent(cloned)

        // 1. System message: cache only the first block so dynamic content (e.g. Date.now()) in later blocks doesn't invalidate the cache
        for (let i = cloned.length - 1; i >= 0; i--) {
            if (cloned[i]._getType() === 'system') {
                this._addCacheControlToSystemMessage(cloned[i])
                break
            }
        }

        // 2. Add cache_control to the last message (caches conversation history)
        const lastMessage = cloned[cloned.length - 1]
        if (lastMessage._getType() !== 'system') {
            this._addCacheControlToLastMessage(lastMessage)
        }

        return cloned
    }

    /**
     * Returns true if content is empty or only whitespace (string or block array).
     * Anthropic returns 400 when a message has empty content (except optional final assistant).
     */
    private _isContentEmpty(content: BaseMessage['content']): boolean {
        if (content === undefined || content === null) return true
        if (typeof content === 'string') return content.trim().length === 0
        if (!Array.isArray(content) || content.length === 0) return true
        return content.every((block: any) => {
            if (block?.type === 'text' && typeof block.text === 'string') return block.text.trim().length === 0
            return false
        })
    }

    /**
     * Ensures text is non-empty for API payload. Anthropic rejects empty content blocks.
     */
    private _nonEmptyText(text: string): string {
        return (text ?? '').trim().length > 0 ? text : EMPTY_CONTENT_PLACEHOLDER
    }

    /**
     * Normalizes empty message content in place so every message has non-empty content.
     * Prevents Anthropic 400 "all messages must have non-empty content".
     */
    private _normalizeEmptyContent(messages: BaseMessage[]): void {
        for (const msg of messages) {
            if (!this._isContentEmpty(msg.content)) continue
            const role = (msg as any)._getType?.() ?? (msg as any).constructor?.name ?? ''
            if (role === 'assistant' && msg === messages[messages.length - 1]) continue // optional final assistant may be empty
            ;(msg as any).content = typeof msg.content === 'string' ? EMPTY_CONTENT_PLACEHOLDER : [{ type: 'text', text: EMPTY_CONTENT_PLACEHOLDER }]
        }
    }

    /**
     * Adds cache_control to the stable part of the system message only.
     * - If content is a string containing CACHE_BOUNDARY_MARKER (newline + "---" + newline), we split: only the text *before* the marker gets cache_control; the rest (e.g. dynamic date) is a separate block and does not break the cache.
     * - If content is already multiple blocks, only the first block gets cache_control.
     */
    private _addCacheControlToSystemMessage(message: BaseMessage): void {
        if (typeof message.content === 'string') {
            const idx = message.content.indexOf(CACHE_BOUNDARY_MARKER)
            if (idx !== -1) {
                const stable = message.content.slice(0, idx).trimEnd()
                const dynamic = message.content.slice(idx + CACHE_BOUNDARY_MARKER.length).trim()
                message.content = [
                    { type: 'text', text: this._nonEmptyText(stable), cache_control: CACHE_CONTROL } as MessageContentComplex,
                    ...(dynamic ? [{ type: 'text', text: dynamic } as MessageContentComplex] : [])
                ]
            } else {
                message.content = [
                    { type: 'text', text: this._nonEmptyText(message.content), cache_control: CACHE_CONTROL } as MessageContentComplex
                ]
            }
        } else if (Array.isArray(message.content) && message.content.length > 0) {
            const first = message.content[0] as any
            if (first?.type === 'text' && typeof first.text === 'string') first.text = this._nonEmptyText(first.text)
            ;(message.content[0] as any).cache_control = CACHE_CONTROL
        }
    }

    /** Adds cache_control to the last content block (caches conversation history up to that point). */
    private _addCacheControlToLastMessage(message: BaseMessage): void {
        if (typeof message.content === 'string') {
            message.content = [
                { type: 'text', text: this._nonEmptyText(message.content), cache_control: CACHE_CONTROL } as MessageContentComplex
            ]
        } else if (Array.isArray(message.content) && message.content.length > 0) {
            const lastBlock = message.content[message.content.length - 1] as any
            if (lastBlock?.type === 'text' && typeof lastBlock.text === 'string') lastBlock.text = this._nonEmptyText(lastBlock.text)
            lastBlock.cache_control = CACHE_CONTROL
        }
    }

    /**
     * Deep clones message content to avoid mutating the original messages.
     */
    private _cloneContent(content: BaseMessage['content']): BaseMessage['content'] {
        if (typeof content === 'string') return content
        return content.map((block) => ({ ...block }))
    }
}
