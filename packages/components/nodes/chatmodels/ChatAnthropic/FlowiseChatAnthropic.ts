import { AnthropicInput, ChatAnthropic as LangchainChatAnthropic } from '@langchain/anthropic'
import { type BaseChatModelParams } from '@langchain/core/language_models/chat_models'
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import { type BaseMessage, type MessageContentComplex } from '@langchain/core/messages'
import { type ChatResult } from '@langchain/core/outputs'
import { IVisionChatModal, IMultiModalOption } from '../../../src'

const DEFAULT_IMAGE_MODEL = 'claude-3-5-haiku-latest'
const DEFAULT_IMAGE_MAX_TOKEN = 2048

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

    async _generate(
        messages: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun
    ): Promise<ChatResult> {
        if (!this.promptCaching) {
            return super._generate(messages, options, runManager)
        }

        const modifiedMessages = this._injectCacheControl(messages)
        return super._generate(modifiedMessages, options, runManager)
    }

    /**
     * Injects cache_control breakpoints into messages for Anthropic prompt caching.
     * Adds cache_control to:
     *   1. The last system message (caches the system prompt prefix)
     *   2. The last message overall (caches the full conversation history)
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

        // 1. Find the last system message and add cache_control to its last content block
        for (let i = cloned.length - 1; i >= 0; i--) {
            if (cloned[i]._getType() === 'system') {
                this._addCacheControlToMessage(cloned[i])
                break
            }
        }

        // 2. Add cache_control to the last message (caches conversation history)
        const lastMessage = cloned[cloned.length - 1]
        if (lastMessage._getType() !== 'system') {
            this._addCacheControlToMessage(lastMessage)
        }

        return cloned
    }

    /**
     * Adds cache_control: { type: "ephemeral" } to the last content block of a message.
     * Handles both string and array content formats.
     */
    private _addCacheControlToMessage(message: BaseMessage): void {
        if (typeof message.content === 'string') {
            message.content = [
                {
                    type: 'text',
                    text: message.content,
                    cache_control: { type: 'ephemeral' }
                } as MessageContentComplex
            ]
        } else if (Array.isArray(message.content) && message.content.length > 0) {
            const lastBlock = message.content[message.content.length - 1]
            ;(lastBlock as any).cache_control = { type: 'ephemeral' }
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
