import { AnthropicInput, ChatAnthropic as LangchainChatAnthropic } from '@langchain/anthropic'
import { BaseCache } from '@langchain/core/caches'
import { BaseLLMParams } from '@langchain/core/language_models/llms'
import { ICommonObject, IMultiModalOption, INode, INodeData, INodeOptionsValue, INodeParams } from '../../../src/Interface'
import { getBaseClasses, getCredentialData, getCredentialParam } from '../../../src/utils'
import { ChatAnthropic as FlowiseChatAnthropic } from './FlowiseChatAnthropic'
import { buildThinkingConfig, rejectsSamplingParams } from './anthropicModelCompat'
import { getModels, MODEL_TYPE } from '../../../src/modelLoader'

class ChatAnthropic_ChatModels implements INode {
    label: string
    name: string
    version: number
    type: string
    icon: string
    category: string
    description: string
    baseClasses: string[]
    credential: INodeParams
    inputs: INodeParams[]

    constructor() {
        this.label = 'ChatAnthropic'
        this.name = 'chatAnthropic'
        this.version = 9.4
        this.type = 'ChatAnthropic'
        this.icon = 'Anthropic.svg'
        this.category = 'Chat Models'
        this.description = 'Wrapper around ChatAnthropic large language models that use the Chat endpoint'
        this.baseClasses = [this.type, ...getBaseClasses(LangchainChatAnthropic)]
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            credentialNames: ['anthropicApi']
        }
        this.inputs = [
            {
                label: 'Cache',
                name: 'cache',
                type: 'BaseCache',
                optional: true
            },
            {
                label: 'Model Name',
                name: 'modelName',
                type: 'asyncOptions',
                loadMethod: 'listModels',
                default: 'claude-3-haiku'
            },
            {
                label: 'Temperature',
                name: 'temperature',
                type: 'number',
                step: 0.1,
                default: 0.9,
                description:
                    'Not supported on Claude Sonnet 5 and Opus 4.7+ (sampling parameters are omitted for those models).',
                optional: true
            },
            {
                label: 'Streaming',
                name: 'streaming',
                type: 'boolean',
                default: true,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Max Tokens',
                name: 'maxTokensToSample',
                type: 'number',
                step: 1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Top P',
                name: 'topP',
                type: 'number',
                step: 0.1,
                description: 'Not supported on Claude Sonnet 5 and Opus 4.7+.',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Top K',
                name: 'topK',
                type: 'number',
                step: 0.1,
                description: 'Not supported on Claude Sonnet 5 and Opus 4.7+.',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Extended Thinking',
                name: 'extendedThinking',
                type: 'boolean',
                description:
                    'Turn on to stream Claude internal thinking separately from the answer (SSE event name: llmReasoning). On Sonnet 3.7 through Sonnet 4.6, uses manual extended thinking with Budget Tokens. On Claude Sonnet 5 and Opus 4.7+, uses adaptive thinking with summarized display when on, or disables thinking when off.',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Budget Tokens',
                name: 'budgetTokens',
                type: 'number',
                step: 1,
                default: 1024,
                description:
                    'Maximum thinking tokens for manual extended thinking (Sonnet 3.7 through Sonnet 4.6). Ignored on Claude Sonnet 5 and Opus 4.7+, which use adaptive thinking instead.',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Prompt Caching',
                name: 'promptCaching',
                type: 'boolean',
                description:
                    'Enable <a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching" target="_blank">Anthropic Prompt Caching</a> to reduce costs and latency by caching tool definitions and system prompts. Cached input tokens are billed at 10% of base price. Minimum cacheable length varies by model (1024-4096 tokens). If your system prompt includes dynamic content (e.g. current date), put a line with exactly "---" (three dashes) between the static part and the dynamic part: only the text before "---" is cached, so the cache can hit.',
                default: false,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Cache Conversation History',
                name: 'cacheConversationHistory',
                type: 'boolean',
                description:
                    'When Prompt Caching is enabled, also cache the growing conversation history using Anthropic automatic caching (a top-level cache breakpoint that advances each turn). Turn off to cache only the static prefix (tool definitions + system prompt). Only applies when Prompt Caching is enabled.',
                default: true,
                optional: true,
                additionalParams: true,
                show: {
                    promptCaching: true
                }
            },
            {
                label: 'Allow Image Uploads',
                name: 'allowImageUploads',
                type: 'boolean',
                description:
                    'Allow image input. Refer to the <a href="https://docs.flowiseai.com/using-flowise/uploads#image" target="_blank">docs</a> for more details.',
                default: false,
                optional: true
            },
            {
                label: 'Beta',
                name: 'beta',
                type: 'string',
                description:
                    'Anthropic beta header value(s) for experimental features (passed as the <code>anthropic-beta</code> header). Multiple values can be comma-separated.',
                optional: true,
                additionalParams: true
            }
        ]
    }

    //@ts-ignore
    loadMethods = {
        async listModels(): Promise<INodeOptionsValue[]> {
            return await getModels(MODEL_TYPE.CHAT, 'chatAnthropic')
        }
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const temperature = nodeData.inputs?.temperature as string
        const modelName = nodeData.inputs?.modelName as string
        const maxTokens = nodeData.inputs?.maxTokensToSample as string
        const topP = nodeData.inputs?.topP as string
        const topK = nodeData.inputs?.topK as string
        const streaming = nodeData.inputs?.streaming as boolean
        const cache = nodeData.inputs?.cache as BaseCache
        const extendedThinking = nodeData.inputs?.extendedThinking as boolean
        const budgetTokens = nodeData.inputs?.budgetTokens as string
        const beta = nodeData.inputs?.beta as string
        const promptCaching = nodeData.inputs?.promptCaching as boolean
        const cacheConversationHistory = nodeData.inputs?.cacheConversationHistory as boolean

        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const anthropicApiKey = getCredentialParam('anthropicApiKey', credentialData, nodeData)

        const allowImageUploads = nodeData.inputs?.allowImageUploads as boolean

        const omitSamplingParams = rejectsSamplingParams(modelName)

        const obj: Partial<AnthropicInput> & BaseLLMParams & { anthropicApiKey?: string } = {
            modelName,
            anthropicApiKey,
            streaming: streaming ?? true
        }

        if (!omitSamplingParams) {
            obj.temperature = parseFloat(temperature)
        }

        if (maxTokens) obj.maxTokens = parseInt(maxTokens, 10)
        if (!omitSamplingParams) {
            if (topP) obj.topP = parseFloat(topP)
            if (topK) obj.topK = parseFloat(topK)
        }
        if (cache) obj.cache = cache

        const thinking = buildThinkingConfig(modelName, extendedThinking ?? false, budgetTokens)
        if (thinking) {
            // @ts-ignore – adaptive/disabled types added in newer Anthropic API; LangChain types may lag
            obj.thinking = thinking
            if (thinking.type === 'enabled') {
                delete obj.temperature
            }
        }

        // Collect anthropic-beta header values (comma-separated), de-duplicated
        const betaValues = new Set<string>(
            (beta ?? '')
                .split(',')
                .map((value) => value.trim())
                .filter((value) => value.length > 0)
        )

        if (betaValues.size > 0) {
            obj.clientOptions = {
                defaultHeaders: {
                    'anthropic-beta': Array.from(betaValues).join(',')
                }
            }
        }

        const multiModalOption: IMultiModalOption = {
            image: {
                allowImageUploads: allowImageUploads ?? false
            }
        }

        const promptCachingEnabled = promptCaching ?? false
        // Default to true so existing flows (saved before this option existed) keep automatic conversation caching
        const cacheConversationHistoryEnabled = cacheConversationHistory ?? true
        const model = new FlowiseChatAnthropic(nodeData.id, obj, promptCachingEnabled, cacheConversationHistoryEnabled)
        model.setMultiModalOption(multiModalOption)
        return model
    }
}

module.exports = { nodeClass: ChatAnthropic_ChatModels }
