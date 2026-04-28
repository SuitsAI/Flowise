import { AnthropicInput, ChatAnthropic as LangchainChatAnthropic } from '@langchain/anthropic'
import { BaseCache } from '@langchain/core/caches'
import { BaseLLMParams } from '@langchain/core/language_models/llms'
import { ICommonObject, IMultiModalOption, INode, INodeData, INodeOptionsValue, INodeParams } from '../../../src/Interface'
import { getBaseClasses, getCredentialData, getCredentialParam } from '../../../src/utils'
import { ChatAnthropic as FlowiseChatAnthropic } from './FlowiseChatAnthropic'
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
        this.version = 9.1
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
                optional: true,
                additionalParams: true
            },
            {
                label: 'Top K',
                name: 'topK',
                type: 'number',
                step: 0.1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Extended Thinking',
                name: 'extendedThinking',
                type: 'boolean',
                description: 'Enable extended thinking for reasoning model such as Claude Sonnet 3.7 and Claude 4',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Budget Tokens',
                name: 'budgetTokens',
                type: 'number',
                step: 1,
                default: 1024,
                description: 'Maximum number of tokens Claude is allowed use for its internal reasoning process',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Prompt Caching',
                name: 'promptCaching',
                type: 'boolean',
                description:
                    'Enable <a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching" target="_blank">Anthropic Prompt Caching</a> to reduce costs and latency by caching system prompts and conversation history. Cached input tokens are billed at 10% of base price. Minimum cacheable length varies by model (1024-4096 tokens). If your system prompt includes dynamic content (e.g. current date), put a line with exactly "---" (three dashes) between the static part and the dynamic part: only the text before "---" is cached, so the cache can hit.',
                default: false,
                optional: true,
                additionalParams: true
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
                    'Beta parameter for experimental features (e.g. <code>compact-2026-01-12</code> when enabling Compaction below). Multiple values can be comma-separated.',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Compaction',
                name: 'compaction',
                type: 'boolean',
                description:
                    'Enable <a href="https://platform.claude.com/docs/en/build-with-claude/compaction" target="_blank">server-side context compaction</a> to extend long conversations. Requires the <code>compact-2026-01-12</code> beta header (set in the Beta field above) and a supported model (Claude Opus 4.6+, Sonnet 4.6+).',
                default: false,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Compaction Trigger Tokens',
                name: 'compactionTriggerTokens',
                type: 'number',
                step: 1000,
                description:
                    'Input token threshold that triggers compaction. Defaults to 150,000 if left empty. Must be at least 50,000.',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Compaction Instructions',
                name: 'compactionInstructions',
                type: 'string',
                rows: 4,
                description:
                    'Custom summarization prompt used when compaction triggers. Completely replaces the default prompt when provided.',
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
        const compaction = nodeData.inputs?.compaction as boolean
        const compactionTriggerTokens = nodeData.inputs?.compactionTriggerTokens as string
        const compactionInstructions = nodeData.inputs?.compactionInstructions as string

        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const anthropicApiKey = getCredentialParam('anthropicApiKey', credentialData, nodeData)

        const allowImageUploads = nodeData.inputs?.allowImageUploads as boolean

        const obj: Partial<AnthropicInput> & BaseLLMParams & { anthropicApiKey?: string } = {
            temperature: parseFloat(temperature),
            modelName,
            anthropicApiKey,
            streaming: streaming ?? true
        }

        if (beta) obj.clientOptions = {
            defaultHeaders: {
                'anthropic-beta': beta
            }
        }

        if (maxTokens) obj.maxTokens = parseInt(maxTokens, 10)
        if (topP) obj.topP = parseFloat(topP)
        if (topK) obj.topK = parseFloat(topK)
        if (cache) obj.cache = cache
        if (extendedThinking) {
            obj.thinking = {
                type: 'enabled',
                budget_tokens: parseInt(budgetTokens, 10)
            }
            delete obj.temperature
        }

        if (compaction) {
            const edit: Record<string, unknown> = { type: 'compact_20260112' }
            if (compactionTriggerTokens) {
                edit.trigger = { type: 'input_tokens', value: parseInt(compactionTriggerTokens, 10) }
            }
            if (compactionInstructions) edit.instructions = compactionInstructions
            obj.contextManagement = { edits: [edit] } as unknown as AnthropicInput['contextManagement']
        }

        const multiModalOption: IMultiModalOption = {
            image: {
                allowImageUploads: allowImageUploads ?? false
            }
        }

        const promptCachingEnabled = promptCaching ?? false
        const model = new FlowiseChatAnthropic(nodeData.id, obj, promptCachingEnabled)
        model.setMultiModalOption(multiModalOption)
        return model
    }
}

module.exports = { nodeClass: ChatAnthropic_ChatModels }
