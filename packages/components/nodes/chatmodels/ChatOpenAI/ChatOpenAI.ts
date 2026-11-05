import { ChatOpenAI as LangchainChatOpenAI, ChatOpenAIFields } from '@langchain/openai'
import { BaseCache } from '@langchain/core/caches'
import { ICommonObject, IMultiModalOption, INode, INodeData, INodeOptionsValue, INodeParams } from '../../../src/Interface'
import { getBaseClasses, getCredentialData, getCredentialParam } from '../../../src/utils'
import { ChatOpenAI } from './FlowiseChatOpenAI'
import { buildImageGenerationTool } from './openaiImageGeneration'
import { getModels, MODEL_TYPE } from '../../../src/modelLoader'
import { openAIReasoningParamsApplyToModel } from './flowiseOpenAIReasoning'
import { HttpsProxyAgent } from 'https-proxy-agent'
import { OpenAI as OpenAIClient } from 'openai'

class ChatOpenAI_ChatModels implements INode {
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
        this.label = 'ChatOpenAI'
        this.name = 'chatOpenAI'
        this.version = 8.6
        this.type = 'ChatOpenAI'
        this.icon = 'openai.svg'
        this.category = 'Chat Models'
        this.description = 'Wrapper around OpenAI large language models that use the Chat endpoint'
        this.baseClasses = [this.type, ...getBaseClasses(LangchainChatOpenAI)]
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            credentialNames: ['openAIApi']
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
                default: 'gpt-4o-mini'
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
                name: 'maxTokens',
                type: 'number',
                step: 1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Top Probability',
                name: 'topP',
                type: 'number',
                step: 0.1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Frequency Penalty',
                name: 'frequencyPenalty',
                type: 'number',
                step: 0.1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Presence Penalty',
                name: 'presencePenalty',
                type: 'number',
                step: 0.1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Timeout',
                name: 'timeout',
                type: 'number',
                step: 1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Strict Tool Calling',
                name: 'strictToolCalling',
                type: 'boolean',
                description:
                    'Whether the model supports the `strict` argument when passing in tools. If not specified, the `strict` argument will not be passed to OpenAI.',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Stop Sequence',
                name: 'stopSequence',
                type: 'string',
                rows: 4,
                optional: true,
                description: 'List of stop words to use when generating. Use comma to separate multiple stop words.',
                additionalParams: true
            },
            {
                label: 'BasePath',
                name: 'basepath',
                type: 'string',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Proxy Url',
                name: 'proxyUrl',
                type: 'string',
                optional: true,
                additionalParams: true
            },
            {
                label: 'BaseOptions',
                name: 'baseOptions',
                type: 'json',
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
                label: 'Image Resolution',
                description: 'This parameter controls the resolution in which the model views the image.',
                name: 'imageResolution',
                type: 'options',
                options: [
                    {
                        label: 'Low',
                        name: 'low'
                    },
                    {
                        label: 'High',
                        name: 'high'
                    },
                    {
                        label: 'Auto',
                        name: 'auto'
                    }
                ],
                default: 'low',
                optional: false,
                show: {
                    allowImageUploads: true
                }
            },
            {
                label: 'Allow Image Generation',
                name: 'allowImageGeneration',
                type: 'boolean',
                description:
                    'Generate images with the OpenAI Responses API image tool. The chat model stays a mainline model (gpt-4o, gpt-4.1, gpt-5, and newer) and calls gpt-image. Images are returned as chat artifacts, the same way Gemini image models do. See the <a href="https://developers.openai.com/api/docs/guides/image-generation" target="_blank">image generation guide</a>.',
                default: false,
                optional: true
            },
            {
                label: 'Image Model',
                name: 'imageGenerationModel',
                type: 'options',
                description:
                    'Model used inside the image generation tool. 2.6 Sunburst is for detailed visual work. 2.6 Flare is for fast iteration. 2.5 Sunburst is better for precise edits, and 2.5 Flare is faster for everyday generation.',
                options: [
                    { label: 'gpt-image-2.6-sunburst', name: 'gpt-image-2.6-sunburst' },
                    { label: 'gpt-image-2.6-flare', name: 'gpt-image-2.6-flare' },
                    { label: 'gpt-image-2.5-sunburst', name: 'gpt-image-2.5-sunburst' },
                    { label: 'gpt-image-2.5-flare', name: 'gpt-image-2.5-flare' },
                    { label: 'gpt-image-1.5', name: 'gpt-image-1.5' },
                    { label: 'gpt-image-1', name: 'gpt-image-1' },
                    { label: 'gpt-image-1-mini', name: 'gpt-image-1-mini' }
                ],
                default: 'gpt-image-2.6-sunburst',
                optional: true,
                additionalParams: true,
                show: {
                    allowImageGeneration: true
                }
            },
            {
                label: 'Image Size',
                name: 'imageSize',
                type: 'options',
                description: 'Output size. Auto lets the image model choose. These are the sizes OpenAI recommends for GPT Image.',
                options: [
                    { label: 'Auto', name: 'auto' },
                    { label: '1024x1024 (square)', name: '1024x1024' },
                    { label: '1536x1024 (landscape)', name: '1536x1024' },
                    { label: '1024x1536 (portrait)', name: '1024x1536' }
                ],
                default: 'auto',
                optional: true,
                additionalParams: true,
                show: {
                    allowImageGeneration: true
                }
            },
            {
                label: 'Image Quality',
                name: 'imageQuality',
                type: 'options',
                description:
                    'Rendering quality. xhigh and max apply to GPT Image 2.5 and 2.6 Sunburst and Flare. Earlier gpt-image models support up to high.',
                options: [
                    { label: 'Auto', name: 'auto' },
                    { label: 'Low', name: 'low' },
                    { label: 'Medium', name: 'medium' },
                    { label: 'High', name: 'high' },
                    { label: 'Xhigh', name: 'xhigh' },
                    { label: 'Max', name: 'max' }
                ],
                default: 'auto',
                optional: true,
                additionalParams: true,
                show: {
                    allowImageGeneration: true
                }
            },
            {
                label: 'Image Format',
                name: 'imageOutputFormat',
                type: 'options',
                description: 'png and jpeg render in chat. webp is returned as a file. Use png or webp for transparent backgrounds.',
                options: [
                    { label: 'PNG', name: 'png' },
                    { label: 'JPEG', name: 'jpeg' },
                    { label: 'WebP', name: 'webp' }
                ],
                default: 'png',
                optional: true,
                additionalParams: true,
                show: {
                    allowImageGeneration: true
                }
            },
            {
                label: 'Image Background',
                name: 'imageBackground',
                type: 'options',
                description: 'Transparent backgrounds require png or webp output.',
                options: [
                    { label: 'Auto', name: 'auto' },
                    { label: 'Opaque', name: 'opaque' },
                    { label: 'Transparent', name: 'transparent' }
                ],
                default: 'auto',
                optional: true,
                additionalParams: true,
                show: {
                    allowImageGeneration: true
                }
            },
            {
                label: 'Image Compression',
                name: 'imageOutputCompression',
                type: 'number',
                step: 1,
                description: 'Compression from 0 to 100. Only sent for jpeg and webp.',
                optional: true,
                additionalParams: true,
                show: {
                    allowImageGeneration: true
                }
            },
            {
                label: 'Reasoning',
                description:
                    'Turn on to stream model reasoning separately from answer tokens (SSE event name: llmReasoning). Only applicable to reasoning-capable models.',
                name: 'reasoning',
                type: 'boolean',
                default: false,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Reasoning Effort',
                description:
                    'Controls how many reasoning tokens the model generates before responding. GPT-6 Sol/Luna and GPT-5.2+ default to "none" for lower latency. GPT-6 Astra and GPT-6.1 Sol support low through max (not none); 6.1 Sol defaults to medium. Increase to "medium", "high", "xhigh", or "max" for more thorough reasoning. For older models (o1/o3), only low/medium/high are supported.',
                name: 'reasoningEffort',
                type: 'options',
                options: [
                    {
                        label: 'None (default for GPT-5.2+)',
                        name: 'none'
                    },
                    {
                        label: 'Low',
                        name: 'low'
                    },
                    {
                        label: 'Medium',
                        name: 'medium'
                    },
                    {
                        label: 'High',
                        name: 'high'
                    },
                    {
                        label: 'Xhigh',
                        name: 'xhigh'
                    },
                    {
                        label: 'Max',
                        name: 'max'
                    }
                ],
                additionalParams: true,
                show: {
                    reasoning: true
                }
            },
            {
                label: 'Reasoning Summary',
                description: `Controls OpenAI reasoning summary output. When available, Flowise streams it in llmReasoning SSE events.`,
                name: 'reasoningSummary',
                type: 'options',
                options: [
                    {
                        label: 'Auto',
                        name: 'auto'
                    },
                    {
                        label: 'Concise',
                        name: 'concise'
                    },
                    {
                        label: 'Detailed',
                        name: 'detailed'
                    }
                ],
                additionalParams: true,
                show: {
                    reasoning: true
                }
            },
            {
                label: 'Verbosity',
                description:
                    'Controls how many output tokens are generated. "High" is best for thorough explanations or extensive code refactoring. "Low" is best for concise answers or simple code generation. "Medium" is the default for GPT-5.4+. Only applicable for GPT-5+ and GPT-6 models.',
                name: 'verbosity',
                type: 'options',
                options: [
                    {
                        label: 'Low',
                        name: 'low'
                    },
                    {
                        label: 'Medium (default)',
                        name: 'medium'
                    },
                    {
                        label: 'High',
                        name: 'high'
                    }
                ],
                optional: true,
                additionalParams: true
            }
        ]
    }

    //@ts-ignore
    loadMethods = {
        async listModels(): Promise<INodeOptionsValue[]> {
            return await getModels(MODEL_TYPE.CHAT, 'chatOpenAI')
        }
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const temperature = nodeData.inputs?.temperature as string
        const modelName = nodeData.inputs?.modelName as string
        const maxTokens = nodeData.inputs?.maxTokens as string
        const topP = nodeData.inputs?.topP as string
        const frequencyPenalty = nodeData.inputs?.frequencyPenalty as string
        const presencePenalty = nodeData.inputs?.presencePenalty as string
        const timeout = nodeData.inputs?.timeout as string
        const stopSequence = nodeData.inputs?.stopSequence as string
        const streaming = nodeData.inputs?.streaming as boolean
        const strictToolCalling = nodeData.inputs?.strictToolCalling as boolean
        const basePath = nodeData.inputs?.basepath as string
        const proxyUrl = nodeData.inputs?.proxyUrl as string
        const baseOptions = nodeData.inputs?.baseOptions
        const reasoningEnabled = nodeData.inputs?.reasoning === true
        const reasoningEffort = nodeData.inputs?.reasoningEffort as OpenAIClient.ReasoningEffort | null
        const reasoningSummary = nodeData.inputs?.reasoningSummary as 'auto' | 'concise' | 'detailed' | null
        const verbosity = nodeData.inputs?.verbosity as 'low' | 'medium' | 'high' | null

        const allowImageUploads = nodeData.inputs?.allowImageUploads as boolean
        const imageResolution = nodeData.inputs?.imageResolution as string
        const allowImageGeneration = nodeData.inputs?.allowImageGeneration === true
        const imageGenerationModel = nodeData.inputs?.imageGenerationModel as string
        const imageSize = nodeData.inputs?.imageSize as string
        const imageQuality = nodeData.inputs?.imageQuality as string
        const imageOutputFormat = nodeData.inputs?.imageOutputFormat as string
        const imageBackground = nodeData.inputs?.imageBackground as string
        const imageOutputCompression = nodeData.inputs?.imageOutputCompression as string

        if (nodeData.inputs?.credentialId) {
            nodeData.credential = nodeData.inputs?.credentialId
        }
        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const openAIApiKey = getCredentialParam('openAIApiKey', credentialData, nodeData)

        const cache = nodeData.inputs?.cache as BaseCache

        const obj: ChatOpenAIFields = {
            temperature: parseFloat(temperature),
            modelName,
            openAIApiKey,
            apiKey: openAIApiKey,
            streaming: streaming ?? true,
            ...(allowImageGeneration ? { useResponsesApi: true } : {})
        }

        if (maxTokens) obj.maxTokens = parseInt(maxTokens, 10)
        if (topP) obj.topP = parseFloat(topP)
        if (frequencyPenalty) obj.frequencyPenalty = parseFloat(frequencyPenalty)
        if (presencePenalty) obj.presencePenalty = parseFloat(presencePenalty)
        if (timeout) obj.timeout = parseInt(timeout, 10)
        if (cache) obj.cache = cache
        if (stopSequence) {
            const stopSequenceArray = stopSequence.split(',').map((item) => item.trim())
            obj.stop = stopSequenceArray
        }
        if (strictToolCalling) obj.supportsStrictToolCalling = strictToolCalling

        if (openAIReasoningParamsApplyToModel(modelName)) {
            delete obj.temperature
            delete obj.stop
            const reasoning: OpenAIClient.Reasoning = {}
            if (reasoningEffort) {
                reasoning.effort = reasoningEffort
            }
            if (reasoningSummary) {
                reasoning.summary = reasoningSummary
            }
            // UI toggle alone should still request reasoning summary streams when sub-fields were left default-empty.
            if (reasoningEnabled && Object.keys(reasoning).length === 0) {
                reasoning.summary = 'auto'
            }
            if (Object.keys(reasoning).length > 0) {
                obj.reasoning = reasoning
            }
        }

        if ((modelName.includes('gpt-5') || modelName.includes('gpt-6')) && verbosity) {
            obj.modelKwargs = {
                ...obj.modelKwargs,
                text: { verbosity }
            }
        }

        let parsedBaseOptions: any | undefined = undefined

        if (baseOptions) {
            try {
                parsedBaseOptions = typeof baseOptions === 'object' ? baseOptions : JSON.parse(baseOptions)
            } catch (exception) {
                throw new Error("Invalid JSON in the ChatOpenAI's BaseOptions: " + exception)
            }
        }

        if (basePath || parsedBaseOptions) {
            obj.configuration = {
                baseURL: basePath,
                defaultHeaders: parsedBaseOptions
            }
        }

        if (proxyUrl) {
            obj.configuration = {
                ...obj?.configuration,
                httpAgent: new HttpsProxyAgent(proxyUrl)
            }
        }

        const multiModalOption: IMultiModalOption = {
            image: {
                allowImageUploads: allowImageUploads ?? false,
                imageResolution
            }
        }

        const model = new ChatOpenAI(nodeData.id, obj)
        model.setMultiModalOption(multiModalOption)
        if (allowImageGeneration) {
            const compression =
                imageOutputCompression !== undefined && imageOutputCompression !== null && String(imageOutputCompression).trim() !== ''
                    ? parseInt(String(imageOutputCompression), 10)
                    : undefined
            model.setImageGenerationTool(
                buildImageGenerationTool({
                    model: imageGenerationModel,
                    size: imageSize,
                    quality: imageQuality,
                    outputFormat: imageOutputFormat,
                    background: imageBackground,
                    outputCompression: compression
                })
            )
            model.setFlowContext({
                chatflowid: options.chatflowid,
                orgId: options.orgId,
                chatId: options.chatId
            })
        }
        return model
    }
}

module.exports = { nodeClass: ChatOpenAI_ChatModels }
