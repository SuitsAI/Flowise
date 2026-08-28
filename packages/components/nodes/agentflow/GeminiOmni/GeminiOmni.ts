import { ICommonObject, INode, INodeData, INodeOptionsValue, INodeParams, IServerSideEventStreamer } from '../../../src/Interface'
import { getCredentialData, getCredentialParam, processTemplateVariables } from '../../../src/utils'
import { updateFlowState } from '../utils'
import {
    generateOmniVideo,
    OmniTask,
    parseChatImageUploads,
    parseNodeImageFiles,
    storeOmniVideoArtifact
} from '../../tools/GeminiOmni/core'

class GeminiOmni_Agentflow implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    color: string
    baseClasses: string[]
    documentation?: string
    credential: INodeParams
    inputs: INodeParams[]

    constructor() {
        this.label = 'Gemini Omni'
        this.name = 'geminiOmniAgentflow'
        this.version = 1.0
        this.type = 'GeminiOmni'
        this.icon = 'GoogleGemini.svg'
        this.category = 'Agent Flows'
        this.color = '#4285F4'
        this.description = 'Generate and conversationally edit videos with Gemini Omni via the Interactions API'
        this.documentation = 'https://ai.google.dev/gemini-api/docs/omni'
        this.baseClasses = [this.type]
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            credentialNames: ['googleGenerativeAI'],
            optional: false
        }
        this.inputs = [
            {
                label: 'Model Name',
                name: 'omniModelName',
                type: 'options',
                options: [
                    { label: 'gemini-omni-1.1-flash', name: 'gemini-omni-1.1-flash' },
                    { label: 'gemini-omni-1.1-flash-preview', name: 'gemini-omni-1.1-flash-preview' },
                    { label: 'gemini-omni-flash-preview', name: 'gemini-omni-flash-preview' }
                ],
                default: 'gemini-omni-1.1-flash'
            },
            {
                label: 'Prompt',
                name: 'omniPrompt',
                type: 'string',
                rows: 4,
                acceptVariable: true,
                description: 'Leave empty to use the user question as the prompt'
            },
            {
                label: 'Previous Interaction ID',
                name: 'omniPreviousInteractionId',
                type: 'string',
                acceptVariable: true,
                optional: true,
                description: 'Pass an earlier Omni interaction id to edit the same video without re-uploading'
            },
            {
                label: 'Task',
                name: 'omniTask',
                type: 'options',
                options: [
                    { label: 'Auto', name: '' },
                    { label: 'Text to video', name: 'text_to_video' },
                    { label: 'Image to video', name: 'image_to_video' },
                    { label: 'Reference to video', name: 'reference_to_video' },
                    { label: 'Edit', name: 'edit' }
                ],
                default: '',
                optional: true
            },
            {
                label: 'Reference Images',
                name: 'omniReferenceImages',
                type: 'file',
                fileType: '.jpg, .jpeg, .png, .webp, .gif',
                optional: true
            },
            {
                label: 'Use Chat Image Uploads',
                name: 'omniUseChatImages',
                type: 'boolean',
                default: false,
                optional: true,
                description: 'Include images the user uploaded in chat as Omni input'
            },
            {
                label: 'Aspect Ratio',
                name: 'omniAspectRatio',
                type: 'options',
                options: [
                    { label: '16:9', name: '16:9' },
                    { label: '9:16', name: '9:16' }
                ],
                default: '16:9',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Resolution',
                name: 'omniResolution',
                type: 'options',
                options: [
                    { label: 'Default', name: '' },
                    { label: '360p', name: '360p' },
                    { label: '720p', name: '720p' }
                ],
                default: '',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Duration',
                name: 'omniDuration',
                type: 'options',
                options: [
                    { label: 'Default', name: '' },
                    { label: '4s', name: '4s' },
                    { label: '6s', name: '6s' },
                    { label: '8s', name: '8s' },
                    { label: '10s', name: '10s' }
                ],
                default: '',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Update Flow State',
                name: 'omniUpdateState',
                description: 'Save values such as {{ output.interactionId }} for later edit turns',
                type: 'array',
                optional: true,
                acceptVariable: true,
                array: [
                    {
                        label: 'Key',
                        name: 'key',
                        type: 'asyncOptions',
                        loadMethod: 'listRuntimeStateKeys',
                        freeSolo: true
                    },
                    {
                        label: 'Value',
                        name: 'value',
                        type: 'string',
                        acceptVariable: true,
                        acceptNodeOutputAsVariable: true
                    }
                ]
            }
        ]
    }

    //@ts-ignore
    loadMethods = {
        async listRuntimeStateKeys(_: INodeData, options: ICommonObject): Promise<INodeOptionsValue[]> {
            const previousNodes = options.previousNodes as ICommonObject[]
            const startAgentflowNode = previousNodes.find((node) => node.name === 'startAgentflow')
            const state = (startAgentflowNode?.inputs?.startState as ICommonObject[]) || []
            return state.map((item) => ({ label: item.key, name: item.key }))
        }
    }

    async run(nodeData: INodeData, input: string, options: ICommonObject): Promise<any> {
        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const apiKey = getCredentialParam('googleGenerativeAPIKey', credentialData, nodeData)
        if (!apiKey) {
            throw new Error('Google Generative AI API key is required for Gemini Omni')
        }

        const prompt = ((nodeData.inputs?.omniPrompt as string) || input || '').trim()
        if (!prompt) {
            throw new Error('Gemini Omni requires a prompt')
        }

        const previousInteractionId = (nodeData.inputs?.omniPreviousInteractionId as string) || undefined
        const images = [
            ...(await parseNodeImageFiles(nodeData.inputs?.omniReferenceImages, options)),
            ...(nodeData.inputs?.omniUseChatImages ? await parseChatImageUploads(options) : [])
        ]

        const video = await generateOmniVideo({
            apiKey,
            model: (nodeData.inputs?.omniModelName as string) || 'gemini-omni-1.1-flash',
            prompt,
            task: (nodeData.inputs?.omniTask as OmniTask | '') || '',
            previousInteractionId,
            aspectRatio: nodeData.inputs?.omniAspectRatio as string,
            resolution: nodeData.inputs?.omniResolution as string,
            duration: nodeData.inputs?.omniDuration as string,
            images
        })

        const artifact = await storeOmniVideoArtifact(video, {
            orgId: options.orgId,
            chatflowid: options.chatflowid,
            chatId: options.chatId
        })

        const content = `Video generated with Gemini Omni. Interaction ID: ${video.interactionId}`
        const chatId = options.chatId as string
        const isLastNode = options.isLastNode as boolean
        const sseStreamer: IServerSideEventStreamer | undefined = options.sseStreamer

        if (isLastNode && sseStreamer) {
            sseStreamer.streamTokenEvent(chatId, content)
            sseStreamer.streamArtifactsEvent(chatId, [artifact])
        }

        const state = options.agentflowRuntime?.state as ICommonObject
        let newState = { ...state }
        const updateState = nodeData.inputs?.omniUpdateState
        if (Array.isArray(updateState) && updateState.length > 0) {
            newState = updateFlowState(state, updateState)
        }

        const outputPayload = {
            content,
            interactionId: video.interactionId,
            artifacts: [artifact]
        }
        newState = processTemplateVariables(newState, outputPayload)

        return {
            id: nodeData.id,
            name: this.name,
            input: {
                prompt,
                previousInteractionId,
                model: nodeData.inputs?.omniModelName
            },
            output: outputPayload,
            state: newState
        }
    }
}

module.exports = { nodeClass: GeminiOmni_Agentflow }
