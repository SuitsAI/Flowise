import { StructuredTool } from '@langchain/core/tools'
import { ICommonObject, INode, INodeData, INodeParams } from '../../../src/Interface'
import { getBaseClasses, getCredentialData, getCredentialParam } from '../../../src/utils'
import { GeminiOmniTool, OmniTask, parseNodeImageFiles } from './core'

class GeminiOmni_Tools implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    credential: INodeParams
    inputs: INodeParams[]
    documentation?: string

    constructor() {
        this.label = 'Gemini Omni'
        this.name = 'geminiOmni'
        this.version = 1.0
        this.type = 'GeminiOmni'
        this.icon = 'GoogleGemini.svg'
        this.category = 'Tools'
        this.description = 'Generate and conversationally edit videos with Gemini Omni using the Interactions API'
        this.documentation = 'https://ai.google.dev/gemini-api/docs/omni'
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            credentialNames: ['googleGenerativeAI'],
            optional: false
        }
        this.baseClasses = [this.type, ...getBaseClasses(StructuredTool), 'Tool']
        this.inputs = [
            {
                label: 'Model Name',
                name: 'modelName',
                type: 'options',
                options: [
                    { label: 'gemini-omni-1.1-flash', name: 'gemini-omni-1.1-flash' },
                    { label: 'gemini-omni-1.1-flash-preview', name: 'gemini-omni-1.1-flash-preview' },
                    { label: 'gemini-omni-flash-preview', name: 'gemini-omni-flash-preview' }
                ],
                default: 'gemini-omni-1.1-flash'
            },
            {
                label: 'Task',
                name: 'task',
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
                name: 'referenceImages',
                type: 'file',
                fileType: '.jpg, .jpeg, .png, .webp, .gif',
                description: 'Optional images for image-to-video or subject reference',
                optional: true
            },
            {
                label: 'Aspect Ratio',
                name: 'aspectRatio',
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
                name: 'resolution',
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
                name: 'duration',
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
                label: 'Name',
                name: 'toolName',
                type: 'string',
                default: 'gemini_omni',
                description: 'Name of the tool',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Description',
                name: 'toolDescription',
                type: 'string',
                rows: 3,
                default:
                    'Generate or conversationally edit short videos with Gemini Omni. Use previous_interaction_id to edit a clip from a prior turn.',
                additionalParams: true,
                optional: true
            }
        ]
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const apiKey = getCredentialParam('googleGenerativeAPIKey', credentialData, nodeData)
        if (!apiKey) {
            throw new Error('Google Generative AI API key is required for Gemini Omni')
        }

        const images = await parseNodeImageFiles(nodeData.inputs?.referenceImages, options)

        return new GeminiOmniTool({
            apiKey,
            model: (nodeData.inputs?.modelName as string) || 'gemini-omni-1.1-flash',
            task: (nodeData.inputs?.task as OmniTask | '') || '',
            aspectRatio: nodeData.inputs?.aspectRatio as string,
            resolution: nodeData.inputs?.resolution as string,
            duration: nodeData.inputs?.duration as string,
            images,
            storage: {
                orgId: options.orgId,
                chatflowid: options.chatflowid,
                chatId: options.chatId
            },
            name: (nodeData.inputs?.toolName as string) || 'gemini_omni',
            description: nodeData.inputs?.toolDescription as string
        })
    }
}

module.exports = { nodeClass: GeminiOmni_Tools }
