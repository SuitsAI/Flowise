import axios, { AxiosError } from 'axios'
import { StructuredTool } from '@langchain/core/tools'
import { z } from 'zod'
import { ICommonObject } from '../../../src/Interface'
import { ARTIFACTS_PREFIX } from '../../../src/agents'
import { addSingleFileToStorage, getFileFromStorage } from '../../../src/storageUtils'
import { getImageUploads } from '../../../src/multiModalUtils'

export const GEMINI_INTERACTIONS_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'
export const GEMINI_INTERACTIONS_API_REVISION = '2026-05-20'

export type OmniTask = 'text_to_video' | 'image_to_video' | 'reference_to_video' | 'edit'

export interface OmniImagePart {
    type: 'image'
    data: string
    mime_type: string
}

export interface OmniGenerateParams {
    apiKey: string
    model: string
    prompt: string
    task?: OmniTask | ''
    previousInteractionId?: string
    aspectRatio?: string
    resolution?: string
    duration?: string
    delivery?: 'inline' | 'uri'
    images?: OmniImagePart[]
    timeoutMs?: number
}

export interface OmniVideoResult {
    interactionId: string
    status: string
    mimeType: string
    videoBuffer: Buffer
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

function googleErrorMessage(error: unknown): string {
    if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError<any>
        const data = axiosError.response?.data
        const message = data?.error?.message || data?.message || axiosError.message
        return typeof message === 'string' ? message : JSON.stringify(data || message)
    }
    return error instanceof Error ? error.message : String(error)
}

function authHeaders(apiKey: string): Record<string, string> {
    return {
        'Content-Type': 'application/json',
        'x-goog-api-key': apiKey,
        'Api-Revision': GEMINI_INTERACTIONS_API_REVISION
    }
}

function extractFileId(uri: string): string | undefined {
    const match = uri.match(/files\/([A-Za-z0-9._-]+)/)
    return match?.[1]
}

export function extractVideoFromInteraction(interaction: any): { mime_type?: string; data?: string; uri?: string } | undefined {
    if (interaction?.output_video) {
        return interaction.output_video
    }
    const steps = Array.isArray(interaction?.steps) ? interaction.steps : []
    for (const step of steps) {
        const contents = Array.isArray(step?.content) ? step.content : []
        for (const content of contents) {
            if (content?.type === 'video' || content?.mime_type?.startsWith?.('video/')) {
                return content
            }
        }
    }
    return undefined
}

async function getInteraction(apiKey: string, interactionId: string, timeoutMs: number): Promise<any> {
    const response = await axios.get(`${GEMINI_INTERACTIONS_BASE_URL}/interactions/${encodeURIComponent(interactionId)}`, {
        headers: authHeaders(apiKey),
        params: { key: apiKey },
        timeout: timeoutMs
    })
    return response.data
}

async function waitForInteraction(apiKey: string, interaction: any, timeoutMs: number): Promise<any> {
    const started = Date.now()
    let current = interaction
    while (current?.status && !['completed', 'failed', 'cancelled'].includes(current.status)) {
        if (Date.now() - started > timeoutMs) {
            throw new Error(`Gemini Omni interaction timed out (status: ${current.status})`)
        }
        await sleep(5000)
        current = await getInteraction(apiKey, current.id, timeoutMs)
    }
    if (current?.status === 'failed' || current?.status === 'cancelled') {
        throw new Error(`Gemini Omni interaction ${current.status}: ${JSON.stringify(current.error || current)}`)
    }
    return current
}

async function downloadFileUri(apiKey: string, uri: string, timeoutMs: number): Promise<Buffer> {
    const fileId = extractFileId(uri)
    if (!fileId) {
        throw new Error(`Could not parse Gemini file id from uri: ${uri}`)
    }

    const started = Date.now()
    while (true) {
        const statusResp = await axios.get(`${GEMINI_INTERACTIONS_BASE_URL}/files/${encodeURIComponent(fileId)}`, {
            headers: authHeaders(apiKey),
            params: { key: apiKey },
            timeout: timeoutMs
        })
        const state = statusResp.data?.state || statusResp.data?.state?.name
        if (state === 'ACTIVE') break
        if (state === 'FAILED') {
            throw new Error('Gemini Omni file processing failed')
        }
        if (Date.now() - started > timeoutMs) {
            throw new Error(`Timed out waiting for Gemini file ${fileId} to become ACTIVE`)
        }
        await sleep(5000)
    }

    const downloadResp = await axios.get(`${GEMINI_INTERACTIONS_BASE_URL}/files/${encodeURIComponent(fileId)}:download`, {
        headers: authHeaders(apiKey),
        params: { key: apiKey, alt: 'media' },
        responseType: 'arraybuffer',
        timeout: timeoutMs
    })
    return Buffer.from(downloadResp.data)
}

export async function generateOmniVideo(params: OmniGenerateParams): Promise<OmniVideoResult> {
    const timeoutMs = params.timeoutMs ?? 600000
    const input: any[] = []

    for (const image of params.images || []) {
        input.push({
            type: 'image',
            data: image.data,
            mime_type: image.mime_type
        })
    }
    input.push({ type: 'text', text: params.prompt })

    const responseFormat: ICommonObject = {
        type: 'video',
        delivery: params.delivery || 'uri'
    }
    if (params.aspectRatio) responseFormat.aspect_ratio = params.aspectRatio
    if (params.resolution) responseFormat.resolution = params.resolution
    if (params.duration) responseFormat.duration = params.duration

    const body: ICommonObject = {
        model: params.model,
        input,
        store: true,
        response_format: responseFormat
    }
    if (params.previousInteractionId) {
        body.previous_interaction_id = params.previousInteractionId
    }
    if (params.task) {
        body.generation_config = {
            video_config: {
                task: params.task
            }
        }
    }

    let interaction: any
    try {
        const response = await axios.post(`${GEMINI_INTERACTIONS_BASE_URL}/interactions`, body, {
            headers: authHeaders(params.apiKey),
            params: { key: params.apiKey },
            timeout: timeoutMs
        })
        interaction = response.data
    } catch (error) {
        throw new Error(`[GoogleGenerativeAI Error]: ${googleErrorMessage(error)}`)
    }

    interaction = await waitForInteraction(params.apiKey, interaction, timeoutMs)
    const video = extractVideoFromInteraction(interaction)
    if (!video) {
        throw new Error('Gemini Omni returned no video. Check model access and prompt.')
    }

    let videoBuffer: Buffer
    if (video.data) {
        videoBuffer = Buffer.from(video.data, 'base64')
    } else if (video.uri) {
        videoBuffer = await downloadFileUri(params.apiKey, video.uri, timeoutMs)
    } else {
        throw new Error('Gemini Omni video had neither inline data nor a file uri')
    }

    return {
        interactionId: interaction.id,
        status: interaction.status,
        mimeType: video.mime_type || 'video/mp4',
        videoBuffer
    }
}

export async function storeOmniVideoArtifact(
    video: OmniVideoResult,
    options: { orgId?: string; chatflowid?: string; chatId?: string }
): Promise<{ type: string; data: string; mimeType?: string }> {
    const filename = `gemini-omni_${Date.now()}.mp4`
    if (options.orgId && options.chatflowid && options.chatId) {
        const { path } = await addSingleFileToStorage(
            video.mimeType,
            video.videoBuffer,
            filename,
            options.orgId,
            options.chatflowid,
            options.chatId
        )
        return { type: 'mp4', data: path }
    }
    return {
        type: 'mp4',
        data: `data:${video.mimeType};base64,${video.videoBuffer.toString('base64')}`,
        mimeType: video.mimeType
    }
}

export function mimeFromFilename(name: string): string {
    const ext = name.split('.').pop()?.toLowerCase()
    if (ext === 'jpg' || ext === 'jpeg') return 'image/jpeg'
    if (ext === 'webp') return 'image/webp'
    if (ext === 'gif') return 'image/gif'
    return 'image/png'
}

export async function parseNodeImageFiles(
    fileInput: unknown,
    options: ICommonObject
): Promise<OmniImagePart[]> {
    if (!fileInput) return []
    const images: OmniImagePart[] = []

    let files: string[] = []
    let fromStorage = false
    if (typeof fileInput === 'string') {
        if (fileInput.startsWith('FILE-STORAGE::')) {
            fromStorage = true
            const fileName = fileInput.replace('FILE-STORAGE::', '')
            files = fileName.startsWith('[') && fileName.endsWith(']') ? JSON.parse(fileName) : [fileName]
        } else if (fileInput.startsWith('[') && fileInput.endsWith(']')) {
            files = JSON.parse(fileInput)
        } else {
            files = [fileInput]
        }
    }

    for (const file of files) {
        if (fromStorage) {
            const buffer = await getFileFromStorage(file, options.orgId, options.chatflowid)
            images.push({
                type: 'image',
                data: buffer.toString('base64'),
                mime_type: mimeFromFilename(file)
            })
        } else if (typeof file === 'string' && file.includes(',')) {
            const splitDataURI = file.split(',')
            const mimeMatch = splitDataURI[0]?.match(/data:([^;]+)/)
            const base64 = splitDataURI[1] || splitDataURI.pop() || ''
            images.push({
                type: 'image',
                data: base64,
                mime_type: mimeMatch?.[1] || 'image/png'
            })
        }
    }
    return images
}

export async function parseChatImageUploads(options: ICommonObject): Promise<OmniImagePart[]> {
    const images: OmniImagePart[] = []
    if (!Array.isArray(options.uploads) || options.uploads.length === 0) return images
    const imageUploads = getImageUploads(options.uploads)
    for (const upload of imageUploads) {
        if (upload.type === 'stored-file') {
            const contents = await getFileFromStorage(upload.name, options.orgId, options.chatflowid, options.chatId)
            images.push({
                type: 'image',
                data: contents.toString('base64'),
                mime_type: upload.mime || 'image/png'
            })
        } else if (upload.data) {
            const data = upload.data.includes(',') ? upload.data.split(',').pop() || '' : upload.data
            images.push({
                type: 'image',
                data,
                mime_type: upload.mime || 'image/png'
            })
        }
    }
    return images
}

export class GeminiOmniTool extends StructuredTool {
    name = 'gemini_omni'
    description =
        'Generate or conversationally edit short videos with Gemini Omni via the Interactions API. Use previous_interaction_id to edit a video from a prior turn.'
    schema = z.object({
        prompt: z.string().describe('Text prompt describing the video to generate, or the edit to apply'),
        previous_interaction_id: z
            .string()
            .optional()
            .describe('Interaction id from a previous Omni generation, used to edit the same video')
    })

    private apiKey: string
    private model: string
    private task?: OmniTask | ''
    private aspectRatio?: string
    private resolution?: string
    private duration?: string
    private delivery: 'inline' | 'uri'
    private images: OmniImagePart[]
    private storage: { orgId?: string; chatflowid?: string; chatId?: string }

    constructor(fields: {
        apiKey: string
        model: string
        task?: OmniTask | ''
        aspectRatio?: string
        resolution?: string
        duration?: string
        delivery?: 'inline' | 'uri'
        images?: OmniImagePart[]
        storage?: { orgId?: string; chatflowid?: string; chatId?: string }
        name?: string
        description?: string
    }) {
        super()
        this.apiKey = fields.apiKey
        this.model = fields.model
        this.task = fields.task
        this.aspectRatio = fields.aspectRatio
        this.resolution = fields.resolution
        this.duration = fields.duration
        this.delivery = fields.delivery || 'uri'
        this.images = fields.images || []
        this.storage = fields.storage || {}
        if (fields.name) this.name = fields.name
        if (fields.description) this.description = fields.description
    }

    protected async _call(arg: z.infer<typeof this.schema>): Promise<string> {
        const video = await generateOmniVideo({
            apiKey: this.apiKey,
            model: this.model,
            prompt: arg.prompt,
            task: this.task,
            previousInteractionId: arg.previous_interaction_id,
            aspectRatio: this.aspectRatio,
            resolution: this.resolution,
            duration: this.duration,
            delivery: this.delivery,
            images: this.images
        })
        const artifact = await storeOmniVideoArtifact(video, this.storage)
        const summary = `Video generated with Gemini Omni. Interaction ID: ${video.interactionId}. Pass this id as previous_interaction_id to edit the same clip.`
        return summary + ARTIFACTS_PREFIX + JSON.stringify([artifact])
    }
}
