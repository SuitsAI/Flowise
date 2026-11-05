import type { BaseMessage } from '@langchain/core/messages'
import { addSingleFileToStorage } from '../../../src/storageUtils'

export interface OpenAIImageGenerationConfig {
    model?: string
    size?: string
    quality?: string
    outputFormat?: string
    background?: string
    outputCompression?: number
}

export interface OpenAIImageFlowContext {
    chatflowid?: string
    orgId?: string
    chatId?: string
}

type ImageArtifact = { type: string; data: string; mimeType?: string }

/**
 * Built-in Responses API tool. Passed through by LangChain when `type !== "function"`.
 * Field names match the OpenAI API (snake_case), not the newer LangChain helper.
 */
export function buildImageGenerationTool(config: OpenAIImageGenerationConfig): Record<string, unknown> {
    const tool: Record<string, unknown> = {
        type: 'image_generation',
        model: config.model || 'gpt-image-2.6-sunburst'
    }
    if (config.size) tool.size = config.size
    if (config.quality) tool.quality = config.quality
    if (config.outputFormat) tool.output_format = config.outputFormat
    if (config.background) tool.background = config.background
    if (
        config.outputCompression !== undefined &&
        !Number.isNaN(config.outputCompression) &&
        (config.outputFormat === 'jpeg' || config.outputFormat === 'webp')
    ) {
        tool.output_compression = config.outputCompression
    }
    return tool
}

export function isImageGenerationTool(tool: unknown): boolean {
    return !!tool && typeof tool === 'object' && 'type' in tool && (tool as { type?: string }).type === 'image_generation'
}

function artifactTypeForFormat(format: string): { extension: string; mimeType: string; artifactType: string } {
    const normalized = format.toLowerCase()
    if (normalized === 'jpeg' || normalized === 'jpg') {
        return { extension: 'jpg', mimeType: 'image/jpeg', artifactType: 'jpeg' }
    }
    if (normalized === 'webp') {
        return { extension: 'webp', mimeType: 'image/webp', artifactType: 'file' }
    }
    if (normalized === 'gif') {
        return { extension: 'gif', mimeType: 'image/gif', artifactType: 'file' }
    }
    return { extension: 'png', mimeType: 'image/png', artifactType: 'png' }
}

function imageCallsFromMessage(message: BaseMessage): Array<Record<string, any>> {
    const kwargs = message.additional_kwargs ?? {}
    const metadata = message.response_metadata ?? {}
    const buckets = [kwargs.tool_outputs, metadata.output, metadata.tool_outputs]
    const calls: Array<Record<string, any>> = []
    for (const bucket of buckets) {
        if (!Array.isArray(bucket)) continue
        for (const item of bucket) {
            if (item && typeof item === 'object' && item.type === 'image_generation_call') {
                calls.push(item)
            }
        }
    }
    return calls
}

function appendArtifacts(message: BaseMessage, artifacts: ImageArtifact[]): void {
    if (!artifacts.length) return
    const blocks = artifacts.map((artifact) => ({ type: 'artifact', artifact }))
    if (typeof message.content === 'string') {
        message.content = message.content
            ? [{ type: 'text', text: message.content }, ...blocks]
            : blocks
        return
    }
    if (Array.isArray(message.content)) {
        message.content = [...message.content, ...blocks]
        return
    }
    message.content = blocks
}

/**
 * Saves `image_generation_call` results and appends Flowise artifact blocks to the message.
 * The Tool Agent reads those blocks from the model output, same as Gemini inline images.
 * Base64 is removed afterwards so it is not replayed on the next Responses API turn;
 * multi-turn edits only need the call id.
 */
export async function attachImageGenerationArtifacts(
    generation: { message?: BaseMessage },
    context: OpenAIImageFlowContext | undefined,
    seen: Set<string>
): Promise<void> {
    const message = generation.message
    if (!message) return

    const artifacts: ImageArtifact[] = []
    for (const item of imageCallsFromMessage(message)) {
        const result = typeof item.result === 'string' ? item.result : ''
        if (!result || result.startsWith('FILE-STORAGE::') || result.startsWith('data:')) continue

        const resultKey = result.slice(0, 80)
        const id = typeof item.id === 'string' ? item.id : ''
        if ((id && seen.has(id)) || seen.has(resultKey)) {
            delete item.result
            continue
        }
        if (id) seen.add(id)
        seen.add(resultKey)

        const format = typeof item.output_format === 'string' ? item.output_format : 'png'
        const { extension, mimeType, artifactType } = artifactTypeForFormat(format)
        const filename = `generated_image_${Date.now()}.${extension}`

        try {
            const imageData = Buffer.from(result, 'base64')
            if (context?.chatflowid && context.orgId && context.chatId) {
                const { path } = await addSingleFileToStorage(
                    mimeType,
                    imageData,
                    filename,
                    context.orgId,
                    context.chatflowid,
                    context.chatId
                )
                artifacts.push({ type: artifactType, data: path })
            } else {
                artifacts.push({
                    type: artifactType,
                    data: `data:${mimeType};base64,${result}`,
                    mimeType
                })
            }
        } catch (error) {
            console.error('Error storing OpenAI image generation artifact:', error)
        }

        delete item.result
    }

    appendArtifacts(message, artifacts)
}
