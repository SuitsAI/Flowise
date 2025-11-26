/**
 * NOTE: This file has been updated to support thought signatures for Gemini 3 Pro.
 * 
 * Thought signature support has been implemented according to:
 * https://ai.google.dev/gemini-api/docs/thought-signatures
 * 
 * To complete the migration from @google/generative-ai to @google/genai:
 * 1. Install @google/genai package: npm install @google/genai
 * 2. Update this import to use @google/genai
 * 3. Update all other files in this directory that import from @google/generative-ai
 * 4. Verify that export names match between packages (they may differ)
 * 
 * Thought signature implementation:
 * - Extracts thought signatures from function call parts and text parts
 * - Stores them in AIMessage additional_kwargs (thought_signatures for function calls, thought_signature for text)
 * - Preserves them when converting messages back to parts
 * - Handles both parallel (first call only) and sequential (each call) function calls correctly
 * - Required for Gemini 3 Pro to avoid validation errors during function calling
 */
import {
    Content,
    EnhancedGenerateContentResponse,
    FileDataPart,
    FunctionCallPart,
    InlineDataPart,
    POSSIBLE_ROLES,
    Part,
    TextPart,
    type FunctionDeclaration as GenerativeAIFunctionDeclaration,
    type FunctionDeclarationsTool as GoogleGenerativeAIFunctionDeclarationsTool
} from '@google/generative-ai'
import { isOpenAITool } from '@langchain/core/language_models/base'
import {
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    MessageContent,
    MessageContentComplex,
    StandardContentBlockConverter,
    ToolMessage,
    ToolMessageChunk,
    UsageMetadata,
    convertToProviderContentBlock,
    isAIMessage,
    isBaseMessage,
    isDataContentBlock,
    isToolMessage,
    parseBase64DataUrl
} from '@langchain/core/messages'
import { ToolCallChunk } from '@langchain/core/messages/tool'
import { ChatGeneration, ChatGenerationChunk, ChatResult } from '@langchain/core/outputs'
import { isLangChainTool } from '@langchain/core/utils/function_calling'
import { v4 as uuidv4 } from 'uuid'
import { ARTIFACTS_PREFIX } from '../../../../src/agents'
import { addSingleFileToStorage } from '../../../../src/storageUtils'
import { GoogleGenerativeAIToolType } from './types.js'
import { jsonSchemaToGeminiParameters, schemaToGenerativeAIParameters } from './zod_to_genai_parameters.js'

/**
 * Extracts thought signature from a Part if it exists.
 * Thought signatures are found directly on FunctionCallPart as thoughtSignature (per SDK type definitions).
 * Note: extra_content is a LangChain-specific field and may exist in responses, but we should only
 * extract from the direct thoughtSignature field to match what the API expects.
 * According to Google's documentation: https://ai.google.dev/gemini-api/docs/thought-signatures
 */
function extractThoughtSignature(part: Part): string | undefined {
    const partAny = part as any
    
    // Check if it's directly on the part (FunctionCallPart has thoughtSignature?: string)
    // This is the primary and correct location per SDK type definitions
    if ('functionCall' in part && partAny.thoughtSignature && typeof partAny.thoughtSignature === 'string') {
        return partAny.thoughtSignature
    }
    
    // Fallback: Check extra_content.google.thought_signature (may exist in LangChain responses)
    // But note: we should NOT set this when creating parts - only extract it if present
    if (partAny.extra_content?.google?.thought_signature) {
        return partAny.extra_content.google.thought_signature
    }
    
    return undefined
}

/**
 * Creates a Part with thought signature if provided.
 * According to the SDK type definitions, FunctionCallPart has thoughtSignature?: string directly on it.
 * The Gemini API only accepts thoughtSignature directly on the part, NOT in extra_content.
 * extra_content is a LangChain-specific field and causes API validation errors.
 * This is required for Gemini 3 Pro to preserve reasoning context across turns.
 */
function createPartWithThoughtSignature(part: Part, thoughtSignature?: string): Part {
    if (!thoughtSignature) {
        return part
    }
    
    const partWithSignature = { ...part } as any
    
    // Set directly on the part (per SDK type definition and API requirements)
    // DO NOT set in extra_content as it causes "Unknown name extra_content" API errors
    if ('functionCall' in part) {
        partWithSignature.thoughtSignature = thoughtSignature
    }
    
    return partWithSignature as Part
}

export function getMessageAuthor(message: BaseMessage) {
    const type = message._getType()
    if (ChatMessage.isInstance(message)) {
        return message.role
    }
    if (type === 'tool') {
        return type
    }
    return message.name ?? type
}

/**
 * Maps a message type to a Google Generative AI chat author.
 * @param message The message to map.
 * @param model The model to use for mapping.
 * @returns The message type mapped to a Google Generative AI chat author.
 */
export function convertAuthorToRole(author: string): (typeof POSSIBLE_ROLES)[number] {
    switch (author) {
        /**
         *  Note: Gemini currently is not supporting system messages
         *  we will convert them to human messages and merge with following
         * */
        case 'supervisor':
        case 'ai':
        case 'model': // getMessageAuthor returns message.name. code ex.: return message.name ?? type;
            return 'model'
        case 'system':
            return 'system'
        case 'human':
            return 'user'
        case 'tool':
        case 'function':
            return 'function'
        default:
            return 'user' // return user as default instead of throwing error
    }
}

function messageContentMedia(content: MessageContentComplex): Part {
    if ('mimeType' in content && 'data' in content) {
        return {
            inlineData: {
                mimeType: content.mimeType,
                data: content.data
            }
        }
    }
    if ('mimeType' in content && 'fileUri' in content) {
        return {
            fileData: {
                mimeType: content.mimeType,
                fileUri: content.fileUri
            }
        }
    }

    throw new Error('Invalid media content')
}

function inferToolNameFromPreviousMessages(message: ToolMessage | ToolMessageChunk, previousMessages: BaseMessage[]): string | undefined {
    return previousMessages
        .map((msg) => {
            if (isAIMessage(msg)) {
                return msg.tool_calls ?? []
            }
            return []
        })
        .flat()
        .find((toolCall) => {
            return toolCall.id === message.tool_call_id
        })?.name
}

function _getStandardContentBlockConverter(isMultimodalModel: boolean) {
    const standardContentBlockConverter: StandardContentBlockConverter<{
        text: TextPart
        image: FileDataPart | InlineDataPart
        audio: FileDataPart | InlineDataPart
        file: FileDataPart | InlineDataPart | TextPart
    }> = {
        providerName: 'Google Gemini',

        fromStandardTextBlock(block) {
            return {
                text: block.text
            }
        },

        fromStandardImageBlock(block): FileDataPart | InlineDataPart {
            if (!isMultimodalModel) {
                throw new Error('This model does not support images')
            }
            if (block.source_type === 'url') {
                const data = parseBase64DataUrl({ dataUrl: block.url })
                if (data) {
                    return {
                        inlineData: {
                            mimeType: data.mime_type,
                            data: data.data
                        }
                    }
                } else {
                    return {
                        fileData: {
                            mimeType: block.mime_type ?? '',
                            fileUri: block.url
                        }
                    }
                }
            }

            if (block.source_type === 'base64') {
                return {
                    inlineData: {
                        mimeType: block.mime_type ?? '',
                        data: block.data
                    }
                }
            }

            throw new Error(`Unsupported source type: ${block.source_type}`)
        },

        fromStandardAudioBlock(block): FileDataPart | InlineDataPart {
            if (!isMultimodalModel) {
                throw new Error('This model does not support audio')
            }
            if (block.source_type === 'url') {
                const data = parseBase64DataUrl({ dataUrl: block.url })
                if (data) {
                    return {
                        inlineData: {
                            mimeType: data.mime_type,
                            data: data.data
                        }
                    }
                } else {
                    return {
                        fileData: {
                            mimeType: block.mime_type ?? '',
                            fileUri: block.url
                        }
                    }
                }
            }

            if (block.source_type === 'base64') {
                return {
                    inlineData: {
                        mimeType: block.mime_type ?? '',
                        data: block.data
                    }
                }
            }

            throw new Error(`Unsupported source type: ${block.source_type}`)
        },

        fromStandardFileBlock(block): FileDataPart | InlineDataPart | TextPart {
            if (!isMultimodalModel) {
                throw new Error('This model does not support files')
            }
            if (block.source_type === 'text') {
                return {
                    text: block.text
                }
            }
            if (block.source_type === 'url') {
                const data = parseBase64DataUrl({ dataUrl: block.url })
                if (data) {
                    return {
                        inlineData: {
                            mimeType: data.mime_type,
                            data: data.data
                        }
                    }
                } else {
                    return {
                        fileData: {
                            mimeType: block.mime_type ?? '',
                            fileUri: block.url
                        }
                    }
                }
            }

            if (block.source_type === 'base64') {
                return {
                    inlineData: {
                        mimeType: block.mime_type ?? '',
                        data: block.data
                    }
                }
            }
            throw new Error(`Unsupported source type: ${block.source_type}`)
        }
    }
    return standardContentBlockConverter
}

function _convertLangChainContentToPart(content: MessageContentComplex, isMultimodalModel: boolean): Part | undefined {
    if (isDataContentBlock(content)) {
        return convertToProviderContentBlock(content, _getStandardContentBlockConverter(isMultimodalModel))
    }

    if (content.type === 'text') {
        return { text: content.text }
    } else if (content.type === 'executableCode') {
        return { executableCode: content.executableCode }
    } else if (content.type === 'codeExecutionResult') {
        return { codeExecutionResult: content.codeExecutionResult }
    } else if (content.type === 'image_url') {
        if (!isMultimodalModel) {
            throw new Error(`This model does not support images`)
        }
        let source
        if (typeof content.image_url === 'string') {
            source = content.image_url
        } else if (typeof content.image_url === 'object' && 'url' in content.image_url) {
            source = content.image_url.url
        } else {
            throw new Error('Please provide image as base64 encoded data URL')
        }
        const [dm, data] = source.split(',')
        if (!dm.startsWith('data:')) {
            throw new Error('Please provide image as base64 encoded data URL')
        }

        const [mimeType, encoding] = dm.replace(/^data:/, '').split(';')
        if (encoding !== 'base64') {
            throw new Error('Please provide image as base64 encoded data URL')
        }

        return {
            inlineData: {
                data,
                mimeType
            }
        }
    } else if (content.type === 'media') {
        return messageContentMedia(content)
    } else if (content.type === 'tool_use') {
        return {
            functionCall: {
                name: content.name,
                args: content.input
            }
        }
    } else if (
        content.type?.includes('/') &&
        // Ensure it's a single slash.
        content.type.split('/').length === 2 &&
        'data' in content &&
        typeof content.data === 'string'
    ) {
        return {
            inlineData: {
                mimeType: content.type,
                data: content.data
            }
        }
    } else if ('functionCall' in content) {
        // No action needed here — function calls will be added later from message.tool_calls
        return undefined
    } else {
        if ('type' in content) {
            throw new Error(`Unknown content type ${content.type}`)
        } else {
            throw new Error(`Unknown content ${JSON.stringify(content)}`)
        }
    }
}

export function convertMessageContentToParts(message: BaseMessage, isMultimodalModel: boolean, previousMessages: BaseMessage[]): Part[] {
    if (isToolMessage(message)) {
        const messageName = message.name ?? inferToolNameFromPreviousMessages(message, previousMessages)
        if (messageName === undefined) {
            throw new Error(
                `Google requires a tool name for each tool call response, and we could not infer a called tool name for ToolMessage "${message.id}" from your passed messages. Please populate a "name" field on that ToolMessage explicitly.`
            )
        }

        const result = Array.isArray(message.content)
            ? (message.content.map((c) => _convertLangChainContentToPart(c, isMultimodalModel)).filter((p) => p !== undefined) as Part[])
            : message.content

        if (message.status === 'error') {
            return [
                {
                    functionResponse: {
                        name: messageName,
                        // The API expects an object with an `error` field if the function call fails.
                        // `error` must be a valid object (not a string or array), so we wrap `message.content` here
                        response: { error: { details: result } }
                    }
                }
            ]
        }

        return [
            {
                functionResponse: {
                    name: messageName,
                    // again, can't have a string or array value for `response`, so we wrap it as an object here
                    response: { result }
                }
            }
        ]
    }

    let functionCalls: FunctionCallPart[] = []
    const messageParts: Part[] = []

    if (typeof message.content === 'string' && message.content) {
        messageParts.push({ text: message.content })
    }

    if (Array.isArray(message.content)) {
        messageParts.push(
            ...(message.content.map((c) => _convertLangChainContentToPart(c, isMultimodalModel)).filter((p) => p !== undefined) as Part[])
        )
    }

    if (isAIMessage(message) && message.tool_calls?.length) {
        // Extract thought signatures from additional_kwargs if they exist
        // Thought signatures are stored per tool call, keyed by tool call index or id
        const thoughtSignatures = (message.additional_kwargs?.thought_signatures as Record<string, string>) || {}
        
        // For Gemini 3 Pro, ALL function calls in the current turn must have thought signatures
        // For parallel calls, only the first one has a signature in the response
        // But when sending back, ALL function calls in the same step need the signature
        // Find the signature to use for all calls (from first call, _first_parallel, or any available)
        let signatureForAllCalls: string | undefined = undefined
        
        // Try to find a signature from various sources
        // 1. Check the special parallel key (stored when we extracted from response)
        signatureForAllCalls = thoughtSignatures['_first_parallel']
        
        // 2. Check the first call's signature (by index 0 or id)
        if (!signatureForAllCalls && message.tool_calls[0]) {
            const firstCall = message.tool_calls[0]
            signatureForAllCalls = (firstCall.id ? thoughtSignatures[firstCall.id] : undefined) || 
                                   thoughtSignatures['0'] ||
                                   thoughtSignatures[0]
        }
        
        // 3. Check any available signature (for sequential calls where each has one)
        if (!signatureForAllCalls) {
            const anySignature = Object.values(thoughtSignatures).find(sig => typeof sig === 'string' && sig.length > 0)
            if (anySignature) {
                signatureForAllCalls = anySignature
            }
        }
        
        functionCalls = message.tool_calls.map((tc, index) => {
            const basePart: any = {
                functionCall: {
                    name: tc.name,
                    args: tc.args
                }
            }
            
            // Get thought signature for this specific tool call (by id or index)
            let thoughtSignature: string | undefined = (tc.id ? thoughtSignatures[tc.id] : undefined) || 
                                                       thoughtSignatures[String(index)] || 
                                                       undefined
            
            // If this call doesn't have its own signature, use the signature for all calls
            // This ensures ALL parallel calls get the signature (required for Gemini 3 Pro)
            if (!thoughtSignature) {
                thoughtSignature = signatureForAllCalls
            }
            
            // According to Google docs for Gemini 3 Pro:
            // "The first functionCall part in each step of the current turn must include its thought_signature"
            // For parallel calls, all calls in the same step should use the first call's signature
            
            // If we still don't have a signature, use a dummy signature as fallback
            // According to Google's FAQ, we can use these dummy signatures to skip validation
            // when transferring from a different model or when signatures weren't captured
            if (!thoughtSignature) {
                // Use dummy signature as last resort to avoid validation errors
                // This is acceptable per Google's documentation for cases where signatures weren't captured
                thoughtSignature = 'skip_thought_signature_validator'
            }
            
            // Always attach the signature (either real or dummy) to avoid validation errors
            return createPartWithThoughtSignature(basePart, thoughtSignature) as FunctionCallPart
        })
    }

    // If there are no function calls but there's a thought signature in the last part,
    // preserve it on the last message part
    if (isAIMessage(message) && !message.tool_calls?.length && messageParts.length > 0) {
        const thoughtSignature = message.additional_kwargs?.thought_signature as string | undefined
        if (thoughtSignature) {
            const lastIndex = messageParts.length - 1
            messageParts[lastIndex] = createPartWithThoughtSignature(messageParts[lastIndex], thoughtSignature)
        }
    }

    return [...messageParts, ...functionCalls]
}

export function convertBaseMessagesToContent(
    messages: BaseMessage[],
    isMultimodalModel: boolean,
    convertSystemMessageToHumanContent: boolean = false
) {
    return messages.reduce<{
        content: Content[]
        mergeWithPreviousContent: boolean
    }>(
        (acc, message, index) => {
            if (!isBaseMessage(message)) {
                throw new Error('Unsupported message input')
            }
            const author = getMessageAuthor(message)
            if (author === 'system' && index !== 0) {
                throw new Error('System message should be the first one')
            }
            const role = convertAuthorToRole(author)

            const prevContent = acc.content[acc.content.length]
            if (!acc.mergeWithPreviousContent && prevContent && prevContent.role === role) {
                throw new Error('Google Generative AI requires alternate messages between authors')
            }

            const parts = convertMessageContentToParts(message, isMultimodalModel, messages.slice(0, index))

            if (acc.mergeWithPreviousContent) {
                const prevContent = acc.content[acc.content.length - 1]
                if (!prevContent) {
                    throw new Error('There was a problem parsing your system message. Please try a prompt without one.')
                }
                prevContent.parts.push(...parts)

                return {
                    mergeWithPreviousContent: false,
                    content: acc.content
                }
            }
            let actualRole = role
            if (actualRole === 'function' || (actualRole === 'system' && !convertSystemMessageToHumanContent)) {
                // GenerativeAI API will throw an error if the role is not "user" or "model."
                actualRole = 'user'
            }
            const content: Content = {
                role: actualRole,
                parts
            }
            return {
                mergeWithPreviousContent: author === 'system' && !convertSystemMessageToHumanContent,
                content: [...acc.content, content]
            }
        },
        { content: [], mergeWithPreviousContent: false }
    ).content
}

export function mapGenerateContentResultToChatResult(
    response: EnhancedGenerateContentResponse,
    extra?: {
        usageMetadata: UsageMetadata | undefined
        chatflowid?: string
        orgId?: string
        chatId?: string
    }
): ChatResult {
    // if rejected or error, return empty generations with reason in filters
    if (!response.candidates || response.candidates.length === 0 || !response.candidates[0]) {
        return {
            generations: [],
            llmOutput: {
                filters: response.promptFeedback
            }
        }
    }

    const functionCalls = response.functionCalls()
    const [candidate] = response.candidates
    const { content: candidateContent, ...generationInfo } = candidate
    let content: MessageContent | undefined
    const artifacts: any[] = []
    
    // Extract thought signatures from parts
    const thoughtSignatures: Record<string, string> = {}
    let lastPartThoughtSignature: string | undefined

    if (Array.isArray(candidateContent?.parts) && candidateContent.parts.length === 1 && candidateContent.parts[0].text) {
        content = candidateContent.parts[0].text
        // Check for thought signature in text part
        lastPartThoughtSignature = extractThoughtSignature(candidateContent.parts[0])
    } else if (Array.isArray(candidateContent?.parts) && candidateContent.parts.length > 0) {
        // Process parts and extract thought signatures
        let functionCallIndex = 0
        
        content = candidateContent.parts.map((p: Part, partIndex: number) => {
            // Extract thought signature from function call parts
            if ('functionCall' in p && p.functionCall) {
                const fc = functionCalls?.[functionCallIndex]
                const thoughtSignature = extractThoughtSignature(p)
                if (thoughtSignature) {
                    // For parallel calls, only first has signature
                    // For sequential calls, each has signature
                    // Store by function call index and ID
                    const fcId = (fc && 'id' in fc && typeof fc.id === 'string') ? fc.id : String(functionCallIndex)
                    thoughtSignatures[fcId] = thoughtSignature
                    thoughtSignatures[String(functionCallIndex)] = thoughtSignature
                    
                    // For parallel calls, also store under a special key so all parallel calls can access it
                    // This ensures that when we send back parallel calls, all can use the first call's signature
                    if (functionCallIndex === 0) {
                        thoughtSignatures['_first_parallel'] = thoughtSignature
                    }
                }
                functionCallIndex++
            }
            
            // Check last part for thought signature (when no function calls)
            if (partIndex === candidateContent.parts.length - 1 && functionCalls?.length === 0) {
                lastPartThoughtSignature = extractThoughtSignature(p)
            }
            if ('text' in p) {
                return {
                    type: 'text',
                    text: p.text
                }
            } else if ('executableCode' in p) {
                return {
                    type: 'executableCode',
                    executableCode: p.executableCode
                }
            } else if ('codeExecutionResult' in p) {
                return {
                    type: 'codeExecutionResult',
                    codeExecutionResult: p.codeExecutionResult
                }
            } else if ('inlineData' in p) {
                // Handle inlineData for artifacts
                const inlineData = p.inlineData
                if (inlineData && inlineData.mimeType && inlineData.data) {
                    // Determine file extension based on mime type
                    let fileExtension = ''
                    if (inlineData.mimeType === 'image/png') {
                        fileExtension = 'png'
                    } else if (inlineData.mimeType === 'image/jpeg' || inlineData.mimeType === 'image/jpg') {
                        fileExtension = 'jpg'
                    } else if (inlineData.mimeType === 'image/gif') {
                        fileExtension = 'gif'
                    } else if (inlineData.mimeType === 'image/webp') {
                        fileExtension = 'webp'
                    } else if (inlineData.mimeType === 'image/svg+xml') {
                        fileExtension = 'svg'
                    } else {
                        // Default to png if unknown
                        fileExtension = 'png'
                    }

                    const filename = `artifact_${Date.now()}.${fileExtension}`
                    
                    // Store the artifact if we have the required context
                    if (extra?.chatflowid && extra?.orgId && extra?.chatId) {
                        try {
                            const imageData = Buffer.from(inlineData.data, 'base64')
                            addSingleFileToStorage(
                                inlineData.mimeType,
                                imageData,
                                filename,
                                extra.orgId,
                                extra.chatflowid,
                                extra.chatId
                            ).then(({ path }: { path: string }) => {
                                artifacts.push({ type: fileExtension, data: path })
                            }).catch((error: any) => {
                                console.error('Error storing artifact:', error)
                            })
                        } catch (error) {
                            console.error('Error processing inlineData artifact:', error)
                        }
                    } else {
                        // If no context available, add base64 data as fallback
                        artifacts.push({ 
                            type: fileExtension, 
                            data: `data:${inlineData.mimeType};base64,${inlineData.data}`,
                            mimeType: inlineData.mimeType
                        })
                    }
                }
                
                return {
                    type: 'inlineData',
                    inlineData: p.inlineData
                }
            }
            return p
        })
    } else {
        // no content returned - likely due to abnormal stop reason, e.g. malformed function call
        content = []
    }

    let text = ''
    if (typeof content === 'string') {
        text = content
    } else if (Array.isArray(content) && content.length > 0) {
        const block = content.find((b) => 'text' in b) as { text: string } | undefined
        text = block?.text ?? text
    }

    // Don't append artifacts to text - they will be handled separately via llmOutput.artifacts
    const finalText = text

    // Prepare additional_kwargs with thought signatures
    const additionalKwargs: Record<string, any> = {
        ...generationInfo
    }
    
    // Store thought signatures if any were found
    if (Object.keys(thoughtSignatures).length > 0) {
        additionalKwargs.thought_signatures = thoughtSignatures
    }
    
    // Store last part thought signature if no function calls
    if (lastPartThoughtSignature && functionCalls?.length === 0) {
        additionalKwargs.thought_signature = lastPartThoughtSignature
    }

    const generation: ChatGeneration = {
        text: finalText,
        message: new AIMessage({
            content: content ?? '',
            tool_calls: functionCalls?.map((fc: any) => {
                return {
                    ...fc,
                    type: 'tool_call',
                    id: 'id' in fc && typeof fc.id === 'string' ? fc.id : uuidv4()
                }
            }),
            additional_kwargs: additionalKwargs,
            usage_metadata: extra?.usageMetadata
        }),
        generationInfo
    }

    const result: ChatResult = {
        generations: [generation],
        llmOutput: {
            tokenUsage: {
                promptTokens: extra?.usageMetadata?.input_tokens,
                completionTokens: extra?.usageMetadata?.output_tokens,
                totalTokens: extra?.usageMetadata?.total_tokens
            }
        }
    }

    // Add artifacts to the result if any exist
    if (artifacts.length > 0 && result.llmOutput) {
        result.llmOutput.artifacts = artifacts
    }

    return result
}

export async function convertResponseContentToChatGenerationChunk(
    response: EnhancedGenerateContentResponse,
    extra: {
        usageMetadata?: UsageMetadata | undefined
        index: number
        chatflowid?: string
        orgId?: string
        chatId?: string
    }
): Promise<ChatGenerationChunk | null> {
    if (!response.candidates || response.candidates.length === 0) {
        return null
    }
    const functionCalls = response.functionCalls()
    const [candidate] = response.candidates
    const { content: candidateContent, ...generationInfo } = candidate
    let content: MessageContent | undefined
    
    // Extract thought signatures from parts for streaming
    const thoughtSignatures: Record<string, string> = {}
    let lastPartThoughtSignature: string | undefined
    
    // Checks if some parts do not have text. If false, it means that the content is a string.
    if (Array.isArray(candidateContent?.parts) && candidateContent.parts.every((p: Part) => 'text' in p)) {
        content = candidateContent.parts.map((p: Part) => p.text).join('')
        // Check last part for thought signature
        if (candidateContent.parts.length > 0) {
            lastPartThoughtSignature = extractThoughtSignature(candidateContent.parts[candidateContent.parts.length - 1])
        }
    } else if (Array.isArray(candidateContent?.parts)) {
        content = []
        let functionCallIndex = 0
        for (let partIndex = 0; partIndex < candidateContent.parts.length; partIndex++) {
            const p = candidateContent.parts[partIndex]
            
            // Extract thought signature from function call parts
            if ('functionCall' in p && p.functionCall) {
                const thoughtSignature = extractThoughtSignature(p)
                if (thoughtSignature) {
                    const fc = functionCalls?.[functionCallIndex]
                    const fcId = (fc && 'id' in fc && typeof fc.id === 'string') ? fc.id : String(functionCallIndex)
                    thoughtSignatures[fcId] = thoughtSignature
                    thoughtSignatures[String(functionCallIndex)] = thoughtSignature
                    
                    // For parallel calls, also store under a special key so all parallel calls can access it
                    if (functionCallIndex === 0) {
                        thoughtSignatures['_first_parallel'] = thoughtSignature
                    }
                }
                functionCallIndex++
            }
            
            // Check last part for thought signature (when no function calls)
            if (partIndex === candidateContent.parts.length - 1 && functionCalls?.length === 0) {
                lastPartThoughtSignature = extractThoughtSignature(p)
            }
            if ('text' in p) {
                content.push({
                    type: 'text',
                    text: p.text
                })
            } else if ('executableCode' in p) {
                content.push({
                    type: 'executableCode',
                    executableCode: p.executableCode
                })
            } else if ('codeExecutionResult' in p) {
                content.push({
                    type: 'codeExecutionResult',
                    codeExecutionResult: p.codeExecutionResult
                })
            } else if ('inlineData' in p) {
                // Handle inlineData for artifacts
                const inlineData = p.inlineData
                if (inlineData && inlineData.mimeType && inlineData.data) {
                    // Determine file extension based on mime type
                    let fileExtension = ''
                    if (inlineData.mimeType === 'image/png') {
                        fileExtension = 'png'
                    } else if (inlineData.mimeType === 'image/jpeg' || inlineData.mimeType === 'image/jpg') {
                        fileExtension = 'jpg'
                    } else if (inlineData.mimeType === 'image/gif') {
                        fileExtension = 'gif'
                    } else if (inlineData.mimeType === 'image/webp') {
                        fileExtension = 'webp'
                    } else if (inlineData.mimeType === 'image/svg+xml') {
                        fileExtension = 'svg'
                    } else {
                        // Default to png if unknown
                        fileExtension = 'png'
                    }

                    const filename = `artifact_${Date.now()}.${fileExtension}`
                    
                    // Store the artifact if we have the required context
                    if (extra?.chatflowid && extra?.orgId && extra?.chatId) {
                        try {
                            const imageData = Buffer.from(inlineData.data, 'base64')
                            const {path} = await addSingleFileToStorage(
                                inlineData.mimeType,
                                imageData,
                                filename,
                                extra.orgId,
                                extra.chatflowid,
                                extra.chatId
                            )
                            // Add artifact to content as a special artifact block
                            content.push({
                                type: 'artifact',
                                artifact: { type: fileExtension, data: path }
                            })
                        } catch (error) {
                            console.error('Error storing artifact:', error)
                            // Fallback to base64 data
                            content.push({
                                type: 'artifact',
                                artifact: { 
                                    type: fileExtension, 
                                    data: `data:${inlineData.mimeType};base64,${inlineData.data}`,
                                    mimeType: inlineData.mimeType
                                }
                            })
                        }
                    } else {
                        // If no context available, add base64 data as fallback
                        content.push({
                            type: 'artifact',
                            artifact: { 
                                type: fileExtension, 
                                data: `data:${inlineData.mimeType};base64,${inlineData.data}`,
                                mimeType: inlineData.mimeType
                            }
                        })
                    }
                } else {
                    // Regular inlineData handling
                    content.push({
                        type: 'inlineData',
                        inlineData: p.inlineData
                    })
                }
            } else {
                // Handle other part types
                content.push(p)
            }
        }
    } else {
        // no content returned - likely due to abnormal stop reason, e.g. malformed function call
        content = []
    }

    let text = ''
    const artifacts: any[] = []
    
    if (content && typeof content === 'string') {
        text = content
    } else if (Array.isArray(content)) {
        const textBlock = content.find((b) => 'text' in b) as { text: string } | undefined
        text = textBlock?.text ?? ''
        
        // Extract artifacts from content
        const artifactBlocks = content.filter((b) => 'artifact' in b) as { artifact: any }[]
        artifacts.push(...artifactBlocks.map(block => block.artifact))
    }

    // Don't append artifacts to text in streaming - they will be handled separately
    const finalText = text

    const toolCallChunks: ToolCallChunk[] = []
    if (functionCalls) {
        toolCallChunks.push(
            ...functionCalls.map((fc: any) => ({
                ...fc,
                args: JSON.stringify(fc.args),
                index: extra.index,
                type: 'tool_call_chunk' as const,
                id: 'id' in fc && typeof fc.id === 'string' ? fc.id : uuidv4()
            }))
        )
    }
    
    // Prepare additional_kwargs with thought signatures for streaming
    const additionalKwargs: Record<string, any> = {}
    if (Object.keys(thoughtSignatures).length > 0) {
        additionalKwargs.thought_signatures = thoughtSignatures
    }
    if (lastPartThoughtSignature && functionCalls?.length === 0) {
        additionalKwargs.thought_signature = lastPartThoughtSignature
    }

    return new ChatGenerationChunk({
        text: finalText,
        message: new AIMessageChunk({
            content: content || '',
            name: !candidateContent ? undefined : candidateContent.role,
            tool_call_chunks: toolCallChunks,
            // Store thought signatures in additional_kwargs for later retrieval
            additional_kwargs: additionalKwargs,
            usage_metadata: extra.usageMetadata
        }),
        generationInfo
    })
}

export function convertToGenerativeAITools(tools: GoogleGenerativeAIToolType[]): GoogleGenerativeAIFunctionDeclarationsTool[] {
    if (tools.every((tool) => 'functionDeclarations' in tool && Array.isArray(tool.functionDeclarations))) {
        return tools as GoogleGenerativeAIFunctionDeclarationsTool[]
    }
    return [
        {
            functionDeclarations: tools.map((tool): GenerativeAIFunctionDeclaration => {
                if (isLangChainTool(tool)) {
                    const jsonSchema = schemaToGenerativeAIParameters(tool.schema)
                    if (jsonSchema.type === 'object' && 'properties' in jsonSchema && Object.keys(jsonSchema.properties).length === 0) {
                        return {
                            name: tool.name,
                            description: tool.description
                        }
                    }
                    return {
                        name: tool.name,
                        description: tool.description,
                        parameters: jsonSchema
                    }
                }
                if (isOpenAITool(tool)) {
                    return {
                        name: tool.function.name,
                        description: tool.function.description ?? `A function available to call.`,
                        parameters: jsonSchemaToGeminiParameters(tool.function.parameters)
                    }
                }
                return tool as unknown as GenerativeAIFunctionDeclaration
            })
        }
    ]
}
