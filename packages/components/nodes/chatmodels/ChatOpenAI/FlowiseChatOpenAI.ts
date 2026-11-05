import { ChatOpenAI as LangchainChatOpenAI, ChatOpenAIFields } from '@langchain/openai'
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import type { BaseMessage } from '@langchain/core/messages'
import type { ChatGenerationChunk, ChatResult } from '@langchain/core/outputs'
import { IMultiModalOption, IVisionChatModal } from '../../../src'
import { flowiseMergeOpenAIReasoningParams, patchInnerOpenAIReasoningDelegates } from './flowiseOpenAIReasoning'
import {
    attachImageGenerationArtifacts,
    isImageGenerationTool,
    type OpenAIImageFlowContext
} from './openaiImageGeneration'

export class ChatOpenAI extends LangchainChatOpenAI implements IVisionChatModal {
    configuredModel: string
    configuredMaxToken?: number
    multiModalOption: IMultiModalOption
    builtInTools: Record<string, any>[] = []
    id: string
    imageGenerationTool?: Record<string, unknown>
    flowContext?: OpenAIImageFlowContext
    private imageArtifactIds = new Set<string>()

    constructor(id: string, fields?: ChatOpenAIFields) {
        super(fields)
        patchInnerOpenAIReasoningDelegates(this)
        this.id = id
        this.configuredModel = fields?.modelName ?? ''
        this.configuredMaxToken = fields?.maxTokens
    }

    revertToOriginalModel(): void {
        this.model = this.configuredModel
        this.maxTokens = this.configuredMaxToken
    }

    setMultiModalOption(multiModalOption: IMultiModalOption): void {
        this.multiModalOption = multiModalOption
    }

    setVisionModel(): void {
        // pass
    }

    addBuiltInTools(builtInTool: Record<string, any>): void {
        this.builtInTools.push(builtInTool)
    }

    setFlowContext(context: OpenAIImageFlowContext): void {
        this.flowContext = context
    }

    setImageGenerationTool(tool: Record<string, unknown> | undefined): void {
        this.imageGenerationTool = tool
        if (tool) {
            this.useResponsesApi = true
            this.bindTools([])
        }
    }

    /**
     * Tool Agent replaces bound tools. Keep the image generation tool on every bind
     * so the Responses API still receives it alongside the agent's function tools.
     */
    override bindTools(tools: any[], kwargs?: any): any {
        if (!this.imageGenerationTool) {
            return super.bindTools(tools, kwargs)
        }
        const list = Array.isArray(tools) ? tools : []
        const withoutImage = list.filter((tool) => !isImageGenerationTool(tool))
        return super.bindTools([...withoutImage, this.imageGenerationTool], kwargs)
    }

    /**
     * Same merge as patched inner delegates (LangSmith reads outer `invocationParams`, which still
     * delegates here only indirectly — inner patch fixes the real API payload).
     */
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    _getReasoningParams(options?: any) {
        return flowiseMergeOpenAIReasoningParams(this, options)
    }

    /**
     * LangChain's ChatOpenAI delegates to the Responses API when reasoning summaries are enabled.
     * That path yields chunks but does not call runManager.handleLLMNewToken (unlike the Completions
     * stream). Flowise relies on those callbacks for SSE; without them, streaming appears "dead".
     */
    async *_streamResponseChunks(
        messages: BaseMessage[],
        options: Parameters<LangchainChatOpenAI['_streamResponseChunks']>[1],
        runManager?: CallbackManagerForLLMRun
    ): AsyncGenerator<ChatGenerationChunk> {
        const combinedOptions = (this as unknown as { _combineCallOptions: (o: typeof options) => typeof options })._combineCallOptions(
            options
        )
        const self = this as unknown as {
            _useResponsesApi: (opts: typeof combinedOptions) => boolean
            responses: { _streamResponseChunks: typeof LangchainChatOpenAI.prototype._streamResponseChunks }
        }
        if (self._useResponsesApi(combinedOptions)) {
            this.imageArtifactIds.clear()
            for await (const chunk of self.responses._streamResponseChunks(messages, combinedOptions)) {
                if (this.imageGenerationTool) {
                    await attachImageGenerationArtifacts(chunk, this.flowContext, this.imageArtifactIds)
                }
                const newTokenIndices = {
                    prompt: (options as { promptIndex?: number })?.promptIndex ?? 0,
                    completion: (chunk.generationInfo as { completion?: number } | undefined)?.completion ?? 0
                }
                yield chunk
                await runManager?.handleLLMNewToken(chunk.text ?? '', newTokenIndices, undefined, undefined, undefined, { chunk })
            }
            return
        }
        yield* super._streamResponseChunks(messages, options, runManager)
    }

    /**
     * Image generation forces the Responses API. Consume that stream here so token callbacks
     * still fire, then leave artifact blocks on the final message for the Tool Agent.
     */
    override async _generate(
        messages: BaseMessage[],
        options: Parameters<LangchainChatOpenAI['_generate']>[1],
        runManager?: CallbackManagerForLLMRun
    ): Promise<ChatResult> {
        if (!this.imageGenerationTool) {
            return super._generate(messages, options, runManager)
        }

        if (this.streaming) {
            let finalChunk: ChatGenerationChunk | undefined
            for await (const chunk of this._streamResponseChunks(messages, options, runManager)) {
                finalChunk = finalChunk ? finalChunk.concat(chunk) : chunk
            }
            const usage = (finalChunk?.message as { usage_metadata?: { input_tokens?: number; output_tokens?: number; total_tokens?: number } } | undefined)
                ?.usage_metadata
            return {
                generations: finalChunk ? [finalChunk] : [],
                llmOutput: usage
                    ? {
                          estimatedTokenUsage: {
                              promptTokens: usage.input_tokens ?? 0,
                              completionTokens: usage.output_tokens ?? 0,
                              totalTokens: usage.total_tokens ?? 0
                          }
                      }
                    : undefined
            }
        }

        const result = await super._generate(messages, options, runManager)
        this.imageArtifactIds.clear()
        for (const generation of result.generations) {
            await attachImageGenerationArtifacts(generation, this.flowContext, this.imageArtifactIds)
        }
        return result
    }
}
