import { ChatOpenAI as LangchainChatOpenAI, ChatOpenAIFields } from '@langchain/openai'
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import type { BaseMessage } from '@langchain/core/messages'
import type { ChatGenerationChunk } from '@langchain/core/outputs'
import { IMultiModalOption, IVisionChatModal } from '../../../src'
import { flowiseMergeOpenAIReasoningParams, patchInnerOpenAIReasoningDelegates } from './flowiseOpenAIReasoning'

export class ChatOpenAI extends LangchainChatOpenAI implements IVisionChatModal {
    configuredModel: string
    configuredMaxToken?: number
    multiModalOption: IMultiModalOption
    builtInTools: Record<string, any>[] = []
    id: string

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
            for await (const chunk of self.responses._streamResponseChunks(messages, combinedOptions)) {
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
}
