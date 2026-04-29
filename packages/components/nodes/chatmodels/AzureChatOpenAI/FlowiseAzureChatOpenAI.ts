import { AzureChatOpenAI as LangchainAzureChatOpenAI, OpenAIChatInput, AzureOpenAIInput, ClientOptions } from '@langchain/openai'
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import type { BaseMessage } from '@langchain/core/messages'
import type { ChatGenerationChunk } from '@langchain/core/outputs'
import { IMultiModalOption, IVisionChatModal } from '../../../src'
import { BaseChatModelParams } from '@langchain/core/language_models/chat_models'
import { flowiseMergeOpenAIReasoningParams, patchInnerOpenAIReasoningDelegates } from '../ChatOpenAI/flowiseOpenAIReasoning'

export class AzureChatOpenAI extends LangchainAzureChatOpenAI implements IVisionChatModal {
    configuredModel: string
    configuredMaxToken?: number
    multiModalOption: IMultiModalOption
    builtInTools: Record<string, any>[] = []
    id: string

    constructor(
        id: string,
        fields?: Partial<OpenAIChatInput> &
            Partial<AzureOpenAIInput> & {
                openAIApiKey?: string
                openAIApiVersion?: string
                openAIBasePath?: string
                deploymentName?: string
            } & BaseChatModelParams & {
                configuration?: ClientOptions
            }
    ) {
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

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    _getReasoningParams(options?: any) {
        return flowiseMergeOpenAIReasoningParams(this, options)
    }

    /** Same as Flowise ChatOpenAI: Responses API stream must forward handleLLMNewToken for SSE. */
    async *_streamResponseChunks(
        messages: BaseMessage[],
        options: Parameters<LangchainAzureChatOpenAI['_streamResponseChunks']>[1],
        runManager?: CallbackManagerForLLMRun
    ): AsyncGenerator<ChatGenerationChunk> {
        const combinedOptions = (this as unknown as { _combineCallOptions: (o: typeof options) => typeof options })._combineCallOptions(
            options
        )
        const self = this as unknown as {
            _useResponsesApi: (opts: typeof combinedOptions) => boolean
            responses: { _streamResponseChunks: typeof LangchainAzureChatOpenAI.prototype._streamResponseChunks }
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
