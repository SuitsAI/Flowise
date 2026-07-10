import { context, ROOT_CONTEXT } from '@opentelemetry/api'
import { CallbackHandler } from '@langfuse/langchain'
import { Serialized } from '@langchain/core/load/serializable'
import { ChainValues } from '@langchain/core/utils/types'
import { BaseMessage } from '@langchain/core/messages'
import { LLMResult } from '@langchain/core/outputs'
import { extractLangfuseToolsFromExtraParams, normalizeLangfuseToolCalls } from './langfuseTools'

function normalizeLLMResultToolCalls(output: LLMResult): LLMResult {
    try {
        const generations = output.generations?.map((generationGroup) =>
            generationGroup.map((generation) => {
                const message = (generation as { message?: BaseMessage }).message
                if (!message || !('tool_calls' in message) || !Array.isArray((message as any).tool_calls)) {
                    return generation
                }

                const normalizedCalls = normalizeLangfuseToolCalls((message as any).tool_calls)
                const clonedMessage = Object.create(Object.getPrototypeOf(message))
                Object.assign(clonedMessage, message, { tool_calls: normalizedCalls })

                return { ...generation, message: clonedMessage }
            })
        )

        return generations ? { ...output, generations } : output
    } catch {
        return output
    }
}

/**
 * @langfuse/langchain's CallbackHandler falls back to the ambient OTel `context.active()`
 * as the parent span whenever a call has no `parentRunId` (i.e. it's the root of a trace).
 * If some other instrumentation left a span active in that async context, the "root" Langfuse
 * trace silently nests under it instead of starting a fresh trace. Detach root calls onto
 * `ROOT_CONTEXT` so every top-level Flowise run always starts its own trace.
 *
 * Also attaches bound tools onto generation input (`{ messages, tools }`) so Langfuse's
 * Tools UI can extract tool definitions — upstream JS CallbackHandler only stores messages.
 * @see https://opentelemetry.io/docs/languages/js/context/
 */
export class DetachedLangfuseCallbackHandler extends CallbackHandler {
    private runDetached<T>(parentRunId: string | undefined, run: () => T): T {
        if (parentRunId) return run()
        return context.with(ROOT_CONTEXT, run)
    }

    private getObservation(runId: string): { update?: (attrs: Record<string, unknown>) => unknown } | undefined {
        return (this as any).runMap?.get(runId)
    }

    handleChainStart(
        chain: Serialized,
        inputs: ChainValues,
        runId: string,
        parentRunId?: string,
        tags?: string[],
        metadata?: Record<string, unknown>,
        runType?: string,
        name?: string
    ): Promise<void> {
        return this.runDetached(parentRunId, () => super.handleChainStart(chain, inputs, runId, parentRunId, tags, metadata, runType, name))
    }

    handleLLMStart(
        llm: Serialized,
        prompts: string[],
        runId: string,
        parentRunId?: string,
        extraParams?: Record<string, unknown>,
        tags?: string[],
        metadata?: Record<string, unknown>,
        name?: string
    ): Promise<void> {
        return this.runDetached(parentRunId, () => super.handleLLMStart(llm, prompts, runId, parentRunId, extraParams, tags, metadata, name))
    }

    handleChatModelStart(
        llm: Serialized,
        messages: BaseMessage[][],
        runId: string,
        parentRunId?: string,
        extraParams?: Record<string, unknown>,
        tags?: string[],
        metadata?: Record<string, unknown>,
        name?: string
    ): Promise<void> {
        return this.runDetached(parentRunId, () =>
            super.handleChatModelStart(llm, messages, runId, parentRunId, extraParams, tags, metadata, name)
        )
    }

    /**
     * Called by upstream handleChatModelStart / handleLLMStart via polymorphic `this`.
     * After the parent creates the generation with messages-only input, rewrite input to
     * include normalized tool definitions when present.
     */
    async handleGenerationStart(
        llm: Serialized,
        messages: unknown[],
        runId: string,
        parentRunId?: string,
        extraParams?: Record<string, unknown>,
        tags?: string[],
        metadata?: Record<string, unknown>,
        name?: string
    ): Promise<void> {
        await super.handleGenerationStart(llm, messages as any, runId, parentRunId, extraParams, tags, metadata, name)

        const tools = extractLangfuseToolsFromExtraParams(extraParams)
        if (!tools.length) return

        const observation = this.getObservation(runId)
        observation?.update?.({ input: { messages, tools } })
    }

    async handleLLMEnd(output: LLMResult, runId: string, parentRunId?: string): Promise<void> {
        return super.handleLLMEnd(normalizeLLMResultToolCalls(output), runId, parentRunId)
    }

    handleToolStart(
        tool: Serialized,
        input: string,
        runId: string,
        parentRunId?: string,
        tags?: string[],
        metadata?: Record<string, unknown>,
        name?: string
    ): Promise<void> {
        return this.runDetached(parentRunId, () => super.handleToolStart(tool, input, runId, parentRunId, tags, metadata, name))
    }

    handleRetrieverStart(
        retriever: Serialized,
        query: string,
        runId: string,
        parentRunId?: string,
        tags?: string[],
        metadata?: Record<string, unknown>,
        name?: string
    ): Promise<void> {
        return this.runDetached(parentRunId, () => super.handleRetrieverStart(retriever, query, runId, parentRunId, tags, metadata, name))
    }
}

export {
    buildLangfuseGenerationInput,
    extractLangfuseToolsFromExtraParams,
    normalizeLangfuseToolCalls,
    toLangfuseTool
} from './langfuseTools'
export type { LangfuseToolDefinition } from './langfuseTools'
