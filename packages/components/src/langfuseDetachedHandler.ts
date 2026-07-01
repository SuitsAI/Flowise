import { context, ROOT_CONTEXT } from '@opentelemetry/api'
import { CallbackHandler } from '@langfuse/langchain'
import { Serialized } from '@langchain/core/load/serializable'
import { ChainValues } from '@langchain/core/utils/types'
import { BaseMessage } from '@langchain/core/messages'

/**
 * @langfuse/langchain's CallbackHandler falls back to the ambient OTel `context.active()`
 * as the parent span whenever a call has no `parentRunId` (i.e. it's the root of a trace).
 * If some other instrumentation left a span active in that async context, the "root" Langfuse
 * trace silently nests under it instead of starting a fresh trace. Detach root calls onto
 * `ROOT_CONTEXT` so every top-level Flowise run always starts its own trace.
 * @see https://opentelemetry.io/docs/languages/js/context/
 */
export class DetachedLangfuseCallbackHandler extends CallbackHandler {
    private runDetached<T>(parentRunId: string | undefined, run: () => T): T {
        if (parentRunId) return run()
        return context.with(ROOT_CONTEXT, run)
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
