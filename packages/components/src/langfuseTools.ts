/**
 * Flat tool shape expected by Langfuse's extractToolsFromObservation / Tools UI.
 * @see https://github.com/langfuse/langfuse-python/pull/1568
 */
export type LangfuseToolDefinition = {
    name: string
    description?: string
    parameters?: unknown
}

/**
 * Normalize OpenAI (`function` wrapper) and Anthropic (`input_schema`) tool defs
 * into Langfuse's flat `{ name, description, parameters }` schema.
 */
export function toLangfuseTool(tool: unknown): LangfuseToolDefinition | null {
    if (!tool || typeof tool !== 'object') return null

    const raw = tool as Record<string, unknown>
    const fn =
        raw.function && typeof raw.function === 'object' ? (raw.function as Record<string, unknown>) : raw

    if (typeof fn.name !== 'string' || !fn.name) return null

    // `schema` covers Flowise Agentflow's ISimpliefiedTool shape
    const parameters = fn.parameters ?? fn.input_schema ?? fn.inputSchema ?? fn.schema
    const definition: LangfuseToolDefinition = { name: fn.name }

    if (typeof fn.description === 'string') definition.description = fn.description
    if (parameters !== undefined) definition.parameters = parameters

    return definition
}

/**
 * Pull bound tools from LangChain's `extraParams.invocation_params` and normalize
 * them for Langfuse generation input.
 */
export function extractLangfuseToolsFromExtraParams(extraParams?: Record<string, unknown>): LangfuseToolDefinition[] {
    const invocationParams = extraParams?.invocation_params as Record<string, unknown> | undefined
    if (!invocationParams) return []

    const rawTools = invocationParams.tools ?? invocationParams.functions
    if (!Array.isArray(rawTools) || rawTools.length === 0) return []

    const tools: LangfuseToolDefinition[] = []
    const seen = new Set<string>()

    for (const raw of rawTools) {
        const tool = toLangfuseTool(raw)
        if (!tool || seen.has(tool.name)) continue
        seen.add(tool.name)
        tools.push(tool)
    }

    return tools
}

/**
 * Structure generation input so Langfuse can find tool definitions at `input.tools`.
 * Without this, the Tools tab stays empty even when the model emits tool_calls.
 */
export function buildLangfuseGenerationInput(messages: unknown, extraParams?: Record<string, unknown>): unknown {
    const tools = extractLangfuseToolsFromExtraParams(extraParams)
    if (!tools.length) return messages
    return { messages, tools }
}

/**
 * Convert LangChain tool_calls (`args`) to Langfuse's preferred shape (`arguments` JSON string).
 */
export function normalizeLangfuseToolCalls(toolCalls: unknown): unknown {
    if (!Array.isArray(toolCalls)) return toolCalls

    return toolCalls.map((call) => {
        if (!call || typeof call !== 'object') return call
        const c = call as Record<string, unknown>
        const rawArgs = c.arguments ?? c.args ?? c.input
        const argumentsJson = typeof rawArgs === 'string' ? rawArgs : JSON.stringify(rawArgs ?? {})

        return {
            id: c.id,
            type: c.type ?? 'function',
            name: c.name,
            arguments: argumentsJson
        }
    })
}
