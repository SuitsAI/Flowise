import {
    buildLangfuseGenerationInput,
    extractLangfuseToolsFromExtraParams,
    normalizeLangfuseToolCalls,
    toLangfuseTool
} from '../../../src/langfuseTools'

describe('langfuse tool helpers', () => {
    describe('toLangfuseTool', () => {
        it('normalizes OpenAI function-wrapped tools', () => {
            expect(
                toLangfuseTool({
                    type: 'function',
                    function: {
                        name: 'get_weather',
                        description: 'Get weather',
                        parameters: { type: 'object', properties: { city: { type: 'string' } } }
                    }
                })
            ).toEqual({
                name: 'get_weather',
                description: 'Get weather',
                parameters: { type: 'object', properties: { city: { type: 'string' } } }
            })
        })

        it('normalizes Anthropic input_schema tools', () => {
            expect(
                toLangfuseTool({
                    name: 'GMAIL_GET_EMAIL_BY_ID',
                    description: 'Get email by id',
                    input_schema: {
                        type: 'object',
                        properties: { messageId: { type: 'string' } }
                    }
                })
            ).toEqual({
                name: 'GMAIL_GET_EMAIL_BY_ID',
                description: 'Get email by id',
                parameters: {
                    type: 'object',
                    properties: { messageId: { type: 'string' } }
                }
            })
        })

        it('normalizes Flowise Agentflow schema tools', () => {
            expect(
                toLangfuseTool({
                    name: 'search',
                    description: 'Search docs',
                    schema: { type: 'object', properties: { q: { type: 'string' } } }
                })
            ).toEqual({
                name: 'search',
                description: 'Search docs',
                parameters: { type: 'object', properties: { q: { type: 'string' } } }
            })
        })

        it('returns null for invalid tools', () => {
            expect(toLangfuseTool(null)).toBeNull()
            expect(toLangfuseTool({ description: 'no name' })).toBeNull()
        })
    })

    describe('extractLangfuseToolsFromExtraParams', () => {
        it('extracts and dedupes tools from invocation_params', () => {
            const tools = extractLangfuseToolsFromExtraParams({
                invocation_params: {
                    tools: [
                        { name: 'a', input_schema: { type: 'object' } },
                        { name: 'a', input_schema: { type: 'object' } },
                        { name: 'b', parameters: { type: 'object' } }
                    ]
                }
            })

            expect(tools.map((t) => t.name)).toEqual(['a', 'b'])
        })

        it('falls back to functions', () => {
            const tools = extractLangfuseToolsFromExtraParams({
                invocation_params: {
                    functions: [{ name: 'legacy_fn', parameters: { type: 'object' } }]
                }
            })

            expect(tools).toEqual([{ name: 'legacy_fn', parameters: { type: 'object' } }])
        })

        it('returns empty when tools are missing', () => {
            expect(extractLangfuseToolsFromExtraParams(undefined)).toEqual([])
            expect(extractLangfuseToolsFromExtraParams({ invocation_params: {} })).toEqual([])
        })
    })

    describe('buildLangfuseGenerationInput', () => {
        it('returns messages unchanged when no tools', () => {
            const messages = [{ role: 'user', content: 'hi' }]
            expect(buildLangfuseGenerationInput(messages)).toBe(messages)
        })

        it('wraps messages with tools when present', () => {
            const messages = [{ role: 'user', content: 'hi' }]
            expect(
                buildLangfuseGenerationInput(messages, {
                    invocation_params: {
                        tools: [{ name: 'search', description: 'Search', parameters: { type: 'object' } }]
                    }
                })
            ).toEqual({
                messages,
                tools: [{ name: 'search', description: 'Search', parameters: { type: 'object' } }]
            })
        })
    })

    describe('normalizeLangfuseToolCalls', () => {
        it('converts LangChain args to arguments JSON string', () => {
            expect(
                normalizeLangfuseToolCalls([
                    {
                        name: 'GMAIL_GET_EMAIL_BY_ID',
                        args: { messageId: 'abc', userId: 'user@example.com' },
                        id: 'toolu_123',
                        type: 'tool_call'
                    }
                ])
            ).toEqual([
                {
                    id: 'toolu_123',
                    type: 'tool_call',
                    name: 'GMAIL_GET_EMAIL_BY_ID',
                    arguments: JSON.stringify({ messageId: 'abc', userId: 'user@example.com' })
                }
            ])
        })
    })
})
