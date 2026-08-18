import { convertSchemaToZod } from './utils'

describe('convertSchemaToZod', () => {
    it('skips rows without a property name', () => {
        const zodObj = convertSchemaToZod([
            { property: '', type: 'string', description: 'no name', required: true },
            { property: '   ', type: 'number', description: 'blank name', required: false },
            { property: 'query', type: 'string', description: 'search query', required: true }
        ])

        expect(Object.keys(zodObj)).toEqual(['query'])
    })

    it('trims property names', () => {
        const zodObj = convertSchemaToZod([{ property: ' query ', type: 'string', description: 'search query', required: true }])

        expect(Object.keys(zodObj)).toEqual(['query'])
    })

    it('accepts a JSON string schema', () => {
        const zodObj = convertSchemaToZod(
            JSON.stringify([
                { property: '', type: 'boolean', description: '', required: false },
                { property: 'limit', type: 'number', description: 'max results', required: false }
            ])
        )

        expect(Object.keys(zodObj)).toEqual(['limit'])
    })
})
