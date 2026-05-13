import { ICommonObject, INode, INodeData, INodeParams } from '../../../src/Interface'
import { getBaseClasses, getCredentialData, getCredentialParam, parseWithTypeConversion } from '../../../src/utils'
import { StructuredTool, ToolInputParsingException, ToolParams } from '@langchain/core/tools'
import { Sandbox } from '@e2b/code-interpreter'
import { z } from 'zod'
import { addSingleFileToStorage } from '../../../src/storageUtils'
import { CallbackManager, CallbackManagerForToolRun, Callbacks, parseCallbackConfigArg } from '@langchain/core/callbacks/manager'
import { RunnableConfig } from '@langchain/core/runnables'
import { ARTIFACTS_PREFIX } from '../../../src/agents'

const DESC = `Evaluates python code in a sandbox environment. \
The environment is long running and exists across multiple executions. \
You must send the whole script every time and print your outputs. \
Script should be pure python code that can be evaluated. \
It should be in python format NOT markdown. \
The code should NOT be wrapped in backticks. \
All python packages including requests, matplotlib, scipy, numpy, pandas, \
etc are available. Create and display chart using "plt.show()". \
Save generated files (e.g. plots, CSVs) to the /generated directory so they can be returned as artifacts.`
const NAME = 'code_interpreter'

const GENERATED_DIR = '/generated'

const EXT_TO_MIME: Record<string, string> = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.svg': 'image/svg+xml',
    '.csv': 'text/csv',
    '.json': 'application/json',
    '.html': 'text/html',
    '.htm': 'text/html',
    '.txt': 'text/plain',
    '.pdf': 'application/pdf',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.xls': 'application/vnd.ms-excel'
}

function getMimeFromPath(filePath: string): string {
    const ext = filePath.slice(filePath.lastIndexOf('.')).toLowerCase()
    return EXT_TO_MIME[ext] ?? 'application/octet-stream'
}

/** Same semantics as executeJavaScriptCode in src/utils.ts (default 300000 ms, override via SANDBOX_TIMEOUT). */
function getSandboxTimeoutMs(): number {
    let timeoutMs = 300000
    if (process.env.SANDBOX_TIMEOUT) {
        const parsed = parseInt(process.env.SANDBOX_TIMEOUT, 10)
        if (Number.isFinite(parsed) && parsed > 0) {
            timeoutMs = parsed
        }
    }
    return timeoutMs
}

class Code_Interpreter_Tools implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    inputs: INodeParams[]
    badge: string
    credential: INodeParams

    constructor() {
        this.label = 'Code Interpreter by E2B'
        this.name = 'codeInterpreterE2B'
        this.version = 1.0
        this.type = 'CodeInterpreter'
        this.icon = 'e2b.png'
        this.category = 'Tools'
        this.description = 'Execute code in a sandbox environment'
        this.baseClasses = [this.type, 'Tool', ...getBaseClasses(E2BTool)]
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            credentialNames: ['E2BApi'],
            optional: true
        }
        this.inputs = [
            {
                label: 'Tool Name',
                name: 'toolName',
                type: 'string',
                description: 'Specify the name of the tool',
                default: 'code_interpreter'
            },
            {
                label: 'Tool Description',
                name: 'toolDesc',
                type: 'string',
                rows: 4,
                description: 'Specify the description of the tool',
                default: DESC
            },
            {
                label: 'Sandbox ID',
                name: 'sandboxId',
                type: 'string',
                description: 'Connect to an existing sandbox environment using its ID. This allows reusing the same environment for a specific user or chatflow, maintaining session state and installed packages.',
                default: '',
                optional: true,
                additionalParams: true
            }
        ]
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const toolDesc = nodeData.inputs?.toolDesc as string
        const toolName = nodeData.inputs?.toolName as string
        const sandboxId = nodeData.inputs?.sandboxId as string

        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const e2bApiKey = getCredentialParam('e2bApiKey', credentialData, nodeData)

        return await E2BTool.initialize({
            description: toolDesc ?? DESC,
            name: toolName ?? NAME,
            apiKey: e2bApiKey,
            schema: z.object({
                code: z
                  .string()
                  .min(1, "Code must not be empty")
                  .describe(
                    [
                      "Complete, self-contained Python code to execute in the persistent sandbox.",
                      "Rules:",
                      "- Always send the FULL script, not diffs or fragments (sandbox is stateful across calls)",
                      "- Use print() for all outputs you want to see",
                      "- Save any files (plots, CSVs, etc.) to /generated/ directory",
                      "- Use plt.show() to render charts",
                      "- Do NOT wrap code in markdown backticks or add any prose",
                      "- Never leave this field empty — if there is nothing to run, do not call this tool"
                    ].join("\n")
                  ),
              
                command: z
                  .string()
                  .min(1)
                  .regex(/^[a-zA-Z0-9 _.=<>!@#$%^&*()\-\[\]\/\\|;:'"`,~+]+$/, "Must be a valid shell command")
                  .optional()
                  .describe(
                    [
                      "Optional shell command to run BEFORE the Python code (e.g. to install packages).",
                      "Examples: 'pip install pyarrow', 'apt-get install -y libgdal-dev'",
                      "Only provide when a package or system dependency is needed that is not pre-installed.",
                      "Do NOT use for Python logic — that belongs in `code`."
                    ].join("\n")
                  ),
              }),
            chatflowid: options.chatflowid,
            sandboxId: sandboxId,
            orgId: options.orgId
        })
    }
}

type E2BToolParams = ToolParams
type E2BToolInput = {
    name: string
    description: string
    apiKey: string
    schema: any
    chatflowid: string
    orgId: string
    templateCodeInterpreterE2B?: string
    domainCodeInterpreterE2B?: string
    sandboxId?: string
}

export class E2BTool extends StructuredTool {
    static lc_name() {
        return 'E2BTool'
    }

    name = NAME

    description = DESC

    instance: Sandbox

    apiKey: string

    schema

    chatflowid: string

    orgId: string

    flowObj: ICommonObject

    templateCodeInterpreterE2B?: string
    domainCodeInterpreterE2B?: string
    sandboxId?: string

    constructor(options: E2BToolParams & E2BToolInput) {
        super(options)
        this.description = options.description
        this.name = options.name
        this.apiKey = options.apiKey
        this.schema = options.schema
        this.chatflowid = options.chatflowid
        this.orgId = options.orgId
        this.templateCodeInterpreterE2B = options.templateCodeInterpreterE2B
        this.domainCodeInterpreterE2B = options.domainCodeInterpreterE2B
        this.sandboxId = options.sandboxId
    }

    static async initialize(options: Partial<E2BToolParams> & E2BToolInput) {
        return new this({
            name: options.name,
            description: options.description,
            apiKey: options.apiKey,
            schema: options.schema,
            chatflowid: options.chatflowid,
            orgId: options.orgId,
            templateCodeInterpreterE2B: options.templateCodeInterpreterE2B,
            domainCodeInterpreterE2B: options.domainCodeInterpreterE2B,
            sandboxId: options.sandboxId
        })
    }

    async call(
        arg: z.infer<typeof this.schema>,
        configArg?: RunnableConfig | Callbacks,
        tags?: string[],
        flowConfig?: { sessionId?: string; chatId?: string; input?: string; state?: ICommonObject }
    ): Promise<string> {
        const config = parseCallbackConfigArg(configArg)
        if (config.runName === undefined) {
            config.runName = this.name
        }
        let parsed
        try {
            parsed = await parseWithTypeConversion(this.schema, arg)
        } catch (e) {
            throw new ToolInputParsingException(`Received tool input did not match expected schema`, JSON.stringify(arg))
        }
        const callbackManager_ = await CallbackManager.configure(
            config.callbacks,
            this.callbacks,
            config.tags || tags,
            this.tags,
            config.metadata,
            this.metadata,
            { verbose: this.verbose }
        )
        const runManager = await callbackManager_?.handleToolStart(
            {...this.toJSON(), name: this.name},
            typeof parsed === 'string' ? parsed : JSON.stringify(parsed),
            undefined,
            undefined,
            undefined,
            undefined,
            config.runName
        )
        let result
        try {
            result = await this._call(parsed, runManager, flowConfig)
        } catch (e) {
            await runManager?.handleToolError(e)
            throw e
        }
        if (result && typeof result !== 'string') {
            result = JSON.stringify(result)
        }
        await runManager?.handleToolEnd(result)
        return result
    }

    // @ts-ignore
    protected async _call(
        arg: z.infer<typeof this.schema>,
        _?: CallbackManagerForToolRun,
        flowConfig?: { sessionId?: string; chatId?: string; input?: string }
    ): Promise<string> {
        flowConfig = { ...this.flowObj, ...flowConfig }
        try {
            if ('code' in arg) {
                // this.instance = await CodeInterpreter.create({ apiKey: this.apiKey })
                // const execution = await this.instance.notebook.execCell(arg?.input)

                const sandboxTimeoutMs = getSandboxTimeoutMs()
                if (this.sandboxId && this.sandboxId !== ' ') {
                    // Connect to an existing sandbox if sandboxId is provided
                    this.instance = await Sandbox.connect(this.sandboxId, {
                        apiKey: this.apiKey,
                        requestTimeoutMs: sandboxTimeoutMs
                    })
                } else {
                    // Create a new sandbox if no sandboxId is provided
                    this.instance = await Sandbox.create({
                        apiKey: this.apiKey,
                        timeoutMs: sandboxTimeoutMs,
                        requestTimeoutMs: sandboxTimeoutMs
                    })
                }
                
                // Ensure /generated exists so code can save files there
                await this.instance.files.makeDir(GENERATED_DIR).catch(() => {})

                if (arg?.command) {
                    await this.instance.commands.run(arg?.command);
                }

                const execution = await this.instance.runCode(arg?.code, {
                    language: 'python',
                    timeoutMs: sandboxTimeoutMs,
                    requestTimeoutMs: sandboxTimeoutMs
                })

                const artifacts = []
                for (const result of execution.results) {
                    for (const key in result) {
                        if (!(result as any)[key]) continue

                        if (key === 'png') {
                            //@ts-ignore
                            const pngData = Buffer.from(result.png, 'base64')

                            const filename = `artifact_${Date.now()}.png`

                            // Don't check storage usage because this is incoming file, and if we throw error, agent will keep on retrying
                            const { path } = await addSingleFileToStorage(
                                'image/png',
                                pngData,
                                filename,
                                this.orgId,
                                this.chatflowid,
                                flowConfig!.chatId as string
                            )

                            artifacts.push({ type: 'png', data: path })
                        } else if (key === 'jpeg') {
                            //@ts-ignore
                            const jpegData = Buffer.from(result.jpeg, 'base64')

                            const filename = `artifact_${Date.now()}.jpg`

                            const { path } = await addSingleFileToStorage(
                                'image/jpg',
                                jpegData,
                                filename,
                                this.orgId,
                                this.chatflowid,
                                flowConfig!.chatId as string
                            )

                            artifacts.push({ type: 'jpeg', data: path })
                        } else if (key === 'html' || key === 'markdown' || key === 'latex' || key === 'json' || key === 'javascript') {
                            artifacts.push({ type: key, data: (result as any)[key] })
                        } //TODO: support for pdf
                    }
                }

                // Download all files from /generated folder and add as artifacts (ignore files without extension)
                try {
                    const entries = await this.instance.files.list(GENERATED_DIR, { depth: 100 })
                    const fileEntries = entries.filter((e) => {
                        if (e.type !== 'file') return false
                        const name = e.name
                        const lastDot = name.lastIndexOf('.')
                        return lastDot > 0 && lastDot < name.length - 1
                    })
                    for (const entry of fileEntries) {
                        try {
                            const content = await this.instance.files.read(entry.path, { format: 'bytes' })
                            const buffer = Buffer.from(content as Uint8Array)
                            const fileName = entry.name
                            const mime = getMimeFromPath(entry.path)
                            const { path: storagePath } = await addSingleFileToStorage(
                                mime,
                                buffer,
                                fileName,
                                this.orgId,
                                this.chatflowid,
                                flowConfig!.chatId as string
                            )
                            artifacts.push({ type: 'file', data: storagePath })
                        } catch (readErr) {
                            // Skip single file read errors (e.g. permission, deleted)
                        }
                    }
                } catch {
                    // /generated may not exist or list may fail; skip filesystem artifacts
                }

                // this.instance.close()

                let output = ''

                if (execution.text) output = execution.text
                if (!execution.text && execution.logs.stdout.length) output = execution.logs.stdout.join('\n')

                if (execution.error) {
                    return `${execution.error.name}: ${execution.error.value}`
                }

                return artifacts.length > 0 ? output + ARTIFACTS_PREFIX + JSON.stringify(artifacts) : output
            } else {
                return 'No input provided'
            }
        } catch (e) {
            // if (this.instance) this.instance.close()
            //if (this.instance) this.instance.kill()
            return typeof e === 'string' ? e : JSON.stringify(e, null, 2)
        }
    }

    setFlowObject(flowObj: ICommonObject) {
        this.flowObj = flowObj
    }
}

module.exports = { nodeClass: Code_Interpreter_Tools }
