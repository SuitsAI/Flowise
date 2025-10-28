import { Embeddings, EmbeddingsParams } from '@langchain/core/embeddings'
import { GoogleAuth, GoogleAuthOptions } from 'google-auth-library'
import { buildGoogleCredentials } from '../../../src/google-utils'
import { ICommonObject, INode, INodeData, INodeOptionsValue, INodeParams } from '../../../src/Interface'
import { getBaseClasses } from '../../../src/utils'

interface GoogleVertexAIMultimodalEmbeddingsParams extends EmbeddingsParams {
    modelName?: string
    location?: string
    authOptions?: GoogleAuthOptions
    stripNewLines?: boolean
}

class GoogleVertexAIMultimodalEmbeddings extends Embeddings {
    modelName: string
    location: string
    private auth: GoogleAuth
    stripNewLines: boolean
    private projectId: string

    constructor(params: GoogleVertexAIMultimodalEmbeddingsParams) {
        super(params)
        this.modelName = params.modelName || 'multimodalembedding@001'
        this.location = params.location || 'us-central1'
        this.stripNewLines = params.stripNewLines ?? false
        
        // Initialize Google Auth
        this.auth = new GoogleAuth({
            scopes: ['https://www.googleapis.com/auth/cloud-platform'],
            ...params.authOptions
        })
        
        // Get project ID from auth options or default
        this.projectId = params.authOptions?.projectId || ''
    }

    async embedDocuments(texts: string[]): Promise<number[][]> {
        const processedTexts = this.stripNewLines ? texts.map((text) => text.replace(/\n/g, ' ')) : texts
        
        // Process in batches to avoid rate limits
        const embeddings: number[][] = []
        for (const text of processedTexts) {
            const embedding = await this.embedQuery(text)
            embeddings.push(embedding)
        }
        return embeddings
    }

    async embedQuery(text: string): Promise<number[]> {
        const processedText = this.stripNewLines ? text.replace(/\n/g, ' ') : text
        
        try {
            // Get access token
            const client = await this.auth.getClient()
            const accessToken = await client.getAccessToken()
            
            // Get project ID if not set
            if (!this.projectId) {
                this.projectId = await this.auth.getProjectId()
            }
            
            // Construct the API endpoint
            const endpoint = `https://${this.location}-aiplatform.googleapis.com/v1/projects/${this.projectId}/locations/${this.location}/publishers/google/models/${this.modelName}:predict`
            
            // Make the API call
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${accessToken.token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    instances: [
                        {
                            text: processedText
                        }
                    ],
                    parameters: {
                        dimension: 1408
                    }
                })
            })
            
            if (!response.ok) {
                const errorText = await response.text()
                throw new Error(`Vertex AI API error: ${response.status} - ${errorText}`)
            }
            
            const result = await response.json()
            
            // Extract the text embedding from the response
            if (result.predictions && result.predictions[0] && result.predictions[0].textEmbedding) {
                return result.predictions[0].textEmbedding
            }
            
            throw new Error('Invalid response format from Vertex AI Multimodal Embedding API')
        } catch (error: any) {
            throw new Error(`Failed to generate embedding: ${error.message}`)
        }
    }
}

class GoogleVertexAIMultimodalEmbedding_Embeddings implements INode {
    label: string
    name: string
    version: number
    type: string
    icon: string
    category: string
    description: string
    baseClasses: string[]
    credential: INodeParams
    inputs: INodeParams[]

    constructor() {
        this.label = 'GoogleVertexAI Multimodal Embeddings'
        this.name = 'googlevertexaiMultimodalEmbeddings'
        this.version = 1.0
        this.type = 'GoogleVertexAIMultimodalEmbeddings'
        this.icon = 'GoogleVertex.svg'
        this.category = 'Embeddings'
        this.description = 'Google Vertex AI Multimodal Embedding API (multimodalembedding@001) - generates 1408-dimensional embeddings for text queries to match image/video embeddings'
        this.baseClasses = [this.type, ...getBaseClasses(GoogleVertexAIMultimodalEmbeddings)]
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            credentialNames: ['googleVertexAuth'],
            optional: true,
            description:
                'Google Vertex AI credential. If you are using a GCP service like Cloud Run, or if you have installed default credentials on your local machine, you do not need to set this credential.'
        }
        this.inputs = [
            {
                label: 'Model Name',
                name: 'modelName',
                type: 'string',
                default: 'multimodalembedding@001',
                description: 'The multimodal embedding model to use'
            },
            {
                label: 'Region',
                description: 'Region to use for the model (e.g., us-central1, us-east1)',
                name: 'region',
                type: 'asyncOptions',
                loadMethod: 'listRegions',
                optional: true
            },
            {
                label: 'Strip New Lines',
                name: 'stripNewLines',
                type: 'boolean',
                optional: true,
                additionalParams: true,
                description: 'Remove new lines from input text before embedding to reduce token count'
            }
        ]
    }

    //@ts-ignore
    loadMethods = {
        async listRegions(): Promise<INodeOptionsValue[]> {
            return [
                { label: 'us-central1', name: 'us-central1' },
                { label: 'us-east1', name: 'us-east1' },
                { label: 'us-east4', name: 'us-east4' },
                { label: 'us-west1', name: 'us-west1' },
                { label: 'us-west4', name: 'us-west4' },
                { label: 'europe-west1', name: 'europe-west1' },
                { label: 'europe-west2', name: 'europe-west2' },
                { label: 'europe-west3', name: 'europe-west3' },
                { label: 'europe-west4', name: 'europe-west4' },
                { label: 'asia-east1', name: 'asia-east1' },
                { label: 'asia-northeast1', name: 'asia-northeast1' },
                { label: 'asia-southeast1', name: 'asia-southeast1' }
            ]
        }
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const modelName = nodeData.inputs?.modelName as string
        const region = nodeData.inputs?.region as string
        const stripNewLines = nodeData.inputs?.stripNewLines as boolean

        const obj: GoogleVertexAIMultimodalEmbeddingsParams = {
            modelName: modelName || 'multimodalembedding@001',
            stripNewLines
        }

        const authOptions = await buildGoogleCredentials(nodeData, options)
        if (authOptions && Object.keys(authOptions).length !== 0) obj.authOptions = authOptions as GoogleAuthOptions

        if (region) obj.location = region

        const model = new GoogleVertexAIMultimodalEmbeddings(obj)
        return model
    }
}

module.exports = { nodeClass: GoogleVertexAIMultimodalEmbedding_Embeddings }

