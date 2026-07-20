import logger from '../utils/logger'
import { QueueManager } from '../queue/QueueManager'
import { BaseCommand } from './base'
import { getDataSource } from '../DataSource'
import { Telemetry } from '../utils/telemetry'
import { NodesPool } from '../NodesPool'
import { CachePool } from '../CachePool'
import { QueueEvents, QueueEventsListener } from 'bullmq'
import { AbortControllerPool } from '../AbortControllerPool'
import { UsageCacheManager } from '../UsageCacheManager'
import { initializeLangfuseTracing, flushLangfuseTracing } from 'flowise-components'

interface CustomListener extends QueueEventsListener {
    abort: (args: { id: string }, id: string) => void
    abortByChatId: (args: { chatId: string }, id: string) => void
}

export default class Worker extends BaseCommand {
    predictionWorkerId: string
    upsertionWorkerId: string

    async run(): Promise<void> {
        logger.info('Starting Flowise Worker...')

        // Chatflow-level Langfuse credentials can't be known before the first job, but
        // env-configured deployments shouldn't have to wait on that first job to start tracing.
        if (process.env.LANGFUSE_PUBLIC_KEY && process.env.LANGFUSE_SECRET_KEY) {
            initializeLangfuseTracing({
                publicKey: process.env.LANGFUSE_PUBLIC_KEY,
                secretKey: process.env.LANGFUSE_SECRET_KEY,
                baseUrl: process.env.LANGFUSE_BASE_URL,
                release: process.env.LANGFUSE_RELEASE
            })
        }

        const { appDataSource, telemetry, componentNodes, cachePool, abortControllerPool, usageCacheManager } = await this.prepareData()

        const queueManager = QueueManager.getInstance()
        queueManager.setupAllQueues({
            componentNodes,
            telemetry,
            cachePool,
            appDataSource,
            abortControllerPool,
            usageCacheManager
        })

        /** Prediction */
        const predictionQueue = queueManager.getQueue('prediction')
        const predictionWorker = predictionQueue.createWorker()
        this.predictionWorkerId = predictionWorker.id
        logger.info(`Prediction Worker ${this.predictionWorkerId} created`)

        const predictionQueueName = predictionQueue.getQueueName()
        const queueEvents = new QueueEvents(predictionQueueName, { connection: queueManager.getConnection() })

        queueEvents.on<CustomListener>('abort', async ({ id }: { id: string }) => {
            const found = abortControllerPool.abort(id)
            logger.info(`[worker] abort event id=${id} found=${found} poolKeys=${JSON.stringify(abortControllerPool.keys())}`)
        })

        queueEvents.on<CustomListener>('abortByChatId', async ({ chatId }: { chatId: string }) => {
            const abortedIds = abortControllerPool.abortByChatId(chatId)
            logger.info(
                `[worker] abortByChatId chatId=${chatId} aborted=${JSON.stringify(abortedIds)} poolKeys=${JSON.stringify(
                    abortControllerPool.keys()
                )}`
            )
        })

        /** Upsertion */
        const upsertionQueue = queueManager.getQueue('upsert')
        const upsertionWorker = upsertionQueue.createWorker()
        this.upsertionWorkerId = upsertionWorker.id
        logger.info(`Upsertion Worker ${this.upsertionWorkerId} created`)

        // Keep the process running
        process.stdin.resume()
    }

    async prepareData() {
        // Init database
        const appDataSource = getDataSource()
        await appDataSource.initialize()
        await appDataSource.runMigrations({ transaction: 'each' })

        // Initialize abortcontroller pool
        const abortControllerPool = new AbortControllerPool()

        // Init telemetry
        const telemetry = new Telemetry()

        // Initialize nodes pool
        const nodesPool = new NodesPool()
        await nodesPool.initialize()

        // Initialize cache pool
        const cachePool = new CachePool()

        // Initialize usage cache manager
        const usageCacheManager = await UsageCacheManager.getInstance()

        return { appDataSource, telemetry, componentNodes: nodesPool.componentNodes, cachePool, abortControllerPool, usageCacheManager }
    }

    async catch(error: Error) {
        if (error.stack) logger.error(error.stack)
        await new Promise((resolve) => {
            setTimeout(resolve, 1000)
        })
        await this.failExit()
    }

    async stopProcess() {
        try {
            const queueManager = QueueManager.getInstance()
            const predictionWorker = queueManager.getQueue('prediction').getWorker()
            logger.info(`Shutting down Flowise Prediction Worker ${this.predictionWorkerId}...`)
            await predictionWorker.close()

            const upsertWorker = queueManager.getQueue('upsert').getWorker()
            logger.info(`Shutting down Flowise Upsertion Worker ${this.upsertionWorkerId}...`)
            await upsertWorker.close()

            await flushLangfuseTracing()
        } catch (error) {
            logger.error('There was an error shutting down Flowise Worker...', error)
            await this.failExit()
        }

        await this.gracefullyExit()
    }
}
