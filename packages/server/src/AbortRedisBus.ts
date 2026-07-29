import { createClient } from 'redis'
import logger from './utils/logger'

export const ABORT_REDIS_CHANNEL = 'flowise:chatmessage:abort'

function isRedisConfigured(): boolean {
    return !!(process.env.REDIS_URL || process.env.REDIS_HOST)
}

function createRedisClient() {
    if (process.env.REDIS_URL) {
        return createClient({
            url: process.env.REDIS_URL,
            socket: {
                keepAlive:
                    process.env.REDIS_KEEP_ALIVE && !isNaN(parseInt(process.env.REDIS_KEEP_ALIVE, 10))
                        ? parseInt(process.env.REDIS_KEEP_ALIVE, 10)
                        : undefined
            },
            pingInterval:
                process.env.REDIS_KEEP_ALIVE && !isNaN(parseInt(process.env.REDIS_KEEP_ALIVE, 10))
                    ? parseInt(process.env.REDIS_KEEP_ALIVE, 10)
                    : undefined
        })
    }

    return createClient({
        username: process.env.REDIS_USERNAME || undefined,
        password: process.env.REDIS_PASSWORD || undefined,
        socket: {
            host: process.env.REDIS_HOST || 'localhost',
            port: parseInt(process.env.REDIS_PORT || '6379'),
            tls: process.env.REDIS_TLS === 'true',
            cert: process.env.REDIS_CERT ? Buffer.from(process.env.REDIS_CERT, 'base64') : undefined,
            key: process.env.REDIS_KEY ? Buffer.from(process.env.REDIS_KEY, 'base64') : undefined,
            ca: process.env.REDIS_CA ? Buffer.from(process.env.REDIS_CA, 'base64') : undefined,
            keepAlive:
                process.env.REDIS_KEEP_ALIVE && !isNaN(parseInt(process.env.REDIS_KEEP_ALIVE, 10))
                    ? parseInt(process.env.REDIS_KEEP_ALIVE, 10)
                    : undefined
        },
        pingInterval:
            process.env.REDIS_KEEP_ALIVE && !isNaN(parseInt(process.env.REDIS_KEEP_ALIVE, 10))
                ? parseInt(process.env.REDIS_KEEP_ALIVE, 10)
                : undefined
    })
}

/**
 * Cross-instance abort fan-out for non-QUEUE (or multi-web) deployments.
 * Prediction SSE + AbortController live in-memory per process; without this,
 * PUT /abort can hit a different Render instance and silently no-op.
 */
export class AbortRedisBus {
    private publisher: ReturnType<typeof createClient> | null = null
    private subscriber: ReturnType<typeof createClient> | null = null
    private enabled = false

    isEnabled(): boolean {
        return this.enabled
    }

    async connect(onAbort: (payload: { id?: string; chatId?: string }) => void): Promise<void> {
        if (!isRedisConfigured()) {
            logger.info('[AbortRedisBus] Redis not configured; cross-instance abort disabled')
            return
        }

        try {
            this.publisher = createRedisClient()
            this.subscriber = createRedisClient()

            this.publisher.on('error', (err: any) => logger.error('[AbortRedisBus] publisher error:', err))
            this.subscriber.on('error', (err: any) => logger.error('[AbortRedisBus] subscriber error:', err))

            await this.publisher.connect()
            await this.subscriber.connect()

            await this.subscriber.subscribe(ABORT_REDIS_CHANNEL, (message: string) => {
                try {
                    const parsed = JSON.parse(message) as { id?: string; chatId?: string }
                    if (!parsed?.id && !parsed?.chatId) return
                    logger.info(`[AbortRedisBus] received abort id=${parsed.id ?? ''} chatId=${parsed.chatId ?? ''}`)
                    onAbort(parsed)
                } catch (e) {
                    logger.error('[AbortRedisBus] failed to handle abort message:', e)
                }
            })

            this.enabled = true
            logger.info(`[AbortRedisBus] subscribed to ${ABORT_REDIS_CHANNEL}`)
        } catch (e) {
            this.enabled = false
            logger.error('[AbortRedisBus] failed to connect; cross-instance abort disabled:', e)
            await this.disconnect().catch(() => undefined)
        }
    }

    async publish(payload: { id?: string; chatId?: string }): Promise<boolean> {
        if (!this.enabled || !this.publisher) return false
        if (!payload.id && !payload.chatId) return false
        try {
            await this.publisher.publish(ABORT_REDIS_CHANNEL, JSON.stringify(payload))
            logger.info(`[AbortRedisBus] published abort id=${payload.id ?? ''} chatId=${payload.chatId ?? ''}`)
            return true
        } catch (e) {
            logger.error(`[AbortRedisBus] publish failed:`, e)
            return false
        }
    }

    async disconnect(): Promise<void> {
        this.enabled = false
        if (this.subscriber) {
            try {
                await this.subscriber.unsubscribe(ABORT_REDIS_CHANNEL)
                await this.subscriber.quit()
            } catch {
                /* ignore */
            }
            this.subscriber = null
        }
        if (this.publisher) {
            try {
                await this.publisher.quit()
            } catch {
                /* ignore */
            }
            this.publisher = null
        }
    }
}
