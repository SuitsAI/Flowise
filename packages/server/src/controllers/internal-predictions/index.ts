import { NextFunction, Request, Response } from 'express'
import { StatusCodes } from 'http-status-codes'
import { InternalFlowiseError } from '../../errors/internalFlowiseError'
import { getErrorMessage } from '../../errors/utils'
import { MODE } from '../../Interface'
import chatflowService from '../../services/chatflows'
import { utilBuildChatflow } from '../../utils/buildChatflow'
import { getRunningExpressApp } from '../../utils/getRunningExpressApp'
import logger from '../../utils/logger'

// Send input message and get prediction result (Internal)
const createInternalPrediction = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const flowId = req.params.id
        const chatId = req.body?.chatId ?? '(new)'
        const streaming = req.body?.streaming === true || req.body?.streaming === 'true'
        logger.info(`[prediction] Internal request flowId=${flowId} chatId=${chatId} streaming=${streaming}`)

        const workspaceId = req.user?.activeWorkspaceId

        const chatflow = await chatflowService.getChatflowById(flowId, workspaceId)
        if (!chatflow) {
            throw new InternalFlowiseError(StatusCodes.NOT_FOUND, `Chatflow ${flowId} not found`)
        }

        if (streaming) {
            logger.info(`[prediction] Starting stream for flowId=${flowId} chatId=${chatId}`)
            createAndStreamInternalPrediction(req, res, next)
            return
        } else {
            logger.info(`[prediction] Running non-stream for flowId=${flowId} chatId=${chatId}`)
            const apiResponse = await utilBuildChatflow(req, true)
            if (apiResponse) return res.json(apiResponse)
        }
    } catch (error) {
        next(error)
    }
}

// Send input message and stream prediction result using SSE (Internal)
const createAndStreamInternalPrediction = async (req: Request, res: Response, next: NextFunction) => {
    const chatId = req.body.chatId
    const sseStreamer = getRunningExpressApp().sseStreamer
    logger.info(`[prediction] SSE client added chatId=${chatId}, building chatflow...`)

    try {
        sseStreamer.addClient(chatId, res)
        res.setHeader('Content-Type', 'text/event-stream')
        res.setHeader('Cache-Control', 'no-cache')
        res.setHeader('Connection', 'keep-alive')
        res.setHeader('X-Accel-Buffering', 'no') //nginx config: https://serverfault.com/a/801629
        res.flushHeaders()

        if (process.env.MODE === MODE.QUEUE) {
            getRunningExpressApp().redisSubscriber.subscribe(chatId)
        }

        const apiResponse = await utilBuildChatflow(req, true)
        logger.info(`[prediction] Stream chatflow finished chatId=${chatId}`)
        sseStreamer.streamMetadataEvent(apiResponse.chatId, apiResponse)
        if (apiResponse?.aborted) {
            sseStreamer.streamAbortEvent(apiResponse.chatId)
        }
    } catch (error) {
        logger.error(`[prediction] Stream error chatId=${chatId}: ${getErrorMessage(error)}`)
        if (chatId) {
            sseStreamer.streamErrorEvent(chatId, getErrorMessage(error))
        }
        next(error)
    } finally {
        sseStreamer.removeClient(chatId)
    }
}
export default {
    createInternalPrediction
}
