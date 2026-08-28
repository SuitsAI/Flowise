import { Request, Response, NextFunction } from 'express'
import fs from 'fs'
import path from 'path'
import contentDisposition from 'content-disposition'
import { streamStorageFile } from 'flowise-components'
import { StatusCodes } from 'http-status-codes'
import { InternalFlowiseError } from '../../errors/internalFlowiseError'
import { getRunningExpressApp } from '../../utils/getRunningExpressApp'
import { ChatFlow } from '../../database/entities/ChatFlow'
import { Workspace } from '../../enterprise/database/entities/workspace.entity'

const mimeFromFileName = (fileName: string): string | undefined => {
    const ext = path.extname(fileName).toLowerCase()
    if (ext === '.mp4') return 'video/mp4'
    if (ext === '.webm') return 'video/webm'
    if (ext === '.png') return 'image/png'
    if (ext === '.jpg' || ext === '.jpeg') return 'image/jpeg'
    if (ext === '.gif') return 'image/gif'
    if (ext === '.webp') return 'image/webp'
    return undefined
}

const streamUploadedFile = async (req: Request, res: Response, next: NextFunction) => {
    try {
        if (!req.query.chatflowId || !req.query.chatId || !req.query.fileName) {
            return res.status(500).send(`Invalid file path`)
        }
        const chatflowId = req.query.chatflowId as string
        const chatId = req.query.chatId as string
        const fileName = req.query.fileName as string
        const download = req.query.download === 'true' // Check if download parameter is set

        const appServer = getRunningExpressApp()

        // This can be public API, so we can only get orgId from the chatflow
        const chatflow = await appServer.AppDataSource.getRepository(ChatFlow).findOneBy({
            id: chatflowId
        })
        if (!chatflow) {
            throw new InternalFlowiseError(StatusCodes.NOT_FOUND, `Chatflow ${chatflowId} not found`)
        }
        const chatflowWorkspaceId = chatflow.workspaceId
        const workspace = await appServer.AppDataSource.getRepository(Workspace).findOneBy({
            id: chatflowWorkspaceId
        })
        if (!workspace) {
            throw new InternalFlowiseError(StatusCodes.NOT_FOUND, `Workspace ${chatflowWorkspaceId} not found`)
        }
        const orgId = workspace.organizationId as string

        const mimeType = mimeFromFileName(fileName)
        if (download) {
            res.setHeader('Content-Disposition', contentDisposition(fileName, { type: 'attachment' }))
        } else if (mimeType?.startsWith('video/')) {
            res.setHeader('Content-Disposition', contentDisposition(fileName, { type: 'inline' }))
        } else {
            res.setHeader('Content-Disposition', contentDisposition(fileName))
        }
        if (mimeType) {
            res.setHeader('Content-Type', mimeType)
        }
        const fileStream = await streamStorageFile(chatflowId, chatId, fileName, orgId)

        if (!fileStream) throw new InternalFlowiseError(StatusCodes.INTERNAL_SERVER_ERROR, `Error: streamStorageFile`)

        if (fileStream instanceof fs.ReadStream && fileStream?.pipe) {
            fileStream.pipe(res)
        } else {
            res.send(fileStream)
        }
    } catch (error) {
        next(error)
    }
}

export default {
    streamUploadedFile
}
