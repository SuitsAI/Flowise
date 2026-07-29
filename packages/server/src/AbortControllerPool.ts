/**
 * This pool is to keep track of abort controllers mapped to chatflowid_chatid
 */
export class AbortControllerPool {
    abortControllers: Record<string, AbortController> = {}

    /**
     * Add to the pool
     * @param {string} id
     * @param {AbortController} abortController
     */
    add(id: string, abortController: AbortController) {
        this.abortControllers[id] = abortController
    }

    /**
     * Remove from the pool
     * @param {string} id
     */
    remove(id: string) {
        if (Object.prototype.hasOwnProperty.call(this.abortControllers, id)) {
            delete this.abortControllers[id]
        }
    }

    /**
     * Get the abort controller
     * @param {string} id
     */
    get(id: string) {
        return this.abortControllers[id]
    }

    /**
     * Whether a controller exists for this id
     */
    has(id: string): boolean {
        return Object.prototype.hasOwnProperty.call(this.abortControllers, id)
    }

    /**
     * Current pool keys (for diagnostics)
     */
    keys(): string[] {
        return Object.keys(this.abortControllers)
    }

    /**
     * Abort
     * @param {string} id
     * @returns true if a controller was found and aborted, false if missing
     */
    abort(id: string): boolean {
        const abortController = this.abortControllers[id]
        if (abortController) {
            abortController.abort()
            this.remove(id)
            return true
        }
        return false
    }

    /**
     * Abort every controller tied to a chatId (parent + nested sub-chatflows).
     * Pool keys are `${chatflowId}_${chatId}`.
     * @returns ids that were aborted
     */
    abortByChatId(chatId: string): string[] {
        if (!chatId) return []
        const suffix = `_${chatId}`
        const matching = this.keys().filter((key) => key === chatId || key.endsWith(suffix))
        const abortedIds: string[] = []
        for (const key of matching) {
            if (this.abort(key)) abortedIds.push(key)
        }
        return abortedIds
    }
}
