/**
 * WebSocket hook.
 * onBinaryMessage(ArrayBuffer) — audio PCM chunks từ server
 * onJsonMessage(object)        — status/transcript/bot_response events từ server
 */
import { useState, useCallback, useRef } from 'react'

export function useWebSocket(onBinaryMessage, onJsonMessage) {
    const [isConnected, setIsConnected] = useState(false)
    const wsRef = useRef(null)

    // Keep callbacks fresh via refs — tránh stale closure khi re-render
    const onBinaryRef = useRef(onBinaryMessage)
    const onJsonRef = useRef(onJsonMessage)
    onBinaryRef.current = onBinaryMessage
    onJsonRef.current = onJsonMessage

    const connect = useCallback((url) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return

        const ws = new WebSocket(url)
        ws.binaryType = 'arraybuffer'

        ws.onopen = () => setIsConnected(true)
        ws.onclose = () => { setIsConnected(false); wsRef.current = null }
        ws.onerror = (e) => console.error('[WS] error:', e)
        ws.onmessage = (e) => {
            if (e.data instanceof ArrayBuffer) {
                onBinaryRef.current?.(e.data)
            } else {
                try { onJsonRef.current?.(JSON.parse(e.data)) } catch {}
            }
        }

        wsRef.current = ws
    }, [])

    const disconnect = useCallback(() => {
        wsRef.current?.close()
        wsRef.current = null
        setIsConnected(false)
    }, [])

    const send = useCallback((data) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(data)
    }, [])

    return { isConnected, connect, disconnect, send }
}
