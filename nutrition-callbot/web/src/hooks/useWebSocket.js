/**
 * WebSocket Connection Hook
 * Quản lý kết nối WS tới Gateway.
 * Bước 7 sẽ implement đầy đủ.
 */
import { useState, useCallback, useRef } from 'react'

export function useWebSocket(url) {
    const [isConnected, setIsConnected] = useState(false)
    const wsRef = useRef(null)

    const connect = useCallback(() => {
        const ws = new WebSocket(url)
        ws.onopen = () => setIsConnected(true)
        ws.onclose = () => setIsConnected(false)
        ws.onerror = (e) => console.error('WebSocket error:', e)
        wsRef.current = ws
    }, [url])

    const disconnect = useCallback(() => {
        wsRef.current?.close()
        setIsConnected(false)
    }, [])

    const send = useCallback((data) => {
        wsRef.current?.send(data)
    }, [])

    return { isConnected, connect, disconnect, send }
}
