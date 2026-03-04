/**
 * Gateway API Service
 * Gọi REST endpoints cho health check, session info, etc.
 */

const GATEWAY_URL = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000'

export async function checkHealth() {
    const res = await fetch(`${GATEWAY_URL}/health`)
    return res.json()
}

export async function getStatus() {
    const res = await fetch(`${GATEWAY_URL}/`)
    return res.json()
}

export function getWebSocketUrl() {
    const wsUrl = GATEWAY_URL.replace('http', 'ws')
    return `${wsUrl}/ws/voice`
}
