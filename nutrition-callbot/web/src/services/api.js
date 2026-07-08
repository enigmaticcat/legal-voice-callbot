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

export function getHttpUrl() {
    return import.meta.env.VITE_GATEWAY_URL || `${window.location.protocol}//${window.location.host}`
}

export function getWebSocketUrl() {
    // Nếu VITE_GATEWAY_URL được set (local dev với Vite riêng) — dùng nó
    if (import.meta.env.VITE_GATEWAY_URL) {
        return import.meta.env.VITE_GATEWAY_URL.replace(/^http/, 'ws') + '/ws/voice'
    }
    // Same-origin: frontend serve từ gateway (Colab qua ngrok / production)
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${proto}//${window.location.host}/ws/voice`
}
