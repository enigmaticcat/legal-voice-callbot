import { useState, useCallback, useRef, useEffect } from 'react'
import CallButton from './components/CallButton'
import StatusBar from './components/StatusBar'
import Transcript from './components/Transcript'
import { useWebSocket } from './hooks/useWebSocket'
import { useAudioSession } from './hooks/useAudioPlayer'
import { getWebSocketUrl } from './services/api'

function App() {
    const [callActive, setCallActive] = useState(false)
    const [status, setStatus] = useState('idle')
    const [messages, setMessages] = useState([])

    const currentBotTextRef = useRef('')
    const fileInputRef = useRef(null)
    const vadStartedRef = useRef(false)

    const { startCapture, stopCapture, playPcm, resetPlayback } = useAudioSession()

    // ── Xử lý sự kiện từ WS ──────────────────────────────────────────
    const handleBinary = useCallback((buf) => {
        playPcm(buf)
    }, [playPcm])

    const handleJson = useCallback((event) => {
        switch (event.type) {
            case 'transcript':
                if (event.text) {
                    setMessages(prev => [...prev, { role: 'user', text: event.text }])
                    setStatus('thinking')
                }
                break

            case 'bot_response':
                if (!event.is_final && event.text) {
                    currentBotTextRef.current += event.text
                    const botText = currentBotTextRef.current
                    setMessages(prev => {
                        const last = prev[prev.length - 1]
                        if (last?.role === 'bot') {
                            return [...prev.slice(0, -1), { role: 'bot', text: botText }]
                        }
                        return [...prev, { role: 'bot', text: botText }]
                    })
                } else if (event.is_final) {
                    currentBotTextRef.current = ''
                    setStatus('listening')
                }
                break

            case 'audio_start':
                resetPlayback()
                setStatus('speaking')
                break

            case 'vad_ready':
                setStatus('listening')
                break

            case 'vad_stopped':
                setStatus('idle')
                break

            case 'error':
                console.error('[Bot error]', event.code, event.message)
                setStatus('listening')
                break

            default:
                break
        }
    }, [playPcm, resetPlayback])

    const { isConnected, connect, disconnect, send } = useWebSocket(handleBinary, handleJson)

    // ── Tự động khởi động VAD khi WebSocket kết nối thành công ───────
    useEffect(() => {
        if (isConnected && callActive && !vadStartedRef.current) {
            vadStartedRef.current = true
            send(JSON.stringify({ type: 'start_vad' }))
            startCapture((chunk) => send(chunk))
        }
        if (!isConnected || !callActive) {
            vadStartedRef.current = false
        }
    }, [isConnected, callActive, send, startCapture])

    // ── Bắt đầu / kết thúc cuộc hội thoại ───────────────────────────
    const handleCallToggle = useCallback(async () => {
        if (callActive) {
            stopCapture()
            send(JSON.stringify({ type: 'stop_vad' }))
            disconnect()
            setCallActive(false)
            setStatus('idle')
            setMessages([])
            currentBotTextRef.current = ''
        } else {
            setStatus('connecting')
            connect(getWebSocketUrl())
            setCallActive(true)
        }
    }, [callActive, connect, disconnect, stopCapture, send])

    // ── Audio file upload ─────────────────────────────────────────────
    const handleFileUpload = useCallback(async (e) => {
        const file = e.target.files?.[0]
        if (!file) return
        e.target.value = ''

        setStatus('thinking')

        const arrayBuf = await file.arrayBuffer()
        const tmpCtx = new AudioContext()
        const decoded = await tmpCtx.decodeAudioData(arrayBuf)
        await tmpCtx.close()

        const TARGET_SR = 16000
        const offlineCtx = new OfflineAudioContext(
            1,
            Math.ceil(decoded.duration * TARGET_SR),
            TARGET_SR,
        )
        const src = offlineCtx.createBufferSource()
        src.buffer = decoded
        src.connect(offlineCtx.destination)
        src.start(0)
        const resampled = await offlineCtx.startRendering()

        const channelData = resampled.getChannelData(0)
        const pcm = new Int16Array(channelData.length)
        for (let i = 0; i < channelData.length; i++) {
            pcm[i] = Math.round(Math.max(-1, Math.min(1, channelData[i])) * 32767)
        }

        send(pcm.buffer)
    }, [send])

    return (
        <div className="app">
            <div className="container">
                <div className="hero">
                    <div className="icon">🥗</div>
                    <h1>Nutrition CallBot</h1>
                    <p className="subtitle">Tư Vấn Dinh Dưỡng Bằng Giọng Nói AI</p>
                </div>

                <StatusBar status={status} isConnected={isConnected} />

                {callActive && isConnected && (
                    <>
                        <Transcript messages={messages} />

                        <div className="input-row">
                            <button
                                className="upload-button"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={status === 'thinking' || status === 'speaking'}
                                title="Gửi file audio (.wav, .mp3, ...)"
                            >
                                📎
                            </button>

                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="audio/*"
                                style={{ display: 'none' }}
                                onChange={handleFileUpload}
                            />
                        </div>
                    </>
                )}

                <CallButton
                    isActive={callActive}
                    onToggle={handleCallToggle}
                    disabled={false}
                />
            </div>
        </div>
    )
}

export default App
