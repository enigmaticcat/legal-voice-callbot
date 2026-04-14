import { useState, useCallback, useRef } from 'react'
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

    const audioChunksRef = useRef([])    // PCM chunks tích lũy khi đang giữ mic
    const currentBotTextRef = useRef('') // text bot đang stream về
    const fileInputRef = useRef(null)    // hidden <input type="file">

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
                    setStatus('idle')
                }
                break

            case 'audio_start':
                resetPlayback()
                setStatus('speaking')
                break

            case 'error':
                console.error('[Bot error]', event.code, event.message)
                setStatus('idle')
                break

            default:
                break
        }
    }, [playPcm, resetPlayback])

    const { isConnected, connect, disconnect, send } = useWebSocket(handleBinary, handleJson)

    // ── Bắt đầu / kết thúc cuộc gọi ─────────────────────────────────
    const handleCallToggle = useCallback(async () => {
        if (callActive) {
            stopCapture()
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
    }, [callActive, connect, disconnect, stopCapture])

    // ── Push-to-talk: giữ để nói, thả để gửi ─────────────────────────
    const handleMicStart = useCallback(async (e) => {
        e.preventDefault()
        if (status === 'thinking' || status === 'speaking') return
        audioChunksRef.current = []
        setStatus('listening')
        await startCapture((chunk) => {
            audioChunksRef.current.push(chunk)
        })
    }, [status, startCapture])

    const handleMicEnd = useCallback((e) => {
        e.preventDefault()
        stopCapture()
        const chunks = audioChunksRef.current
        if (chunks.length === 0) { setStatus('idle'); return }

        // Merge tất cả PCM Int16 chunks thành một ArrayBuffer
        const totalLen = chunks.reduce((sum, c) => sum + c.byteLength, 0)
        const merged = new Uint8Array(totalLen)
        let offset = 0
        for (const chunk of chunks) {
            merged.set(new Uint8Array(chunk), offset)
            offset += chunk.byteLength
        }

        send(merged.buffer)
        audioChunksRef.current = []
        setStatus('thinking')
    }, [stopCapture, send])

    // ── Audio file upload ─────────────────────────────────────────────
    const handleFileUpload = useCallback(async (e) => {
        const file = e.target.files?.[0]
        if (!file) return
        e.target.value = ''  // reset để chọn lại cùng file được

        setStatus('thinking')

        // Decode audio file → resample về 16kHz mono qua OfflineAudioContext
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

        // Float32 → PCM Int16
        const channelData = resampled.getChannelData(0)
        const pcm = new Int16Array(channelData.length)
        for (let i = 0; i < channelData.length; i++) {
            pcm[i] = Math.round(Math.max(-1, Math.min(1, channelData[i])) * 32767)
        }

        send(pcm.buffer)
    }, [send])

    const isBusy = status === 'thinking' || status === 'speaking'

    return (
        <div className="app">
            <div className="container">
                <div className="hero">
                    <div className="icon">⚖️</div>
                    <h1>Legal CallBot</h1>
                    <p className="subtitle">Tư Vấn Pháp Luật Việt Nam Bằng Giọng Nói AI</p>
                </div>

                <StatusBar status={status} isConnected={isConnected} />

                {callActive && isConnected && (
                    <>
                        <Transcript messages={messages} />

                        <div className="input-row">
                            <button
                                className={`mic-button ${status === 'listening' ? 'recording' : ''}`}
                                onMouseDown={handleMicStart}
                                onMouseUp={handleMicEnd}
                                onTouchStart={handleMicStart}
                                onTouchEnd={handleMicEnd}
                                disabled={isBusy}
                            >
                                {status === 'listening' ? '🔴 Đang nghe...' : '🎙️ Giữ để nói'}
                            </button>

                            <button
                                className="upload-button"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isBusy}
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
