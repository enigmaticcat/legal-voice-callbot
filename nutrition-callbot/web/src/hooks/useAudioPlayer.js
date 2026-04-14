/**
 * Audio Session Hook
 * startCapture(onChunk) — mic → AudioWorklet → PCM Int16 @ 16kHz, gọi onChunk(ArrayBuffer) mỗi 100ms
 * stopCapture()         — dừng mic
 * playPcm(ArrayBuffer)  — phát PCM Int16 @ 24kHz từ server, gapless
 * resetPlayback()       — xóa schedule phát (khi bắt đầu response mới)
 */
import { useCallback, useRef } from 'react'

// AudioWorklet processor chạy trong audio thread riêng
const PROCESSOR_CODE = `
class MicProc extends AudioWorkletProcessor {
  constructor() { super(); this._buf = []; }
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (!ch) return true;
    for (const s of ch) this._buf.push(Math.max(-1, Math.min(1, s)));
    // Gửi từng chunk 100ms (1600 samples @ 16kHz)
    while (this._buf.length >= 1600) {
      const slice = this._buf.splice(0, 1600);
      const pcm = new Int16Array(slice.map(s => Math.round(s * 32767)));
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}
registerProcessor('mic-proc', MicProc);
`

export function useAudioSession() {
    // ── Capture refs ──────────────────────────────────────────────────
    const captureCtxRef = useRef(null)
    const workletRef = useRef(null)
    const streamRef = useRef(null)

    // ── Playback refs ─────────────────────────────────────────────────
    const playCtxRef = useRef(null)
    const nextPlayTimeRef = useRef(0)

    // ── Mic capture ───────────────────────────────────────────────────
    const startCapture = useCallback(async (onChunk) => {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
            video: false,
        })
        streamRef.current = stream

        const ctx = new AudioContext({ sampleRate: 16000 })
        const blob = new Blob([PROCESSOR_CODE], { type: 'application/javascript' })
        const blobUrl = URL.createObjectURL(blob)
        await ctx.audioWorklet.addModule(blobUrl)
        URL.revokeObjectURL(blobUrl)

        const source = ctx.createMediaStreamSource(stream)
        const worklet = new AudioWorkletNode(ctx, 'mic-proc')
        worklet.port.onmessage = (e) => onChunk(e.data)

        // Route qua gain=0 để giữ graph hoạt động, tránh feedback mic
        const silence = ctx.createGain()
        silence.gain.value = 0
        source.connect(worklet)
        worklet.connect(silence)
        silence.connect(ctx.destination)

        captureCtxRef.current = ctx
        workletRef.current = worklet
    }, [])

    const stopCapture = useCallback(() => {
        workletRef.current?.disconnect()
        streamRef.current?.getTracks().forEach(t => t.stop())
        captureCtxRef.current?.close()
        captureCtxRef.current = null
        workletRef.current = null
        streamRef.current = null
    }, [])

    // ── PCM playback (gapless scheduling) ────────────────────────────
    const playPcm = useCallback((arrayBuffer) => {
        const ctx = playCtxRef.current || new AudioContext({ sampleRate: 24000 })
        playCtxRef.current = ctx

        const pcm = new Int16Array(arrayBuffer)
        const audioBuf = ctx.createBuffer(1, pcm.length, 24000)
        const data = audioBuf.getChannelData(0)
        for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768

        const src = ctx.createBufferSource()
        src.buffer = audioBuf
        src.connect(ctx.destination)

        const now = ctx.currentTime
        const startAt = Math.max(now, nextPlayTimeRef.current)
        src.start(startAt)
        nextPlayTimeRef.current = startAt + audioBuf.duration
    }, [])

    const resetPlayback = useCallback(() => {
        nextPlayTimeRef.current = 0
    }, [])

    return { startCapture, stopCapture, playPcm, resetPlayback }
}
