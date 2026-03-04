/**
 * Audio Capture Hook — Microphone via AudioWorklet
 * Thu PCM 16kHz mono từ micro.
 * Bước 7 sẽ implement AudioWorkletProcessor.
 */
import { useState, useCallback } from 'react'

export function useAudioCapture() {
    const [isRecording, setIsRecording] = useState(false)

    const startCapture = useCallback(async (onAudioData) => {
        // TODO: Implement AudioWorklet ở Bước 7
        console.log('Audio capture started (placeholder)')
        setIsRecording(true)
    }, [])

    const stopCapture = useCallback(() => {
        console.log('Audio capture stopped')
        setIsRecording(false)
    }, [])

    return { isRecording, startCapture, stopCapture }
}
