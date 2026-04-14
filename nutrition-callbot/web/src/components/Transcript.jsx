/**
 * Transcript Component
 * Hiển thị transcript real-time — user speech và bot response streaming.
 * Auto-scroll khi có message mới.
 */
import { useEffect, useRef } from 'react'

function Transcript({ messages }) {
    const bottomRef = useRef(null)

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    if (!messages || messages.length === 0) {
        return (
            <div className="transcript empty">
                <p>Giữ nút mic để nói...</p>
            </div>
        )
    }

    return (
        <div className="transcript">
            {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                    <span className="role-label">{msg.role === 'user' ? 'Bạn' : 'Bot'}</span>
                    <p>{msg.text}</p>
                </div>
            ))}
            <div ref={bottomRef} />
        </div>
    )
}

export default Transcript
