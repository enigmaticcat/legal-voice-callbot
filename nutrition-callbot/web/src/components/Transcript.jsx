/**
 * Transcript Component
 * Hiển thị transcript real-time (câu hỏi + câu trả lời).
 * Bước 7 sẽ connect với WebSocket messages.
 */

function Transcript({ messages }) {
  if (!messages || messages.length === 0) {
    return (
      <div className="transcript empty">
        <p>Bắt đầu cuộc gọi để xem transcript...</p>
      </div>
    )
  }

  return (
    <div className="transcript">
      {messages.map((msg, i) => (
        <div key={i} className={`message ${msg.role}`}>
          <span className="role-icon">{msg.role === 'user' ? '' : ''}</span>
          <p>{msg.text}</p>
        </div>
      ))}
    </div>
  )
}

export default Transcript
