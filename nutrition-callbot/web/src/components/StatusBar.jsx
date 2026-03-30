/**
 * StatusBar Component
 * Hiển thị trạng thái: "Đang nghe..." / "Đang suy nghĩ..." / "Đang trả lời..."
 */

function StatusBar({ status, isConnected }) {
  const statusMap = {
    idle: { text: 'Sẵn sàng', icon: '', color: '#a0a0b8' },
    listening: { text: 'Đang nghe...', icon: '', color: '#4ade80' },
    thinking: { text: 'Đang suy nghĩ...', icon: '', color: '#fbbf24' },
    speaking: { text: 'Đang trả lời...', icon: '', color: '#6c63ff' },
  }

  const current = statusMap[status] || statusMap.idle

  return (
    <div className="status-bar" style={{ color: current.color }}>
      <span className="status-icon">{current.icon}</span>
      <span className="status-text">{current.text}</span>
      <span className={`connection-dot ${isConnected ? 'connected' : 'disconnected'}`} />
    </div>
  )
}

export default StatusBar
