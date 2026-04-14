/**
 * StatusBar Component
 * Hiển thị trạng thái pipeline: connecting / idle / listening / thinking / speaking
 */
function StatusBar({ status, isConnected }) {
    const statusMap = {
        connecting: { text: 'Đang kết nối...', color: '#fbbf24' },
        idle:       { text: 'Sẵn sàng',        color: '#a0a0b8' },
        listening:  { text: 'Đang nghe...',     color: '#4ade80' },
        thinking:   { text: 'Đang suy nghĩ...', color: '#fbbf24' },
        speaking:   { text: 'Đang trả lời...',  color: '#6c63ff' },
    }

    const current = statusMap[status] || statusMap.idle

    return (
        <div className="status-bar">
            <span className="status-text" style={{ color: current.color }}>
                {current.text}
            </span>
            <span className={`connection-dot ${isConnected ? 'connected' : 'disconnected'}`} />
        </div>
    )
}

export default StatusBar
