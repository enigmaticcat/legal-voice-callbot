/**
 * CallButton Component
 * Nút gọi tư vấn — to, rõ ràng, có animation.
 * Bước 7 sẽ kết nối với WebSocket.
 */

function CallButton({ isActive, onToggle, disabled }) {
  return (
    <button
      className={`call-button ${isActive ? 'active' : ''}`}
      onClick={onToggle}
      disabled={disabled}
    >
      {isActive ? '📞 Kết thúc cuộc gọi' : '📞 Gọi Tư Vấn'}
    </button>
  )
}

export default CallButton
