import { useState, useEffect } from 'react'

function App() {
  const [gatewayStatus, setGatewayStatus] = useState('Đang kiểm tra...')

  useEffect(() => {
    const gatewayUrl = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000'
    fetch(`${gatewayUrl}/health`)
      .then(res => res.json())
      .then(data => setGatewayStatus(`✅ ${data.status} — ${data.service}`))
      .catch(() => setGatewayStatus('❌ Không kết nối được Gateway'))
  }, [])

  return (
    <div className="app">
      <div className="container">
        <div className="hero">
          <div className="icon">⚖️</div>
          <h1>Legal CallBot</h1>
          <p className="subtitle">Tư Vấn Pháp Luật Việt Nam Bằng Giọng Nói AI</p>
        </div>

        <div className="status-card">
          <h2>🔗 Trạng Thái Hệ Thống</h2>
          <div className="status-item">
            <span className="label">Gateway:</span>
            <span className="value">{gatewayStatus}</span>
          </div>
        </div>

        <div className="info-card">
          <h2>📋 Bước 1 — Hoàn Tất</h2>
          <p>Project skeleton đã sẵn sàng. Giao diện cuộc gọi sẽ được xây dựng ở Bước 7.</p>
          <ul>
            <li>✅ Docker Compose — 5 containers</li>
            <li>✅ Gateway (FastAPI)</li>
            <li>✅ ASR, Brain, TTS (Dummy servers)</li>
            <li>⬜ gRPC Protobufs (Bước 2)</li>
            <li>⬜ Giao diện cuộc gọi (Bước 7)</li>
          </ul>
        </div>

        <button className="call-button" disabled>
          📞 Gọi Tư Vấn (Sắp có ở Bước 7)
        </button>
      </div>
    </div>
  )
}

export default App
