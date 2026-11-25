import { Activity, BrainCircuit, FlaskConical, Sparkles } from 'lucide-react';
import type { ViewMode } from '../types/sentiment';

export type Connection = 'checking' | 'ready' | 'not-ready' | 'offline';
export function Header({ mode, setMode, connection }: { mode: ViewMode; setMode: (mode: ViewMode) => void; connection: Connection }) {
  const status = { checking: 'Đang kết nối', ready: 'Backend sẵn sàng', 'not-ready': 'Mô hình đang khởi động', offline: 'Chưa kết nối Backend' }[connection];
  return <>
    <header className="topbar"><div className="container topbar-inner">
      <a className="wordmark" href="#top" aria-label="AI Sentiment Analysis — đầu trang"><span className="brand-icon"><BrainCircuit size={23} /></span><span>AI SENTIMENT<span className="wordmark-light"> ANALYSIS</span></span></a>
      <span className={`connection ${connection}`} role="status"><span className="status-dot" />{status}<Activity size={13} aria-hidden="true" /></span>
    </div></header>
    <section className="hero container" id="top">
      <div className="hero-copy"><span className="eyebrow"><span className="tiny-line" />ỨNG DỤNG NGHIÊN CỨU NLP</span>
        <h1>Hệ thống phân tích cảm xúc <span>tiếng Việt</span></h1>
        <p>Khám phá sắc thái trong từng câu chữ với SVM, Transformer<br className="desktop-break" /> và phương pháp Weighted Ensemble.</p>
      </div>
      <div className="mode-area"><span className="mode-caption">Góc nhìn của bạn</span>
        <div className="mode-switch" role="tablist" aria-label="Chế độ hiển thị">
          {(['user', 'research'] as const).map((value, i) => <button key={value} type="button" role="tab" id={`tab-${value}`} aria-selected={mode === value}
            aria-controls={value === 'research' ? 'research-panel' : 'analysis-panel'} tabIndex={mode === value ? 0 : -1}
            onClick={() => setMode(value)} onKeyDown={event => {
              if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
                event.preventDefault(); const next = event.key === 'Home' ? 'user' : event.key === 'End' ? 'research' : i === 0 ? 'research' : 'user';
                setMode(next); document.getElementById(`tab-${next}`)?.focus();
              }
            }}>{value === 'user' ? <Sparkles size={16} /> : <FlaskConical size={16} />}{value === 'user' ? 'Người dùng' : 'Nghiên cứu'}</button>)}
        </div>
      </div>
    </section>
  </>;
}
