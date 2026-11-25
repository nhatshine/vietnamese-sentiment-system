import { ArrowRight, LoaderCircle, Lock, Sparkles, Trash2, X } from 'lucide-react';
import { characterCount } from '../utils/formatters';
import { MAX_TEXT_CHARS } from '../hooks/useAnalysis';

const EXAMPLES = [
  { name: 'Tích cực', text: 'Sản phẩm này rất tốt, tôi rất hài lòng.' },
  { name: 'Tiêu cực', text: 'Tôi thất vọng vì chất lượng quá kém.' },
  { name: 'Trung tính', text: 'Sản phẩm được giao vào sáng hôm qua.' },
  { name: 'Hỗn hợp', text: 'Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!' },
];
interface Props { text: string; onChange: (text: string) => void; onSubmit: () => void; loading: boolean; onCancel: () => void }
export function TextInput({ text, onChange, onSubmit, loading, onCancel }: Props) {
  const count = characterCount(text), over = count > MAX_TEXT_CHARS;
  return <form className="card input-card" aria-label="Phân tích văn bản" onSubmit={event => { event.preventDefault(); onSubmit(); }}>
    <div className="card-heading"><span className="step-number">01</span><div><h2><label htmlFor="sentiment-text">Nhập văn bản cần phân tích</label></h2><p>Một câu ngắn hay cả đoạn văn — bắt đầu từ đây.</p></div></div>
    <div className={`textarea-wrap ${over ? 'invalid' : ''}`}><textarea id="sentiment-text" value={text} onChange={event => onChange(event.target.value)}
      aria-describedby="text-counter input-hint" aria-invalid={over} placeholder={'Nhập câu hoặc đoạn văn tiếng Việt tại đây...\n\nVí dụ: Camera rất đẹp. Nhưng pin quá yếu.\nGiao hàng nhanh và đóng gói cẩn thận!'}
      onKeyDown={event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); onSubmit(); } }} />
      <div className="textarea-bottom"><span className="text-format">Aa <span>Văn bản tiếng Việt</span></span><span id="text-counter" className={over ? 'over-limit' : ''}>{count.toLocaleString('vi-VN')} / {MAX_TEXT_CHARS.toLocaleString('vi-VN')} ký tự</span></div>
    </div>
    <p className={over ? 'input-hint over-limit' : 'input-hint'} id="input-hint">{over ? 'Vượt giới hạn ký tự. Văn bản không bị cắt bớt; vui lòng rút ngắn trước khi gửi.' : 'Hỗ trợ nhiều câu, xuống dòng và dấu tiếng Việt.'}</p>
    <div className="examples"><span>Thử một ví dụ</span><div>{EXAMPLES.map(example => <button type="button" key={example.name} onClick={() => onChange(example.text)} aria-label={`Ví dụ ${example.name.toLocaleLowerCase('vi-VN')}`}>{example.name}</button>)}</div></div>
    <div className="form-actions"><button className="button-clear" type="button" disabled={!text && !loading} onClick={() => onChange('')} aria-label="Xóa văn bản"><Trash2 size={16} />Xóa</button>
      {loading && <button type="button" className="button-cancel" onClick={onCancel} aria-label="Hủy phân tích"><X size={15} />Hủy</button>}
      <button type="submit" className="button-primary" disabled={loading || !text.trim() || over}>{loading ? <><LoaderCircle className="spin" size={18} />Đang phân tích...</> : <><Sparkles size={17} />Phân tích cảm xúc<ArrowRight size={17} /></>}</button>
    </div>
    <div className="privacy-note"><Lock size={12} /><span>Xử lý qua FastAPI · Không lưu lịch sử phân tích</span></div>
  </form>;
}
