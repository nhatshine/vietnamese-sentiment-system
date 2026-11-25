import { Activity, BarChart3, CircleCheck, CircleMinus, Info, LoaderCircle, Minus } from 'lucide-react';
import type { AnalyzeResponse, SentimentLabel } from '../types/sentiment';
import { LABEL_NAMES, MODEL_NAMES, formatPercent } from '../utils/formatters';
import { ProbabilityBars } from './ProbabilityBars';

export function LabelBadge({ label }: { label: SentimentLabel }) { return <span className={`label-badge ${label}`}><span />{LABEL_NAMES[label]}</span>; }
export function SentimentResult({ response, loading }: { response: AnalyzeResponse | null; loading: boolean }) {
  const p = response?.final_result;
  const Icon = p?.label === 'positive' ? CircleCheck : p?.label === 'negative' ? CircleMinus : Minus;
  return <section className={`card result-card ${p ? 'has-result' : ''}`} aria-label="Kết quả phân tích cảm xúc" aria-busy={loading}>
    <div className="card-heading"><span className="step-number">02</span><div><h2>Kết quả phân tích cảm xúc</h2><p>{p ? (response.sentence_count > 1 ? 'Góc nhìn tổng thể của toàn đoạn văn.' : 'Sắc thái từ văn bản của bạn.') : 'Một kết quả chính, rõ ràng và trực quan.'}</p></div></div>
    {!p ? <div className="empty-result" aria-live="polite"><div className={`empty-icon ${loading ? 'loading' : ''}`}>{loading ? <LoaderCircle size={33} className="spin" /> : <BarChart3 size={33} />}</div>
      <h3>{loading ? 'Đang đọc những sắc thái...' : 'Sẵn sàng lắng nghe câu chữ'}</h3><p>{loading ? 'AI đang phân tích cảm xúc văn bản...' : 'Nhập câu hoặc đoạn văn để bắt đầu phân tích cảm xúc.'}</p>
      {!loading && <span className="empty-caption"><Activity size={13} />Kết quả trực tiếp từ mô hình AI</span>}</div> : <>
      <div className={`result-summary ${p.label}`}><div className="sentiment-symbol"><Icon size={28} /></div><div><span className="overline">{response.sentence_count > 1 ? 'CẢM XÚC TOÀN ĐOẠN' : 'CẢM XÚC CHÍNH'}</span><h3>{LABEL_NAMES[p.label]}</h3></div><div className="confidence" role="group" aria-label="Độ tin cậy cuối cùng"><span>Độ tin cậy</span><strong>{formatPercent(p.confidence)}</strong></div></div>
      <div className="distribution-heading"><h4>Phân phối xác suất</h4><span>3 nhãn cảm xúc</span></div>
      <ProbabilityBars probabilities={p.probabilities} />
      <p className="result-context">{response.sentence_count > 1 ? `Kết quả toàn đoạn được tổng hợp từ ${response.sentence_count} câu bằng ${response.aggregation_info.method === 'mean_probability' ? 'Mean Probability' : response.aggregation_info.method}.` : `Văn bản có xu hướng ${LABEL_NAMES[p.label].toLocaleLowerCase('vi-VN')} theo kết quả mô hình.`}</p>
      <div className="selected-method"><span><CircleCheck size={13} />Phương án được chọn</span><strong>{MODEL_NAMES[response.selection_info.selected_method]}</strong></div>
      <p className="confidence-note"><Info size={13} />Độ tin cậy là xác suất ước lượng, không bảo đảm dự đoán luôn chính xác.</p>
    </>}
  </section>;
}
