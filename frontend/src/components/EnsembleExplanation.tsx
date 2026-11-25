import { Layers, Lock } from 'lucide-react';
import type { AnalyzeResponse, ModelInfoResponse } from '../types/sentiment';
import { formatPercent } from '../utils/formatters';

export function EnsembleExplanation({ response, info }: { response: AnalyzeResponse | null; info: ModelInfoResponse | null }) {
  const alpha = response?.selection_info.selected_alpha ?? info?.selected_alpha;
  const weights = response?.research_info.ensemble_weights ?? (alpha === undefined ? null : { svm: alpha, transformer: 1 - alpha });
  return <section className="card ensemble-card" aria-label="Phương pháp kết hợp mô hình"><div className="card-heading"><span className="heading-icon"><Layers size={20} /></span><div><h2>Phương pháp kết hợp mô hình</h2><p>Weighted Soft Voting</p></div></div>
    <div className="formula"><span>P<sub>Ensemble</sub>(c) =</span><span>α × P<sub>SVM</sub>(c)</span><span>+ (1 − α) × P<sub>Transformer</sub>(c)</span></div>
    {weights && <><div className="weight-bar" aria-label="Trọng số từ Backend"><span style={{ width: `${weights.svm * 100}%` }} /><span style={{ width: `${weights.transformer * 100}%` }} /></div><div className="weight-labels"><span><i />SVM<strong>{formatPercent(weights.svm)}</strong></span><span><i />Transformer<strong>{formatPercent(weights.transformer)}</strong></span></div></>}
    <p className="ensemble-copy">Weighted Soft Voting kết hợp xác suất dự đoán của các mô hình theo trọng số được lựa chọn dựa trên Macro-F1 của tập Validation.</p>
    <div className="locked-note"><Lock size={14} /><span>Trọng số đã khóa theo thực nghiệm{alpha !== undefined ? ` · α = ${alpha}` : ''}</span></div>
    {info && <p className="small-muted">{info.models.transformer.name}</p>}
    <p className="small-muted">Kết quả thực nghiệm trên tập dữ liệu khác với confidence của một văn bản cụ thể.</p>
  </section>;
}
