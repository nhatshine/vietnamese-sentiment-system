import { useEffect, useState } from 'react';
import { Info } from 'lucide-react';
import { METHODS, type AnalyzeResponse, type Method } from '../types/sentiment';
import { formatPercent } from '../utils/formatters';
import { LabelBadge } from './SentimentResult';

const NAMES = { majority_vote: 'Majority Voting', mean_probability: 'Mean Probability', confidence_weighted: 'Confidence-Weighted Aggregation' };
export function DocumentAggregation({ response }: { response: AnalyzeResponse }) {
  const [model, setModel] = useState<Method>(response.selection_info.selected_method);
  useEffect(() => setModel(response.selection_info.selected_method), [response]);
  const data = response.document_results[model];
  return <section className="card aggregation-card" aria-label="So sánh phương pháp tổng hợp cảm xúc toàn đoạn"><div className="section-heading"><div><span className="eyebrow">DOCUMENT-LEVEL ANALYSIS</span><h2>So sánh phương pháp tổng hợp cảm xúc toàn đoạn</h2></div></div>
    <div className="aggregation-tabs" role="tablist" aria-label="Mô hình tổng hợp">{METHODS.map((m, i) => <button type="button" role="tab" key={m} id={`agg-tab-${m}`} aria-selected={model === m} aria-controls="aggregation-table" tabIndex={model === m ? 0 : -1}
      onClick={() => setModel(m)} onKeyDown={e => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); const next = METHODS[(i + (e.key === 'ArrowRight' ? 1 : 2)) % 3]; setModel(next); document.getElementById(`agg-tab-${next}`)?.focus(); } }}>{m === 'svm' ? 'SVM' : m === 'transformer' ? 'Transformer' : 'Ensemble'}</button>)}</div>
    <div className="table-scroll" id="aggregation-table" role="tabpanel" aria-labelledby={`agg-tab-${model}`}><table className="aggregation-table"><thead><tr><th scope="col">Phương pháp</th><th scope="col">Cảm xúc</th><th scope="col">Độ đo</th></tr></thead><tbody>
      {(['majority_vote', 'mean_probability', 'confidence_weighted'] as const).map(key => <tr key={key} className={key === response.aggregation_info.method ? 'default-aggregation' : ''}><th scope="row">{NAMES[key]}{key === response.aggregation_info.method && <span className="tiny-tag">Đang áp dụng</span>}</th><td><LabelBadge label={data[key].label} /></td><td>{formatPercent(data[key].confidence)}{key === 'majority_vote' ? ' phiếu' : ''}</td></tr>)}
    </tbody></table></div>
    <p className="research-note"><Info size={14} />Mean Probability chỉ là quy ước triển khai, chưa được chứng minh tối ưu trên bộ dữ liệu document-level có nhãn. Majority Voting hiển thị tỷ lệ phiếu, không phải xác suất dự đoán của mô hình.</p>
  </section>;
}
