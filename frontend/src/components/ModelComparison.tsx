import { Check, FlaskConical } from 'lucide-react';
import { METHODS, type AnalyzeResponse } from '../types/sentiment';
import { MODEL_NAMES, formatPercent } from '../utils/formatters';
import { LabelBadge } from './SentimentResult';
import { ProbabilityBars } from './ProbabilityBars';

export function ModelComparison({ response }: { response: AnalyzeResponse }) {
  return <section aria-label="So sánh kết quả các mô hình"><div className="section-heading"><div><span className="eyebrow"><FlaskConical size={14} />CÙNG VĂN BẢN · BA GÓC NHÌN</span><h2>So sánh kết quả các mô hình</h2></div></div>
    <div className="model-grid">{METHODS.map((method, index) => {
      const p = response.comparison[method], selected = method === response.selection_info.selected_method;
      return <article className={`card model-card ${selected ? 'selected' : ''}`} key={method}><div className="model-card-top"><span className="model-number">MÔ HÌNH 0{index + 1}</span>{selected && <span className="selected-pill"><Check size={12} />Được lựa chọn</span>}</div>
        <h3>{MODEL_NAMES[method]}</h3><p className="model-description">{method === 'svm' ? 'Mô hình cơ sở với đặc trưng TF-IDF' : method === 'transformer' ? 'Mô hình Transformer được huấn luyện từ đầu' : 'Kết hợp xác suất bằng Weighted Soft Voting'}</p>
        <div className="model-prediction"><LabelBadge label={p.label} /><strong>{formatPercent(p.confidence)}<small>độ tin cậy</small></strong></div>
        <ProbabilityBars probabilities={p.probabilities} compact />
        {method === 'ensemble' && <div className="model-weights"><span>SVM <b>{formatPercent(response.research_info.ensemble_weights.svm)}</b></span><span>Transformer <b>{formatPercent(response.research_info.ensemble_weights.transformer)}</b></span></div>}
      </article>;
    })}</div>
  </section>;
}
