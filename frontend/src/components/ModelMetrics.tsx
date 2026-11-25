import { BarChart3, Info } from 'lucide-react';
import { METHODS, type AnalyzeResponse, type ModelInfoResponse } from '../types/sentiment';
import { MODEL_NAMES, formatMetric } from '../utils/formatters';

export function ModelMetrics({ info, response }: { info: ModelInfoResponse | null; response: AnalyzeResponse | null }) {
  const validation = response?.research_info.validation_macro_f1 ?? (info ? Object.fromEntries(METHODS.map(m => [m, info.validation_metrics[m].macro_f1])) : null);
  const test = response ? response.research_info.test_macro_f1 : info?.test_metrics ? Object.fromEntries(METHODS.map(m => [m, info.test_metrics![m].macro_f1])) : null;
  if (!validation) return <div className="card metadata-empty"><Info size={19} /><p>Chưa có số liệu nghiên cứu. Kiểm tra kết nối Backend hoặc phân tích một văn bản để nhận metadata.</p></div>;
  return <section className="card metrics-card" aria-label="Hiệu quả mô hình trên tập dữ liệu đánh giá"><div className="card-heading"><span className="heading-icon"><BarChart3 size={20} /></span><div><h2>Hiệu quả mô hình trên tập dữ liệu đánh giá</h2><p>Macro-F1 toàn tập · Giá trị từ 0 đến 1, cao hơn là tốt hơn</p></div></div>
    <div className="chart-legend"><span><i className="validation-color" />Validation</span><span><i className="test-color" />Test</span></div>
    <div className="metrics-plot" role="img" aria-label="Biểu đồ Macro-F1 Validation và Test của ba mô hình; số liệu chính xác ở bảng bên dưới">
      <div className="chart-axis">{['1.0', '0.75', '0.5', '0.25', '0'].map(t => <span key={t}>{t}</span>)}</div>
      <div className="chart-groups">{METHODS.map(method => <div className="chart-group" key={method}><div className="bar-pair">
        <div className="metric-bar validation-color" style={{ height: `${validation[method] * 100}%` }}><span>{formatMetric(validation[method])}</span></div>
        {test && <div className="metric-bar test-color" style={{ height: `${test[method] * 100}%` }}><span>{formatMetric(test[method])}</span></div>}
      </div><strong>{method === 'svm' ? 'SVM' : method === 'transformer' ? 'Transformer' : 'Ensemble'}</strong></div>)}</div>
    </div>
    <div className="table-scroll"><table className="metrics-table" aria-label="Số liệu Macro-F1"><thead><tr><th scope="col">Phương án</th><th scope="col">Validation</th><th scope="col">Test</th></tr></thead><tbody>{METHODS.map(m => <tr key={m}><th scope="row">{MODEL_NAMES[m]}</th><td>{formatMetric(validation[m])}</td><td>{test ? formatMetric(test[m]) : 'Chưa có số liệu'}</td></tr>)}</tbody></table></div>
    <p className="research-note"><Info size={14} />Trọng số Ensemble được lựa chọn trên tập Validation. Tập Test chỉ được sử dụng để đánh giá cuối cùng.</p>
  </section>;
}
