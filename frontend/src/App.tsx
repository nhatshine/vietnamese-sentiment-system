import { useEffect, useRef, useState } from 'react';
import { FlaskConical, ShieldCheck } from 'lucide-react';
import { Header, type Connection } from './components/Header';
import { TextInput } from './components/TextInput';
import { SentimentResult } from './components/SentimentResult';
import { SentenceAnalysis } from './components/SentenceAnalysis';
import { ModelComparison } from './components/ModelComparison';
import { ModelMetrics } from './components/ModelMetrics';
import { EnsembleExplanation } from './components/EnsembleExplanation';
import { DocumentAggregation } from './components/DocumentAggregation';
import { ErrorMessage } from './components/ErrorMessage';
import { useAnalysis } from './hooks/useAnalysis';
import { checkHealth, getModelInfo } from './services/api';
import type { ModelInfoResponse, ViewMode } from './types/sentiment';
import './App.css';

export default function App() {
  const [viewMode, setViewMode] = useState<ViewMode>('user');
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [connection, setConnection] = useState<Connection>('checking');
  const [metadataError, setMetadataError] = useState(false);
  const provenReady = useRef(false);
  const analysis = useAnalysis();
  useEffect(() => {
    const controller = new AbortController();
    void checkHealth(controller.signal).then(health => { if (!controller.signal.aborted && !provenReady.current) setConnection(health.model_ready ? 'ready' : 'not-ready'); }).catch(() => { if (!controller.signal.aborted && !provenReady.current) setConnection('offline'); });
    void getModelInfo(controller.signal).then(info => { if (!controller.signal.aborted) setModelInfo(info); }).catch(() => { if (!controller.signal.aborted) setMetadataError(true); });
    return () => controller.abort();
  }, []);
  const response = analysis.canonicalResponse;
  useEffect(() => { if (response) { provenReady.current = true; setConnection('ready'); } }, [response]);
  return <>
    <a className="skip-link" href="#analysis-panel">Đến khu vực phân tích</a>
    <Header mode={viewMode} setMode={setViewMode} connection={connection} />
    <main className="container main-content" id="analysis-panel">
      <div className="workspace-label"><span>KHÔNG GIAN PHÂN TÍCH</span><span><ShieldCheck size={14} />Kết quả từ mô hình thật</span></div>
      {analysis.error && <ErrorMessage message={analysis.error} />}
      <div className="workspace-grid">
        <TextInput text={analysis.inputText} onChange={analysis.editText} onSubmit={() => void analysis.submit()} loading={analysis.isAnalyzing} onCancel={analysis.cancel} />
        <SentimentResult response={response} loading={analysis.isAnalyzing} />
      </div>
      {response && <SentenceAnalysis response={response} />}
      {viewMode === 'research' && <div className="research-panel" id="research-panel" role="tabpanel" aria-labelledby="tab-research">
        <div className="research-divider"><span><FlaskConical size={17} />GÓC NHÌN NGHIÊN CỨU</span><p>Dữ liệu đã có từ Backend · Không chạy lại mô hình khi đổi chế độ</p></div>
        {response ? <ModelComparison response={response} /> : <div className="research-intro card"><FlaskConical size={25} /><p>Phân tích một văn bản để xem dự đoán riêng của ba phương án. Bạn có thể khám phá các kết quả thực nghiệm bên dưới.</p></div>}
        {metadataError && !response && <p className="metadata-warning" role="status">Chưa tải được metadata nghiên cứu. Kiểm tra FastAPI Backend; giao diện không tạo số liệu thay thế.</p>}
        <div className="research-metrics-grid"><ModelMetrics info={modelInfo} response={response} /><EnsembleExplanation info={modelInfo} response={response} /></div>
        {response && response.sentence_count > 1 && <DocumentAggregation response={response} />}
      </div>}
    </main>
    <footer className="container footer"><span>AI Sentiment Analysis<span className="footer-dot">·</span>Nghiên cứu xử lý ngôn ngữ tự nhiên</span><span>Tiếng Việt · SVM / Transformer / Ensemble</span></footer>
  </>;
}
