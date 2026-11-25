import { ChevronDown, Layers } from 'lucide-react';
import type { AnalyzeResponse } from '../types/sentiment';
import { formatPercent } from '../utils/formatters';
import { LabelBadge } from './SentimentResult';
import { ProbabilityBars } from './ProbabilityBars';

export function SentenceAnalysis({ response }: { response: AnalyzeResponse }) {
  if (response.sentence_count < 2) return null;
  return <section className="sentence-section" aria-label="Phân tích cảm xúc theo từng câu"><div className="section-heading"><div><span className="eyebrow"><Layers size={14} />ĐI SÂU VÀO VĂN BẢN</span><h2>Phân tích cảm xúc theo từng câu</h2></div><span className="count-pill">{response.sentence_count} câu</span></div>
    <div className="sentence-list">{response.sentences.map(sentence => <article className="card sentence-card" key={sentence.sentence_index}>
      <span className="sentence-index">{String(sentence.sentence_index + 1).padStart(2, '0')}</span><div className="sentence-body"><p data-testid="sentence-text">{sentence.sentence}</p><details><summary>Chi tiết xác suất<ChevronDown size={13} /></summary><ProbabilityBars probabilities={sentence.final_result.probabilities} compact /></details></div>
      <div className="sentence-outcome"><LabelBadge label={sentence.final_result.label} /><span>{formatPercent(sentence.final_result.confidence)} <small>độ tin cậy</small></span></div>
    </article>)}</div>
  </section>;
}
