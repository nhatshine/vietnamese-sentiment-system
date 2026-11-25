import { LABELS, type Probabilities } from '../types/sentiment';
import { LABEL_NAMES, formatPercent } from '../utils/formatters';

export function ProbabilityBars({ probabilities, compact = false }: { probabilities: Probabilities; compact?: boolean }) {
  return <div className={`probability-bars ${compact ? 'compact' : ''}`}>{LABELS.map(label => <div className={`probability-row ${label}`} key={label}>
    <div className="probability-label"><span><i />{LABEL_NAMES[label]}</span><strong>{formatPercent(probabilities[label])}</strong></div>
    <div className="bar-track" role="meter" aria-label={LABEL_NAMES[label]} aria-valuemin={0} aria-valuemax={100} aria-valuenow={probabilities[label] * 100} aria-valuetext={formatPercent(probabilities[label])}>
      <span style={{ width: `${probabilities[label] * 100}%` }} />
    </div>
  </div>)}</div>;
}
