import { LABELS, METHODS, type AnalyzeResponse, type ModelInfoResponse, type PredictionResult } from '../types/sentiment';

export class ContractError extends Error { constructor() { super('Dữ liệu máy chủ không đúng hợp đồng.'); this.name = 'ContractError'; } }
function assert(condition: unknown): asserts condition { if (!condition) throw new ContractError(); }
function object(value: unknown): Record<string, unknown> { assert(value !== null && typeof value === 'object' && !Array.isArray(value)); return value as Record<string, unknown>; }
function unit(value: unknown): asserts value is number { assert(typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1); }
function method(value: unknown) { assert(METHODS.includes(value as typeof METHODS[number])); }
function distribution(value: unknown) {
  const data = object(value);
  assert(Object.keys(data).length === LABELS.length);
  let sum = 0;
  for (const label of LABELS) { unit(data[label]); sum += data[label]; }
  // Validate only; never normalize, choose a label, or calculate an ensemble.
  assert(Math.abs(sum - 1) <= 1e-5);
}
function prediction(value: unknown) {
  const p = object(value); method(p.method); assert(LABELS.includes(p.label as typeof LABELS[number]));
  unit(p.confidence); distribution(p.probabilities);
  const probabilities = object(p.probabilities);
  assert(Math.abs(p.confidence - (probabilities[p.label as string] as number)) <= 1e-5);
}
function comparison(value: unknown) {
  const data = object(value);
  for (const name of METHODS) { prediction(data[name]); assert(object(data[name]).method === name); }
}
function samePrediction(left: unknown, right: unknown) {
  const a = object(left), b = object(right);
  assert(a.method === b.method && a.label === b.label && a.confidence === b.confidence);
  for (const label of LABELS) assert(object(a.probabilities)[label] === object(b.probabilities)[label]);
}
function scores(value: unknown) { const data = object(value); for (const name of METHODS) unit(data[name]); }

export function parseAnalyzeResponse(value: unknown): AnalyzeResponse {
  const data = object(value);
  assert(typeof data.input_text === 'string');
  assert(typeof data.sentence_count === 'number' && Number.isInteger(data.sentence_count) && data.sentence_count > 0);
  const selection = object(data.selection_info); method(selection.selected_method); unit(selection.selected_alpha);
  assert(selection.selected_by === 'validation_macro_f1');
  prediction(data.final_result); comparison(data.comparison);
  samePrediction(data.final_result, object(data.comparison)[selection.selected_method as string]);
  assert(Array.isArray(data.sentences) && data.sentences.length === data.sentence_count);
  data.sentences.forEach((raw, index) => {
    const sentence = object(raw); assert(sentence.sentence_index === index && typeof sentence.sentence === 'string');
    prediction(sentence.final_result); comparison(sentence.comparison);
    samePrediction(sentence.final_result, object(sentence.comparison)[selection.selected_method as string]);
  });
  const aggregation = object(data.aggregation_info);
  assert(aggregation.method === 'mean_probability' && aggregation.validated_optimal === false && typeof aggregation.applied_to_final === 'boolean');
  const documents = object(data.document_results);
  for (const name of METHODS) {
    const doc = object(documents[name]); assert(doc.model === name);
    for (const key of ['majority_vote', 'mean_probability', 'confidence_weighted']) {
      const result = object(doc[key]); assert(result.model === name && result.aggregation_method === key);
      assert(LABELS.includes(result.label as typeof LABELS[number])); unit(result.confidence);
      distribution(key === 'majority_vote' ? result.vote_distribution : result.probabilities);
    }
  }
  const display = object(data.display_info); assert(display.default_view === 'user' && typeof display.comparison_view_available === 'boolean');
  const research = object(data.research_info); scores(research.validation_macro_f1);
  if (research.test_macro_f1 !== null) scores(research.test_macro_f1);
  const weights = object(research.ensemble_weights); unit(weights.svm); unit(weights.transformer);
  assert(research.metric_scope === 'dataset_not_input_confidence');
  return value as AnalyzeResponse;
}

export function parseModelInfo(value: unknown): ModelInfoResponse {
  const data = object(value); method(data.selected_method); unit(data.selected_alpha);
  assert(data.selection_source === 'validation' && data.selection_metric === 'macro_f1');
  for (const name of METHODS) {
    assert(typeof object(object(data.models)[name]).name === 'string');
    unit(object(object(data.validation_metrics)[name]).macro_f1);
    if (data.test_metrics !== null) unit(object(object(data.test_metrics)[name]).macro_f1);
  }
  const agg = object(data.document_aggregation);
  assert(agg.default_method === 'mean_probability' && agg.validated_optimal === false);
  assert(data.metric_scope === 'dataset_not_input_confidence' && data.device === 'cpu');
  return value as ModelInfoResponse;
}
