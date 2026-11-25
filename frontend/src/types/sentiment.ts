export const LABELS = ['negative', 'neutral', 'positive'] as const;
export const METHODS = ['svm', 'transformer', 'ensemble'] as const;
export type SentimentLabel = typeof LABELS[number];
export type Method = typeof METHODS[number];
export type ViewMode = 'user' | 'research';
export type Probabilities = Record<SentimentLabel, number>;
export interface PredictionResult {
  method: Method;
  label: SentimentLabel;
  confidence: number;
  probabilities: Probabilities;
  weights?: Record<'svm' | 'transformer', number>;
  [key: string]: unknown;
}
export type Comparison = Record<Method, PredictionResult>;
export interface SentenceResult {
  sentence_index: number;
  sentence: string;
  final_result: PredictionResult;
  comparison: Comparison;
}
export interface SelectionInfo { selected_method: Method; selected_alpha: number; selected_by: 'validation_macro_f1' }
export interface AggregationInfo { method: 'mean_probability'; applied_to_final: boolean; validated_optimal: false; policy: string }
export interface ProbabilityAggregation {
  model: Method; aggregation_method: 'mean_probability' | 'confidence_weighted';
  label: SentimentLabel; confidence: number; probabilities: Probabilities;
}
export interface MajorityAggregation {
  model: Method; aggregation_method: 'majority_vote'; label: SentimentLabel;
  confidence: number; vote_counts: Record<SentimentLabel, number>; vote_distribution: Probabilities;
}
export interface ModelDocument {
  model: Method; majority_vote: MajorityAggregation;
  mean_probability: ProbabilityAggregation; confidence_weighted: ProbabilityAggregation;
}
export interface AnalyzeResponse {
  input_text: string; sentence_count: number; sentences: SentenceResult[];
  document_results: Record<Method, ModelDocument>;
  final_result: PredictionResult; selection_info: SelectionInfo; aggregation_info: AggregationInfo;
  comparison: Comparison;
  display_info: { default_view: 'user'; comparison_view_available: boolean };
  research_info: {
    validation_macro_f1: Record<Method, number>; test_macro_f1: Record<Method, number> | null;
    ensemble_weights: Record<'svm' | 'transformer', number>;
    metric_scope: 'dataset_not_input_confidence'; frozen_selection_verified: boolean;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}
export interface ModelInfoResponse {
  selected_method: Method; selected_alpha: number; selection_source: 'validation'; selection_metric: 'macro_f1';
  models: Record<Method, { name: string }>;
  validation_metrics: Record<Method, { macro_f1: number }>;
  test_metrics: Record<Method, { macro_f1: number }> | null;
  metric_scope: 'dataset_not_input_confidence';
  document_aggregation: { default_method: 'mean_probability'; validated_optimal: false };
  device: 'cpu';
}
export interface HealthResponse { status: 'ok'; service: string; version: string; model_ready: boolean }
