import type { Method, SentimentLabel } from '../types/sentiment';

export const LABEL_NAMES: Record<SentimentLabel, string> = { negative: 'Tiêu cực', neutral: 'Trung tính', positive: 'Tích cực' };
export const MODEL_NAMES: Record<Method, string> = { svm: 'TF-IDF + SVM', transformer: 'Transformer tự xây', ensemble: 'Weighted Ensemble' };
export const formatPercent = (value: number) => `${new Intl.NumberFormat('vi-VN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value * 100)}%`;
export const formatMetric = (value: number) => value.toFixed(6);
export const characterCount = (text: string) => Array.from(text).length;
