import { useEffect, useRef, useState } from 'react';
import { analyzeText, ApiError } from '../services/api';
import type { AnalyzeResponse } from '../types/sentiment';
import { characterCount } from '../utils/formatters';

export const MAX_TEXT_CHARS = Number(import.meta.env.VITE_MAX_TEXT_CHARS || 20000);
if (!Number.isInteger(MAX_TEXT_CHARS) || MAX_TEXT_CHARS < 1) throw new Error('VITE_MAX_TEXT_CHARS phải là số nguyên dương.');

export function useAnalysis() {
  const [inputText, setInputText] = useState('');
  const [canonicalResponse, setCanonicalResponse] = useState<AnalyzeResponse | null>(null);
  const [isAnalyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0), active = useRef(false), controller = useRef<AbortController | null>(null);
  useEffect(() => () => { generation.current++; controller.current?.abort(); }, []);

  function invalidate() { generation.current++; controller.current?.abort(); controller.current = null; active.current = false; setAnalyzing(false); }
  function editText(text: string) { invalidate(); setInputText(text); setCanonicalResponse(null); setError(null); }
  function cancel() { invalidate(); setError(new ApiError('cancelled').message); }
  async function submit() {
    if (active.current) return;
    if (!inputText.trim()) { setError('Vui lòng nhập câu hoặc đoạn văn tiếng Việt trước khi phân tích.'); return; }
    if (characterCount(inputText) > MAX_TEXT_CHARS) { setError(`Vượt giới hạn ${MAX_TEXT_CHARS.toLocaleString('vi-VN')} ký tự. Văn bản chưa được gửi và không bị cắt bớt.`); return; }
    const id = ++generation.current, abort = new AbortController(), submittedText = inputText;
    controller.current = abort; active.current = true; setAnalyzing(true); setCanonicalResponse(null); setError(null);
    try {
      const response = await analyzeText(submittedText, abort.signal);
      if (generation.current !== id || abort.signal.aborted) return;
      if (response.input_text !== submittedText) throw new ApiError('contract');
      setCanonicalResponse(response);
    } catch (failure) {
      if (generation.current === id) setError(failure instanceof ApiError ? failure.message : new ApiError('server').message);
    } finally {
      if (generation.current === id) { active.current = false; setAnalyzing(false); controller.current = null; }
    }
  }
  return { inputText, canonicalResponse, isAnalyzing, error, editText, submit, cancel };
}
