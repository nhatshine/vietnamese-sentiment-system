import type { HealthResponse } from '../types/sentiment';
import { ContractError, parseAnalyzeResponse, parseModelInfo } from './contracts';

type ErrorCode = 'validation' | 'unavailable' | 'server' | 'network' | 'timeout' | 'cancelled' | 'contract';
const MESSAGES: Record<ErrorCode, string> = {
  validation: 'Văn bản không hợp lệ hoặc một câu không còn nội dung sau xử lý. Vui lòng kiểm tra nội dung và giới hạn ký tự.',
  unavailable: 'Mô hình chưa sẵn sàng. Vui lòng chờ máy chủ khởi động rồi thử lại.',
  server: 'Máy chủ chưa thể hoàn tất phân tích. Vui lòng thử lại; không có kết quả dự đoán thay thế.',
  network: 'Không thể kết nối tới máy chủ phân tích. Vui lòng kiểm tra FastAPI Backend.',
  timeout: 'Yêu cầu đã quá thời gian chờ. Vui lòng thử với văn bản ngắn hơn hoặc kiểm tra máy chủ.',
  cancelled: 'Đã hủy yêu cầu phân tích.',
  contract: 'Dữ liệu phản hồi không đúng định dạng. Vui lòng kiểm tra phiên bản FastAPI Backend.',
};
export class ApiError extends Error {
  constructor(public code: ErrorCode, public status = 0) { super(MESSAGES[code]); this.name = 'ApiError'; }
}

export function createApiClient(baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api', timeoutMs = Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS || 120000)) {
  const base = baseUrl.replace(/\/+$/, '');
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Cấu hình timeout phải là số dương.');
  async function request<T>(path: string, parser: (data: unknown) => T, text?: string, parentSignal?: AbortSignal): Promise<T> {
    const controller = new AbortController(); let timedOut = false;
    const abort = () => controller.abort();
    parentSignal?.addEventListener('abort', abort, { once: true });
    if (parentSignal?.aborted) controller.abort();
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
    try {
      if (controller.signal.aborted) throw new ApiError('cancelled');
      const response = await fetch(`${base}${path}`, {
        method: text === undefined ? 'GET' : 'POST', credentials: 'omit', signal: controller.signal,
        ...(text === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) }),
      });
      if (!response.ok) throw new ApiError(response.status === 422 ? 'validation' : response.status === 503 ? 'unavailable' : 'server', response.status);
      try { return parser(await response.json()); } catch { throw new ApiError('contract'); }
    } catch (error) {
      if (timedOut) throw new ApiError('timeout');
      if (controller.signal.aborted) throw new ApiError('cancelled');
      if (error instanceof ApiError) throw error;
      if (error instanceof ContractError) throw new ApiError('contract');
      throw new ApiError('network');
    } finally { clearTimeout(timer); parentSignal?.removeEventListener('abort', abort); }
  }
  return {
    analyzeText: (text: string, signal?: AbortSignal) => request('/analyze', parseAnalyzeResponse, text, signal),
    getModelInfo: (signal?: AbortSignal) => request('/model-info', parseModelInfo, undefined, signal),
    checkHealth: (signal?: AbortSignal) => request('/health', (data) => {
      if (typeof data !== 'object' || data === null || !('status' in data) || data.status !== 'ok' || !('model_ready' in data) || typeof data.model_ready !== 'boolean') throw new ContractError();
      return data as HealthResponse;
    }, undefined, signal),
  };
}
export const { analyzeText, getModelInfo, checkHealth } = createApiClient();
