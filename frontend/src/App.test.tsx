import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { analyzeText, getModelInfo, checkHealth, ApiError } from './services/api';
import { parseAnalyzeResponse, parseModelInfo } from './services/contracts';
import type { AnalyzeResponse } from './types/sentiment';
import positiveJson from './test/fixtures/positive.json';
import negativeJson from './test/fixtures/negative.json';
import neutralJson from './test/fixtures/neutral.json';
import paragraphJson from './test/fixtures/paragraph.json';
import infoJson from './test/fixtures/model-info.json';

vi.mock('./services/api', async () => ({ ...(await vi.importActual('./services/api')), analyzeText: vi.fn(), getModelInfo: vi.fn(), checkHealth: vi.fn() }));
const positive = parseAnalyzeResponse(positiveJson), negative = parseAnalyzeResponse(negativeJson);
const neutral = parseAnalyzeResponse(neutralJson), paragraph = parseAnalyzeResponse(paragraphJson);
const info = parseModelInfo(infoJson);
const analyzeMock = vi.mocked(analyzeText);
const input = () => screen.getByRole('textbox', { name: 'Nhập văn bản cần phân tích' });
const result = () => screen.getByRole('region', { name: 'Kết quả phân tích cảm xúc' });
async function submit(data = positive) {
  analyzeMock.mockResolvedValueOnce(data);
  fireEvent.change(input(), { target: { value: data.input_text } });
  await userEvent.click(screen.getByRole('button', { name: 'Phân tích cảm xúc' }));
  await waitFor(() => expect(within(result()).getByText('Độ tin cậy')).toBeInTheDocument());
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getModelInfo).mockResolvedValue(info);
  vi.mocked(checkHealth).mockResolvedValue({ status: 'ok', model_ready: true, service: 'sentiment-analysis-api', version: '1.0.0' });
});

describe('user flow', () => {
  it('renders an accessible empty User Mode without predictions', async () => {
    render(<App />);
    expect(screen.getByRole('tab', { name: 'Người dùng' })).toHaveAttribute('aria-selected', 'true');
    expect(input()).toBeInTheDocument();
    expect(screen.getByText('Nhập câu hoặc đoạn văn để bắt đầu phân tích cảm xúc.')).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'So sánh kết quả các mô hình' })).not.toBeInTheDocument();
    await screen.findByText('Backend sẵn sàng');
    expect(analyzeMock).not.toHaveBeenCalled();
  });
  it.each(['', '  \n\t '])('does not submit empty/whitespace %s', (text) => {
    render(<App />); fireEvent.change(input(), { target: { value: text } });
    fireEvent.submit(screen.getByRole('form', { name: 'Phân tích văn bản' }));
    expect(analyzeMock).not.toHaveBeenCalled();
  });
  it('sends the exact Vietnamese raw text once', async () => {
    render(<App />); const raw = '  Tiếng Việt!\nGiữ nguyên.  ';
    analyzeMock.mockResolvedValueOnce({ ...positive, input_text: raw });
    fireEvent.change(input(), { target: { value: raw } });
    await userEvent.click(screen.getByRole('button', { name: 'Phân tích cảm xúc' }));
    await waitFor(() => expect(analyzeMock).toHaveBeenCalledTimes(1));
    expect(analyzeMock.mock.calls[0][0]).toBe(raw);
  });
  it.each([[positive, 'Tích cực'], [negative, 'Tiêu cực'], [neutral, 'Trung tính']] as const)('shows the Backend label for %s', async (data, label) => {
    render(<App />); await submit(data);
    expect(within(result()).getByRole('heading', { name: label })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'So sánh kết quả các mô hình' })).not.toBeInTheDocument();
  });
  it('renders confidence and bars from final_result, not the more confident Transformer', async () => {
    render(<App />); await submit();
    expect(within(screen.getByRole('group', { name: 'Độ tin cậy cuối cùng' })).getByText('99,36%')).toBeInTheDocument();
    expect(within(result()).queryByText('99,84%')).not.toBeInTheDocument();
    expect(within(result()).getAllByRole('meter')).toHaveLength(3);
    expect(within(result()).getByRole('meter', { name: 'Tiêu cực' })).toHaveAttribute('aria-valuetext', '0,32%');
  });
  it('quick examples only fill text, without generating a prediction', async () => {
    render(<App />); await userEvent.click(screen.getByRole('button', { name: 'Ví dụ hỗn hợp' }));
    expect(input()).toHaveValue(paragraph.input_text); expect(analyzeMock).not.toHaveBeenCalled();
    expect(within(result()).queryByText('Độ tin cậy')).not.toBeInTheDocument();
  });
  it('clears stale results when input changes', async () => {
    render(<App />); await submit(); fireEvent.change(input(), { target: { value: 'Nội dung mới.' } });
    expect(within(result()).queryByText('Độ tin cậy')).not.toBeInTheDocument();
  });
  it('clear button resets the input and prediction', async () => {
    render(<App />); await submit(); await userEvent.click(screen.getByRole('button', { name: 'Xóa văn bản' }));
    expect(input()).toHaveValue(''); expect(within(result()).queryByText('Độ tin cậy')).not.toBeInTheDocument();
  });
  it('does not silently truncate overlength input', () => {
    render(<App />); const text = 'a'.repeat(20001); fireEvent.change(input(), { target: { value: text } });
    expect(input()).toHaveValue(text); expect(screen.getByText(/Vượt giới hạn/)).toBeInTheDocument();
    fireEvent.submit(screen.getByRole('form', { name: 'Phân tích văn bản' })); expect(analyzeMock).not.toHaveBeenCalled();
  });
});

describe('retained canonical response and Research', () => {
  it('toggles modes without another inference or changed final prediction', async () => {
    render(<App />); await submit(paragraph);
    await userEvent.click(screen.getByRole('tab', { name: 'Nghiên cứu' }));
    expect(screen.getByRole('heading', { name: 'So sánh kết quả các mô hình' })).toBeInTheDocument();
    const comparison = screen.getByRole('region', { name: 'So sánh kết quả các mô hình' });
    for (const name of ['TF-IDF + SVM', 'Transformer tự xây', 'Weighted Ensemble']) expect(within(comparison).getByRole('heading', { name })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('tab', { name: 'Người dùng' }));
    expect(within(screen.getByRole('group', { name: 'Độ tin cậy cuối cùng' })).getByText('67,24%')).toBeInTheDocument();
    expect(analyzeMock).toHaveBeenCalledTimes(1);
    expect(input()).toHaveValue(paragraph.input_text);
  });
  it('renders the Backend sentence order and document result, not the final sentence', async () => {
    render(<App />); await submit(paragraph);
    const sentences = screen.getByRole('region', { name: 'Phân tích cảm xúc theo từng câu' });
    const cards = within(sentences).getAllByRole('article');
    expect(cards.map(card => within(card).getByTestId('sentence-text').textContent)).toEqual(['Camera rất đẹp.', 'Nhưng pin quá yếu.', 'Giao hàng rất nhanh!']);
    expect(within(screen.getByRole('group', { name: 'Độ tin cậy cuối cùng' })).getByText('67,24%')).toBeInTheDocument();
    expect(within(result()).queryByText('97,15%')).not.toBeInTheDocument();
  });
  it('does not repeat a single-sentence result', async () => {
    render(<App />); await submit();
    expect(screen.queryByRole('region', { name: 'Phân tích cảm xúc theo từng câu' })).not.toBeInTheDocument();
  });
  it('shows dataset metrics from model-info distinct from confidence', async () => {
    render(<App />); await userEvent.click(screen.getByRole('tab', { name: 'Nghiên cứu' }));
    const table = await screen.findByRole('table', { name: 'Số liệu Macro-F1' });
    expect(within(table).getByText('0.825546')).toBeInTheDocument();
    expect(within(table).getByText('0.822183')).toBeInTheDocument();
    expect(screen.getByText(/Tập Test chỉ được sử dụng để đánh giá cuối cùng/)).toBeInTheDocument();
    expect(analyzeMock).not.toHaveBeenCalled();
  });
  it('shows all aggregation methods and labels vote share correctly', async () => {
    render(<App />); await submit(paragraph); await userEvent.click(screen.getByRole('tab', { name: 'Nghiên cứu' }));
    const region = screen.getByRole('region', { name: 'So sánh phương pháp tổng hợp cảm xúc toàn đoạn' });
    expect(within(region).getByText('Majority Voting')).toBeInTheDocument();
    expect(within(region).getByText('66,67% phiếu')).toBeInTheDocument();
    expect(within(region).getByText(/chưa được chứng minh tối ưu/)).toBeInTheDocument();
    await userEvent.click(within(region).getByRole('tab', { name: 'SVM' }));
    expect(within(region).getByText('66,04%')).toBeInTheDocument();
    expect(analyzeMock).toHaveBeenCalledTimes(1);
  });
});

describe('pending/error/race behavior', () => {
  it('ignores a delayed startup health failure after analysis already proved readiness', async () => {
    let fail!: (reason: ApiError) => void;
    vi.mocked(checkHealth).mockImplementationOnce(() => new Promise((_resolve, reject) => { fail = reject; }));
    render(<App />); await submit(); await act(async () => fail(new ApiError('network')));
    expect(screen.getByText('Backend sẵn sàng')).toBeInTheDocument();
  });
  it('updates connectivity after a successful analysis when the initial health check failed', async () => {
    vi.mocked(checkHealth).mockRejectedValueOnce(new ApiError('network'));
    render(<App />); await screen.findByText('Chưa kết nối Backend'); await submit();
    expect(screen.getByText('Backend sẵn sàng')).toBeInTheDocument();
  });
  it('uses response research metadata if model-info was unavailable', async () => {
    vi.mocked(getModelInfo).mockRejectedValueOnce(new ApiError('network'));
    render(<App />); await submit(); await userEvent.click(screen.getByRole('tab', { name: 'Nghiên cứu' }));
    expect(within(screen.getByRole('table', { name: 'Số liệu Macro-F1' })).getByText('0.825546')).toBeInTheDocument();
  });
  it('supports keyboard mode navigation without inference', async () => {
    render(<App />); const tab = screen.getByRole('tab', { name: 'Người dùng' }); tab.focus();
    fireEvent.keyDown(tab, { key: 'ArrowRight' });
    expect(screen.getByRole('tab', { name: 'Nghiên cứu' })).toHaveFocus();
    expect(screen.getByRole('tab', { name: 'Nghiên cứu' })).toHaveAttribute('aria-selected', 'true');
    expect(analyzeMock).not.toHaveBeenCalled();
  });
  it('rejects a successful response belonging to a different input', async () => {
    analyzeMock.mockResolvedValueOnce(positive); render(<App />);
    fireEvent.change(input(), { target: { value: negative.input_text } });
    await userEvent.click(screen.getByRole('button', { name: 'Phân tích cảm xúc' }));
    await screen.findByRole('alert'); expect(within(result()).queryByText('Độ tin cậy')).not.toBeInTheDocument();
  });
  it('shows loading and guards double submits synchronously', async () => {
    let resolve!: (r: AnalyzeResponse) => void;
    analyzeMock.mockImplementationOnce(() => new Promise(res => { resolve = res; }));
    render(<App />); fireEvent.change(input(), { target: { value: positive.input_text } });
    const form = screen.getByRole('form', { name: 'Phân tích văn bản' }); fireEvent.submit(form); fireEvent.submit(form);
    expect(screen.getByRole('button', { name: 'Đang phân tích...' })).toBeDisabled();
    expect(screen.getByText('AI đang phân tích cảm xúc văn bản...')).toBeInTheDocument(); expect(analyzeMock).toHaveBeenCalledTimes(1);
    await act(async () => resolve(positive));
  });
  it.each(['network', 'server', 'validation', 'unavailable', 'timeout'] as const)('shows safe %s error and no fake result', async (code) => {
    analyzeMock.mockRejectedValueOnce(new ApiError(code)); render(<App />);
    fireEvent.change(input(), { target: { value: 'Văn bản.' } }); await userEvent.click(screen.getByRole('button', { name: 'Phân tích cảm xúc' }));
    await screen.findByRole('alert'); expect(within(result()).queryByText('Độ tin cậy')).not.toBeInTheDocument();
  });
  it('explicit cancel ignores a later successful response', async () => {
    let resolve!: (r: AnalyzeResponse) => void;
    analyzeMock.mockImplementationOnce(() => new Promise(res => { resolve = res; })); render(<App />);
    fireEvent.change(input(), { target: { value: positive.input_text } }); fireEvent.submit(screen.getByRole('form', { name: 'Phân tích văn bản' }));
    await userEvent.click(screen.getByRole('button', { name: 'Hủy phân tích' }));
    await act(async () => resolve(positive));
    expect(within(result()).queryByText('Độ tin cậy')).not.toBeInTheDocument();
  });
  it('an old request cannot overwrite a newer input/result even if abort is ignored', async () => {
    let old!: (r: AnalyzeResponse) => void, recent!: (r: AnalyzeResponse) => void;
    analyzeMock.mockImplementationOnce(() => new Promise(res => { old = res; })).mockImplementationOnce(() => new Promise(res => { recent = res; }));
    render(<App />); fireEvent.change(input(), { target: { value: positive.input_text } }); fireEvent.submit(screen.getByRole('form', { name: 'Phân tích văn bản' }));
    fireEvent.change(input(), { target: { value: negative.input_text } }); fireEvent.submit(screen.getByRole('form', { name: 'Phân tích văn bản' }));
    await act(async () => recent(negative)); await act(async () => old(positive));
    expect(within(result()).getByRole('heading', { name: 'Tiêu cực' })).toBeInTheDocument(); expect(analyzeMock).toHaveBeenCalledTimes(2);
  });
});
