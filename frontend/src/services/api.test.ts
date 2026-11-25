import { afterEach, describe, expect, it, vi } from 'vitest';
import paragraph from '../test/fixtures/paragraph.json';
import info from '../test/fixtures/model-info.json';
import { createApiClient } from './api';
import { parseAnalyzeResponse, parseModelInfo } from './contracts';
import { characterCount, formatPercent, formatMetric } from '../utils/formatters';

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

describe('formatting only, no prediction math', () => {
  it('formats probabilities and dataset F1 distinctly', () => {
    expect(formatPercent(.8542)).toBe('85,42%');
    expect(formatMetric(.8255456672498939)).toBe('0.825546');
  });
  it('counts Unicode code points like the Backend', () => expect(characterCount('a😀ế')).toBe(3));
});

describe('actual canonical boundary', () => {
  it('preserves full precision, extras and original reference', () => expect(parseAnalyzeResponse(paragraph)).toBe(paragraph));
  it('accepts real model metadata without embedding metric constants', () => expect(parseModelInfo(info)).toBe(info));
  it.each(['shape', 'NaN', 'label', 'comparison', 'sentences', 'final', 'aggregation'])('rejects malformed %s', (kind) => {
    const data = structuredClone(paragraph);
    if (kind === 'shape') expect(() => parseAnalyzeResponse({})).toThrow();
    else {
      if (kind === 'NaN') data.final_result.confidence = NaN;
      if (kind === 'label') Object.assign(data.final_result, { label: 'other' });
      if (kind === 'comparison') Reflect.deleteProperty(data.comparison, 'svm');
      if (kind === 'sentences') data.sentence_count = 7;
      if (kind === 'final') data.final_result.label = 'negative';
      if (kind === 'aggregation') Reflect.deleteProperty(data.document_results.ensemble, 'majority_vote');
      expect(() => parseAnalyzeResponse(data)).toThrow();
    }
  });
});

describe('API client', () => {
  it('sends raw text once to configured URL, no selection override or credentials', async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify(paragraph), { status: 200 }));
    vi.stubGlobal('fetch', fetcher);
    const client = createApiClient('http://localhost:8000/api/', 1000);
    const text = '  Tiếng Việt.\nGiữ nguyên!  ';
    await client.analyzeText(text);
    expect(fetcher).toHaveBeenCalledTimes(1);
    const [url, options] = fetcher.mock.calls[0];
    expect(url).toBe('http://localhost:8000/api/analyze');
    expect(JSON.parse(options.body)).toEqual({ text });
    expect(options.credentials).toBe('omit');
  });
  it.each([422, 503, 500])('localizes HTTP %s without exposing server details', async (status) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"detail":"C:/private traceback"}', { status })));
    await expect(createApiClient().analyzeText('x')).rejects.toMatchObject({ status });
    try { await createApiClient().analyzeText('x'); } catch (error) { expect(String(error)).not.toContain('C:/private'); }
  });
  it('handles network failure without sample prediction', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    await expect(createApiClient().analyzeText('x')).rejects.toMatchObject({ code: 'network' });
  });
  it('handles invalid successful JSON as a contract error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}')));
    await expect(createApiClient().analyzeText('x')).rejects.toMatchObject({ code: 'contract' });
  });
  it('reports timeout distinctly', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', vi.fn((_url: unknown, options: RequestInit) => new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    })));
    const pending = createApiClient(undefined, 10).analyzeText('x');
    const assertion = expect(pending).rejects.toMatchObject({ code: 'timeout' });
    await vi.advanceTimersByTimeAsync(20); await assertion;
  });
  it('reports explicit cancellation', async () => {
    vi.stubGlobal('fetch', vi.fn((_url: unknown, options: RequestInit) => new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    })));
    const controller = new AbortController();
    const pending = createApiClient().analyzeText('x', controller.signal);
    const assertion = expect(pending).rejects.toMatchObject({ code: 'cancelled' });
    controller.abort(); await assertion;
  });
});
