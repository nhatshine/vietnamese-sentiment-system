import { expect, test } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { AnalyzeResponse, ModelInfoResponse } from '../src/types/sentiment';

const inputs = [
  'Sản phẩm này rất tốt.',
  'Tôi rất thất vọng vì sản phẩm quá tệ.',
  'Sản phẩm này bình thường.',
  'Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!',
  'Giá sản phẩm là 1.5 triệu đồng. TS. Nam đánh giá chất lượng tốt.',
];
const names = { positive: 'Tích cực', negative: 'Tiêu cực', neutral: 'Trung tính' };
const percent = (value: number) => `${(value * 100).toLocaleString('vi-VN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
const output = (name: string) => join(process.env.E2E_RESULTS_DIR ?? 'e2e-results', name);

test('five real inputs, exact canonical display, research without inference, responsive views', async ({ page }) => {
  mkdirSync(output(''), { recursive: true });
  const posts: string[] = [], errors: string[] = [];
  const records: { text: string; response: AnalyzeResponse; postsBeforeToggle: number; postsAfterToggle: number }[] = [];
  page.on('request', request => { if (request.method() === 'POST' && request.url().endsWith('/api/analyze')) posts.push(request.postData() || ''); });
  page.on('pageerror', error => errors.push(error.message));
  const metadata = page.waitForResponse(r => r.url().endsWith('/api/model-info'));
  await page.goto('/');
  const modelInfo: ModelInfoResponse = await (await metadata).json();
  expect(modelInfo.selected_alpha).toBe(.7);
  expect(modelInfo.models.transformer.name).toContain('epoch 21');
  expect(modelInfo.test_metrics?.ensemble.macro_f1).toBeCloseTo(.8252706095647272, 10);
  await expect(page.getByText('Backend sẵn sàng')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Phân tích cảm xúc' })).toBeInViewport({ ratio: 1 });
  await page.screenshot({ animations: 'disabled', path: output('desktop-empty.png') });
  const region = page.getByRole('region', { name: 'Kết quả phân tích cảm xúc' });
  for (const [index, text] of inputs.entries()) {
    await page.getByRole('textbox', { name: 'Nhập văn bản cần phân tích' }).fill(text);
    await expect(region.getByRole('group', { name: 'Độ tin cậy cuối cùng' })).toHaveCount(0);
    const received = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith('/api/analyze'));
    await page.getByRole('button', { name: 'Phân tích cảm xúc' }).click();
    const http = await received; expect(http.status()).toBe(200);
    expect(http.headers()['access-control-allow-origin']).toBe('http://localhost:5173');
    const response: AnalyzeResponse = await http.json();
    expect(response.input_text).toBe(text);
    expect(response.selection_info.selected_alpha).toBe(.7);
    expect(response.research_info.ensemble_weights.svm).toBe(.7);
    expect(response.research_info.ensemble_weights.transformer).toBeCloseTo(.3, 12);
    await expect(region.getByRole('heading', { name: names[response.final_result.label], exact: true })).toBeVisible();
    await expect(region.getByRole('group', { name: 'Độ tin cậy cuối cùng' })).toContainText(percent(response.final_result.confidence));
    for (const label of ['negative', 'neutral', 'positive'] as const) {
      await expect(region.getByRole('meter', { name: names[label], exact: true })).toHaveAttribute('aria-valuetext', percent(response.final_result.probabilities[label]));
    }
    if (response.sentence_count > 1) {
      const sentences = page.getByRole('region', { name: 'Phân tích cảm xúc theo từng câu' });
      await expect(sentences.getByRole('article')).toHaveCount(response.sentence_count);
      expect(await sentences.getByTestId('sentence-text').allTextContents()).toEqual(response.sentences.map(s => s.sentence));
    }
    const before = posts.length;
    if (index === 3) { await page.evaluate(() => window.scrollTo(0, 0)); await page.screenshot({ animations: 'disabled', path: output('desktop-user.png') }); }
    await page.getByRole('tab', { name: 'Nghiên cứu', exact: true }).click();
    await expect(page.getByRole('region', { name: 'So sánh kết quả các mô hình' })).toBeVisible();
    for (const method of ['svm', 'transformer', 'ensemble'] as const) {
      await expect(page.getByRole('table', { name: 'Số liệu Macro-F1' })).toContainText(modelInfo.validation_metrics[method].macro_f1.toFixed(6));
      if (modelInfo.test_metrics) await expect(page.getByRole('table', { name: 'Số liệu Macro-F1' })).toContainText(modelInfo.test_metrics[method].macro_f1.toFixed(6));
    }
    if (index === 3) {
      await page.getByRole('region', { name: 'So sánh kết quả các mô hình' }).scrollIntoViewIfNeeded();
      await page.screenshot({ animations: 'disabled', path: output('desktop-research.png') });
      const aggregation = page.getByRole('region', { name: 'So sánh phương pháp tổng hợp cảm xúc toàn đoạn' });
      await expect(aggregation.getByText(`${percent(response.document_results.ensemble.majority_vote.confidence)} phiếu`)).toBeVisible();
      await aggregation.getByRole('tab', { name: 'SVM', exact: true }).click();
      await expect(aggregation.getByText(percent(response.document_results.svm.mean_probability.confidence))).toBeVisible();
      for (const width of [1920, 768, 390, 360]) {
        await page.setViewportSize({ width, height: width <= 390 ? 844 : 900 });
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
        if (width === 1920) { await page.evaluate(() => window.scrollTo(0, 0)); await page.screenshot({ animations: 'disabled', path: output('fullhd-research.png') }); }
        if (width === 768) { await page.getByRole('region', { name: 'So sánh kết quả các mô hình' }).scrollIntoViewIfNeeded(); await page.screenshot({ animations: 'disabled', path: output('tablet-research.png') }); }
      }
      await page.getByRole('tab', { name: 'Người dùng', exact: true }).click();
      await page.setViewportSize({ width: 390, height: 844 });
      await page.evaluate(() => window.scrollTo(0, 0)); await page.screenshot({ animations: 'disabled', path: output('mobile-top.png') });
      await region.scrollIntoViewIfNeeded(); await page.screenshot({ animations: 'disabled', path: output('mobile-result.png') });
      await page.getByRole('tab', { name: 'Nghiên cứu', exact: true }).click();
      await page.getByRole('region', { name: 'So sánh kết quả các mô hình' }).scrollIntoViewIfNeeded();
      await page.screenshot({ animations: 'disabled', path: output('mobile-research.png') });
      await page.setViewportSize({ width: 1366, height: 768 });
    }
    await page.getByRole('tab', { name: 'Người dùng', exact: true }).click();
    await expect(page.getByRole('region', { name: 'So sánh kết quả các mô hình' })).toHaveCount(0);
    await page.waitForLoadState('networkidle');
    expect(posts.length).toBe(before);
    expect(posts.length).toBe(index + 1);
    records.push({ text, response, postsBeforeToggle: before, postsAfterToggle: posts.length });
  }
  expect(errors).toEqual([]);
  expect(records[4].response.sentences.map(s => s.sentence)).toEqual(['Giá sản phẩm là 1.5 triệu đồng.', 'TS. Nam đánh giá chất lượng tốt.']);
  writeFileSync(output('integration.json'), JSON.stringify({ realBackend: true, totalInferenceRequests: posts.length, cors: 'passed', responsiveWidths: [1366, 1920, 768, 390, 360], browserErrors: errors, modelInfo, records }, null, 2));
});
