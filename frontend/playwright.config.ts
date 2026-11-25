import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 90000,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: 'list',
  use: { baseURL: 'http://localhost:5173', channel: 'chrome', headless: true, viewport: { width: 1366, height: 768 }, screenshot: 'only-on-failure' },
  webServer: [
    { command: 'python -m uvicorn backend.src.api:app --host 127.0.0.1 --port 8000', cwd: '..', url: 'http://127.0.0.1:8000/api/health', reuseExistingServer: true, timeout: 60000 },
    { command: 'npm run dev', url: 'http://localhost:5173', reuseExistingServer: true, timeout: 30000 },
  ],
});
