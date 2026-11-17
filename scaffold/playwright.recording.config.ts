import { defineConfig, devices } from '@playwright/test';

const vw = Number(process.env.RECORDING_VIEWPORT_WIDTH) || 1920;
const vh = Number(process.env.RECORDING_VIEWPORT_HEIGHT) || 1080;
const slowMo = Number(process.env.RECORDING_SLOW_MO_MS) || 300;
const colorScheme = (process.env.RECORDING_COLOR_SCHEME as 'light' | 'dark') || 'light';
const timeout = Number(process.env.RECORDING_TIMEOUT_MS) || 240_000;

export default defineConfig({
  testDir: './parts',
  fullyParallel: false,
  forbidOnly: true,
  retries: 0,
  workers: 1,
  timeout,
  reporter: [['html', { outputFolder: 'recording-report' }]],
  use: {
    ...devices['Desktop Chrome'],
    viewport: { width: vw, height: vh },
    video: { mode: 'on', size: { width: vw, height: vh } },
    launchOptions: { slowMo },
    colorScheme,
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
    trace: 'off',
    screenshot: 'off',
  },
});
