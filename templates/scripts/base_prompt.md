# Playwright Demo Script Generation

You are generating a Playwright demo script that will be recorded as a video.

## Script Structure

```typescript
import { test } from '@playwright/test';
import {
  showChapterTitle,
  highlightElement,
  enableCursorTrail,
  smoothScroll,
  slowType,
  slowClick,
  slowHover,
  pause,
  waitForVisualStability,
  spotlightElement,
} from '../helpers/demo-helpers';

test.describe('Part Title', () => {
  test('part-id: Part Title', async ({ page }) => {
    await enableCursorTrail(page);

    // Step 1: Navigate
    await test.step('Navigate to page', async () => {
      await page.goto('https://example.com');
      await waitForVisualStability(page);
      await pause(page, 1500);
    });

    // Step 2: Interact
    await test.step('Show key features', async () => {
      await highlightElement(page, 'selector', 'Label');
      await pause(page, 2000);
    });
  });
});
```

## Available Demo Helpers API

| Function | Description |
|----------|-------------|
| `showChapterTitle(page, { chapter, title, subtitle?, duration? })` | Full-screen gradient title card |
| `highlightElement(page, selector, label?, durationMs?)` | Pulsing border + label on element |
| `spotlightElement(page, selector, label?, durationMs?)` | SVG mask dimming rest of page |
| `enableCursorTrail(page)` | Visible cursor dot with trail |
| `slowType(page, locator, text, { delay? })` | Character-by-character typing |
| `slowClick(page, selector)` | Deliberate approach → pause → click |
| `slowHover(page, selector)` | Gradual mouse approach |
| `smoothScroll(page, { direction?, distance?, duration? })` | Eased scroll animation |
| `pause(page, ms)` | Fixed delay |
| `waitForVisualStability(page, extraMs?)` | Wait for spinners + network idle |

## Recording Constraints

- Viewport: 1920x1080
- slowMo: 300ms
- Color scheme: light
- Single browser, sequential execution

## Quality Guidelines

- **Pacing**: 2-3 seconds between actions so viewers can follow
- **Highlights**: Use `highlightElement` before interacting with elements
- **Navigation**: Always call `waitForVisualStability(page)` after `page.goto()`
- **Cursor**: Call `enableCursorTrail(page)` at the start
- **Scroll**: Use `smoothScroll` instead of instant scrolling
- **Typing**: Use `slowType` for form fields
- **Duration**: Keep each part under 90 seconds

## Completion

End your response with: `<promise>COMPLETE</promise>`
