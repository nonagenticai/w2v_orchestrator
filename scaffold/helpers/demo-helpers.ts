/**
 * demo-helpers.ts — Visual effects library for demo video recordings.
 *
 * Part of the web_to_video pipeline.
 * All colors are parameterized via CSS custom properties for theming.
 *
 * Usage:
 *   import { showChapterTitle, annotateAndWait, highlightElement, ... } from '../helpers/demo-helpers';
 */

import { Page, Locator } from '@playwright/test';

// ---------------------------------------------------------------------------
// Theme — CSS custom properties for colors (set via setDemoTheme)
// ---------------------------------------------------------------------------

export interface DemoTheme {
  primaryColor: string;
  backgroundColor: string;
  textColor: string;
  accentColor: string;
  subtitleColor: string;
  gradientStart: string;
  gradientEnd: string;
}

const DEFAULT_THEME: DemoTheme = {
  primaryColor: '#6366f1',
  backgroundColor: '#0f172a',
  textColor: '#f8fafc',
  accentColor: '#818cf8',
  subtitleColor: '#94a3b8',
  gradientStart: '#0f172a',
  gradientEnd: '#1e293b',
};

let currentTheme: DemoTheme = { ...DEFAULT_THEME };

/**
 * Set the demo theme. Call this before any visual helpers.
 */
export function setDemoTheme(theme: Partial<DemoTheme>): void {
  currentTheme = { ...DEFAULT_THEME, ...theme };
}

// ---------------------------------------------------------------------------
// Safe evaluate — wraps page.evaluate to survive navigation context destruction
// ---------------------------------------------------------------------------

async function safeEvaluate(page: Page, fn: Parameters<Page['evaluate']>[0], arg?: unknown): Promise<void> {
  try {
    await page.evaluate(fn as never, arg as never);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes('context was destroyed') || msg.includes('navigation')) {
      return;
    }
    throw e;
  }
}

// ---------------------------------------------------------------------------
// Annotation overlay — DISABLED (will be replaced by slide_service)
// Functions kept as no-ops so existing scripts compile without errors.
// ---------------------------------------------------------------------------

export type Position = 'top-left' | 'top-center' | 'top-right' | 'bottom-left' | 'bottom-center' | 'bottom-right';

export interface AnnotationOptions {
  text: string;
  position?: Position;
  duration?: number;
  fontSize?: number;
}

/** @deprecated No-op — text overlays removed. Use slide_service instead. */
export async function showAnnotation(_page: Page, _options: AnnotationOptions): Promise<void> {}

/** @deprecated No-op — text overlays removed. */
export async function hideAnnotation(_page: Page): Promise<void> {}

// ---------------------------------------------------------------------------
// Pause / wait utilities
// ---------------------------------------------------------------------------

/**
 * Pause for a fixed duration.
 */
export async function pause(page: Page, ms: number = 2000): Promise<void> {
  await page.waitForTimeout(ms);
}

/**
 * Wait for the page to reach a visually stable state.
 */
export async function waitForVisualStability(page: Page, extraMs: number = 500): Promise<void> {
  const spinner = page.locator('[data-testid="loading-spinner"], .animate-spin, [role="progressbar"]');
  if (await spinner.count() > 0) {
    await spinner.first().waitFor({ state: 'hidden', timeout: 15000 }).catch(() => {});
  }
  await page.waitForLoadState('networkidle', { timeout: 5000 }).catch(() => {});
  if (extraMs > 0) {
    await page.waitForTimeout(extraMs);
  }
}

/**
 * @deprecated No-op — text overlays removed. Retains pause for recording timing.
 */
export async function annotateAndWait(
  page: Page,
  _text: string,
  waitMs: number = 3000,
  _position: Position = 'bottom-center',
): Promise<void> {
  await pause(page, waitMs);
}

// ---------------------------------------------------------------------------
// Chapter title card — full-screen overlay
// ---------------------------------------------------------------------------

export interface ChapterTitleOptions {
  chapter: number;
  title: string;
  subtitle?: string;
  duration?: number;
}

/**
 * Display a full-screen chapter title card in the browser viewport.
 */
export async function showChapterTitle(page: Page, options: ChapterTitleOptions): Promise<void> {
  const { chapter, title, subtitle, duration = 4000 } = options;
  const theme = currentTheme;

  await safeEvaluate(
    page,
    ({ ch, ttl, sub, primary, text, subtitleColor, gradStart, gradEnd }: {
      ch: number; ttl: string; sub?: string;
      primary: string; text: string; subtitleColor: string;
      gradStart: string; gradEnd: string;
    }) => {
      const overlay = document.createElement('div');
      overlay.id = 'w2v-chapter-title';
      overlay.style.cssText = `
        position: fixed;
        inset: 0;
        z-index: 100000;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, ${gradStart} 0%, ${gradEnd} 50%, ${gradStart} 100%);
        opacity: 0;
        transition: opacity 0.5s ease-in-out;
      `;
      overlay.innerHTML = `
        <div style="color: ${primary}; font-family: 'Inter', sans-serif; font-size: 16px;
          font-weight: 600; letter-spacing: 4px; text-transform: uppercase; margin-bottom: 16px;">
          Chapter ${ch}
        </div>
        <div style="color: ${text}; font-family: 'Inter', sans-serif; font-size: 48px;
          font-weight: 700; text-align: center; max-width: 800px; line-height: 1.2;">
          ${ttl}
        </div>
        ${sub ? `<div style="color: ${subtitleColor}; font-family: 'Inter', sans-serif;
          font-size: 20px; font-weight: 400; text-align: center; max-width: 600px;
          margin-top: 16px; line-height: 1.5;">${sub}</div>` : ''}
      `;
      document.body.appendChild(overlay);
      requestAnimationFrame(() => { overlay.style.opacity = '1'; });
    },
    {
      ch: chapter, ttl: title, sub: subtitle,
      primary: theme.primaryColor, text: theme.textColor,
      subtitleColor: theme.subtitleColor,
      gradStart: theme.gradientStart, gradEnd: theme.gradientEnd,
    },
  );

  await pause(page, duration);

  await safeEvaluate(page, () => {
    const el = document.getElementById('w2v-chapter-title');
    if (el) {
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 600);
    }
  });

  await pause(page, 700);
}

// ---------------------------------------------------------------------------
// Highlight element — pulsing border + optional label
// ---------------------------------------------------------------------------

/**
 * Briefly highlight a page element with a pulsing border and optional label.
 */
export async function highlightElement(
  page: Page,
  selector: string,
  label?: string,
  durationMs: number = 2500,
): Promise<void> {
  const locator = page.locator(selector).first();
  const box = await locator.boundingBox().catch(() => null);
  if (!box) return;
  const theme = currentTheme;

  await safeEvaluate(
    page,
    ({ rect, lbl, dur, color }: {
      rect: { x: number; y: number; width: number; height: number };
      lbl?: string; dur: number; color: string;
    }) => {
      const highlight = document.createElement('div');
      highlight.id = 'w2v-highlight';
      highlight.style.cssText = `
        position: fixed;
        top: ${rect.y - 4}px; left: ${rect.x - 4}px;
        width: ${rect.width + 8}px; height: ${rect.height + 8}px;
        border: 3px solid ${color};
        border-radius: 8px;
        z-index: 99998;
        pointer-events: none;
        box-shadow: 0 0 0 4px ${color}40;
        animation: w2v-pulse 1s ease-in-out infinite;
      `;

      if (!document.getElementById('w2v-pulse-style')) {
        const style = document.createElement('style');
        style.id = 'w2v-pulse-style';
        style.textContent = `
          @keyframes w2v-pulse {
            0%, 100% { box-shadow: 0 0 0 4px ${color}40; }
            50% { box-shadow: 0 0 0 8px ${color}25; }
          }
        `;
        document.head.appendChild(style);
      }

      document.body.appendChild(highlight);

      if (lbl) {
        const labelEl = document.createElement('div');
        labelEl.id = 'w2v-highlight-label';
        labelEl.textContent = lbl;
        labelEl.style.cssText = `
          position: fixed;
          top: ${rect.y - 36}px; left: ${rect.x}px;
          z-index: 99999;
          background: ${color};
          color: #fff;
          font-family: 'Inter', sans-serif;
          font-size: 13px; font-weight: 600;
          padding: 4px 12px; border-radius: 4px;
          pointer-events: none;
        `;
        document.body.appendChild(labelEl);
      }

      setTimeout(() => {
        highlight.remove();
        document.getElementById('w2v-highlight-label')?.remove();
      }, dur);
    },
    { rect: box, lbl: label, dur: durationMs, color: theme.primaryColor },
  );

  await pause(page, durationMs + 200);
}

// ---------------------------------------------------------------------------
// Spotlight element — SVG mask dimming the rest of the page
// ---------------------------------------------------------------------------

/**
 * Spotlight an element by dimming everything else on the page.
 */
export async function spotlightElement(
  page: Page,
  selector: string,
  label?: string,
  durationMs: number = 3000,
): Promise<void> {
  const locator = page.locator(selector).first();
  const box = await locator.boundingBox().catch(() => null);
  if (!box) return;

  await safeEvaluate(
    page,
    ({ rect, lbl, dur }: {
      rect: { x: number; y: number; width: number; height: number };
      lbl?: string; dur: number;
    }) => {
      const svgNS = 'http://www.w3.org/2000/svg';
      const container = document.createElement('div');
      container.id = 'w2v-spotlight';
      container.style.cssText = `
        position: fixed; inset: 0; z-index: 99997;
        pointer-events: none; opacity: 0;
        transition: opacity 0.3s ease-in-out;
      `;

      const svg = document.createElementNS(svgNS, 'svg');
      svg.setAttribute('width', '100%');
      svg.setAttribute('height', '100%');
      svg.style.cssText = 'position: absolute; inset: 0;';

      const defs = document.createElementNS(svgNS, 'defs');
      const mask = document.createElementNS(svgNS, 'mask');
      mask.id = 'w2v-spotlight-mask';

      const bg = document.createElementNS(svgNS, 'rect');
      bg.setAttribute('width', '100%');
      bg.setAttribute('height', '100%');
      bg.setAttribute('fill', 'white');

      const cutout = document.createElementNS(svgNS, 'rect');
      cutout.setAttribute('x', String(rect.x - 8));
      cutout.setAttribute('y', String(rect.y - 8));
      cutout.setAttribute('width', String(rect.width + 16));
      cutout.setAttribute('height', String(rect.height + 16));
      cutout.setAttribute('rx', '8');
      cutout.setAttribute('fill', 'black');

      mask.appendChild(bg);
      mask.appendChild(cutout);
      defs.appendChild(mask);
      svg.appendChild(defs);

      const overlay = document.createElementNS(svgNS, 'rect');
      overlay.setAttribute('width', '100%');
      overlay.setAttribute('height', '100%');
      overlay.setAttribute('fill', 'rgba(0,0,0,0.6)');
      overlay.setAttribute('mask', 'url(#w2v-spotlight-mask)');
      svg.appendChild(overlay);

      container.appendChild(svg);

      if (lbl) {
        const labelEl = document.createElement('div');
        labelEl.textContent = lbl;
        labelEl.style.cssText = `
          position: fixed;
          top: ${rect.y + rect.height + 16}px;
          left: ${rect.x}px;
          z-index: 99999;
          color: #f8fafc;
          font-family: 'Inter', sans-serif;
          font-size: 16px; font-weight: 600;
          background: rgba(0,0,0,0.7);
          padding: 6px 16px; border-radius: 6px;
          pointer-events: none;
        `;
        container.appendChild(labelEl);
      }

      document.body.appendChild(container);
      requestAnimationFrame(() => { container.style.opacity = '1'; });

      setTimeout(() => {
        container.style.opacity = '0';
        setTimeout(() => container.remove(), 350);
      }, dur);
    },
    { rect: box, lbl: label, dur: durationMs },
  );

  await pause(page, durationMs + 400);
}

// ---------------------------------------------------------------------------
// Key combo badge
// ---------------------------------------------------------------------------

/** @deprecated No-op — text overlays removed. Use slide_service instead. */
export async function showKeyCombo(
  _page: Page,
  _keys: string,
  _opts?: { position?: Position; duration?: number },
): Promise<void> {}

// ---------------------------------------------------------------------------
// Step indicator — progress badge
// ---------------------------------------------------------------------------

/** @deprecated No-op — text overlays removed. Use slide_service instead. */
export async function showStepIndicator(
  _page: Page,
  _step: number,
  _total: number,
  _position: Position = 'top-right',
  _durationMs: number = 0,
): Promise<void> {}

// ---------------------------------------------------------------------------
// slowType — character-by-character typing
// ---------------------------------------------------------------------------

/**
 * Type text one character at a time with configurable delay.
 */
export async function slowType(
  page: Page,
  locator: Locator,
  text: string,
  opts?: { delay?: number },
): Promise<void> {
  const delay = opts?.delay ?? 80;
  await locator.pressSequentially(text, { delay });
}

// ---------------------------------------------------------------------------
// slowClick — deliberate approach, pause, click
// ---------------------------------------------------------------------------

/**
 * Slowly move to an element, pause, then click for demo visibility.
 */
export async function slowClick(page: Page, selector: string): Promise<void> {
  const locator = page.locator(selector).first();
  const box = await locator.boundingBox().catch(() => null);
  if (!box) return;

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 15 });
  await pause(page, 400);
  await locator.click();
  await pause(page, 300);
}

// ---------------------------------------------------------------------------
// slowHover — gradual mouse approach
// ---------------------------------------------------------------------------

/**
 * Gradually move mouse to an element for demo visibility.
 */
export async function slowHover(page: Page, selector: string): Promise<void> {
  const locator = page.locator(selector).first();
  const box = await locator.boundingBox().catch(() => null);
  if (!box) return;

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 20 });
  await pause(page, 600);
}

// ---------------------------------------------------------------------------
// smoothScroll — eased scroll animation
// ---------------------------------------------------------------------------

/**
 * Smoothly scroll the page with eased animation.
 */
export async function smoothScroll(
  page: Page,
  opts?: { direction?: 'up' | 'down'; distance?: number; duration?: number },
): Promise<void> {
  const direction = opts?.direction ?? 'down';
  const distance = opts?.distance ?? 500;
  const duration = opts?.duration ?? 1500;

  await safeEvaluate(
    page,
    ({ dir, dist, dur }: { dir: string; dist: number; dur: number }) => {
      return new Promise<void>((resolve) => {
        const sign = dir === 'down' ? 1 : -1;
        const start = performance.now();
        const startY = window.scrollY;
        function step(now: number) {
          const elapsed = now - start;
          const progress = Math.min(elapsed / dur, 1);
          const ease = progress < 0.5
            ? 4 * progress * progress * progress
            : 1 - Math.pow(-2 * progress + 2, 3) / 2;
          window.scrollTo(0, startY + sign * dist * ease);
          if (progress < 1) {
            requestAnimationFrame(step);
          } else {
            resolve();
          }
        }
        requestAnimationFrame(step);
      });
    },
    { dir: direction, dist: distance, dur: duration },
  );
  await pause(page, 200);
}

// ---------------------------------------------------------------------------
// Cursor trail — visible cursor dot with fade trail
// ---------------------------------------------------------------------------

function getCursorTrailScript(color: string): string {
  return `
    if (document.getElementById('w2v-cursor-style')) return;
    const style = document.createElement('style');
    style.id = 'w2v-cursor-style';
    style.textContent = \`
      .w2v-cursor-dot {
        position: fixed;
        width: 20px; height: 20px;
        border-radius: 50%;
        background: ${color}80;
        border: 2px solid ${color}CC;
        pointer-events: none;
        z-index: 100001;
        transform: translate(-50%, -50%);
        transition: opacity 0.15s;
      }
    \`;
    document.head.appendChild(style);
    const dot = document.createElement('div');
    dot.className = 'w2v-cursor-dot';
    dot.id = 'w2v-cursor-dot';
    document.body.appendChild(dot);
    document.addEventListener('mousemove', (e) => {
      const d = document.getElementById('w2v-cursor-dot');
      if (d) {
        d.style.left = e.clientX + 'px';
        d.style.top = e.clientY + 'px';
      }
    });
  `;
}

// ---------------------------------------------------------------------------
// Hide distracting overlays (cookie banners, closed captions, chat widgets)
// ---------------------------------------------------------------------------

const HIDE_DISTRACTIONS_CSS = `
  /* Closed captions / subtitles */
  video::cue { opacity: 0 !important; }
  .ytp-caption-window-container,
  .ytp-caption-window-bottom,
  [class*="closed-caption"],
  [class*="closedcaption"],
  [class*="caption-window"],
  [class*="captions-text"],
  [class*="subtitle-overlay"],
  [class*="vjs-text-track"],
  [data-testid="caption"],
  [data-testid="closedCaptions"],
  .jw-captions,
  .plyr__captions,
  .vjs-text-track-display,
  /* Cookie consent banners */
  .cc-window,
  .cc-banner,
  .cc-revoke,
  [class*="cookie-consent"],
  [class*="cookie-banner"],
  [class*="cookieConsent"],
  [class*="CookieBanner"],
  [id*="cookie-banner"],
  [id*="cookie-consent"],
  [id*="onetrust"],
  .onetrust-consent-sdk,
  #CybotCookiebotDialog,
  /* Chat widgets */
  [class*="intercom-"],
  #hubspot-messages-iframe-container,
  .drift-widget,
  [class*="crisp-client"],
  [id*="tidio"],
  [class*="zE-widget"] {
    display: none !important;
    visibility: hidden !important;
  }
`;

/**
 * Inject CSS that hides closed captions, cookie banners, and chat widgets.
 * Safe to call multiple times — skips if already injected.
 */
export async function hideDistractions(page: Page): Promise<void> {
  try {
    await page.addStyleTag({ content: HIDE_DISTRACTIONS_CSS });
  } catch {
    // Context may be destroyed during navigation — ignore
  }
}

/**
 * Enable a visible cursor trail for demo recordings.
 * Re-injects automatically after navigation.
 * Also injects CSS to hide distracting overlays (CC, cookie banners, chat widgets).
 */
export async function enableCursorTrail(page: Page): Promise<void> {
  const injectScript = getCursorTrailScript(currentTheme.primaryColor);
  await safeEvaluate(page, new Function(injectScript) as () => void);
  await hideDistractions(page);

  page.on('framenavigated', async (frame) => {
    if (frame === page.mainFrame()) {
      try {
        await frame.evaluate(new Function(injectScript) as () => void);
        await page.addStyleTag({ content: HIDE_DISTRACTIONS_CSS });
      } catch {
        // Navigation race — ignore
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `${r}, ${g}, ${b}`;
}
