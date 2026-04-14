/**
 * HoloSplat Player — embeddable Gaussian splat player for any website.
 *
 * Accepts any container element, creates its own canvas inside it, and handles
 * the full lifecycle: WebGPU init, loading spinner, progress bar, error display,
 * and responsive resize.
 *
 * ─── Script-tag usage ───────────────────────────────────────────────────────
 *
 *   <script src="holosplat.iife.js"></script>
 *
 *   <div id="scene" style="width:100%; height:500px"></div>
 *   <script>
 *     HoloSplat.player('#scene', {
 *       src: 'https://cdn.example.com/scene.spz',
 *     });
 *   </script>
 *
 * ─── Data-attribute auto-init (no JS needed) ────────────────────────────────
 *
 *   <div data-holosplat="https://cdn.example.com/scene.spz"
 *        style="width:100%; height:500px"></div>
 *
 *   All [data-holosplat] elements are initialised automatically when the
 *   script loads (or on DOMContentLoaded if the script is in <head>).
 *
 * ─── Returned API ───────────────────────────────────────────────────────────
 *
 *   load(url)           – load a new scene into the same player
 *   destroy()           – stop rendering and remove all created DOM
 *   setBackground(bg)   – '#rrggbb' | '#rrggbbaa' | 'transparent' | [r,g,b,a]
 *   setSplatScale(n)    – multiplier applied to all splat sizes
 *   setAutoRotate(bool) – toggle slow orbit rotation
 *   resetCamera()       – fit camera back to the loaded scene
 *   camera              – OrbitCamera instance for direct manipulation
 */

import { Viewer } from './viewer.js';

// ── CSS (injected once per page) ──────────────────────────────────────────────

const PLAYER_CSS = `
.hs-player{position:relative;overflow:hidden;}
.hs-player canvas{position:absolute;inset:0;width:100%;height:100%;display:block;}
.hs-player .hs-overlay{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:10px;
  pointer-events:none;font-family:system-ui,sans-serif;
}
.hs-player .hs-spinner{
  width:24px;height:24px;border-radius:50%;
  border:2px solid rgba(255,255,255,.15);border-top-color:#3a7aff;
  animation:hs-spin .7s linear infinite;
}
@keyframes hs-spin{to{transform:rotate(360deg);}}
.hs-player .hs-bar-wrap{width:140px;height:3px;background:rgba(255,255,255,.1);border-radius:2px;overflow:hidden;}
.hs-player .hs-bar{height:100%;background:#3a7aff;width:0%;transition:width .1s;}
.hs-player .hs-msg{font-size:.78rem;color:rgba(255,255,255,.45);text-align:center;max-width:260px;line-height:1.5;padding:0 16px;}
.hs-player .hs-msg.hs-err{color:#f87;}
`;

let cssInjected = false;
function injectCss() {
  if (cssInjected || typeof document === 'undefined') return;
  cssInjected = true;
  const s = document.createElement('style');
  s.textContent = PLAYER_CSS;
  document.head.appendChild(s);
}

// ── player() ─────────────────────────────────────────────────────────────────

/**
 * @param {string|HTMLElement} container  CSS selector or DOM element
 * @param {object}  [opts]
 * @param {string}  [opts.src]             URL to load immediately (.splat / .ply / .spz)
 * @param {string|number[]} [opts.background='transparent']
 * @param {number}  [opts.fov=60]
 * @param {number}  [opts.near=0.1]
 * @param {number}  [opts.far=2000]
 * @param {number}  [opts.splatScale=1]
 * @param {boolean} [opts.autoRotate=false]
 * @param {function} [opts.onLoad]
 * @param {function} [opts.onProgress]
 * @param {function} [opts.onError]
 */
export function player(container, opts = {}) {
  injectCss();

  // ── Resolve container ───────────────────────────────────────────────────────
  const root = typeof container === 'string'
    ? document.querySelector(container)
    : container;
  if (!root) throw new Error(`HoloSplat: container not found — "${container}"`);

  const {
    src,
    background  = 'transparent',
    fov         = 60,
    near        = 0.1,
    far         = 2000,
    splatScale  = 1,
    autoRotate  = false,
    onLoad, onProgress, onError,
  } = opts;

  // ── Build DOM ───────────────────────────────────────────────────────────────
  root.classList.add('hs-player');

  const canvas  = document.createElement('canvas');
  const overlay = document.createElement('div');
  overlay.className = 'hs-overlay';
  overlay.innerHTML =
    '<div class="hs-spinner"></div>' +
    '<div class="hs-bar-wrap"><div class="hs-bar"></div></div>' +
    '<div class="hs-msg"></div>';
  root.appendChild(canvas);
  root.appendChild(overlay);

  const spinner = overlay.querySelector('.hs-spinner');
  const barWrap = overlay.querySelector('.hs-bar-wrap');
  const bar     = overlay.querySelector('.hs-bar');
  const msgEl   = overlay.querySelector('.hs-msg');

  function showLoading() {
    spinner.style.display = '';
    barWrap.style.display = '';
    msgEl.textContent = '';
    msgEl.className = 'hs-msg';
    overlay.style.display = 'flex';
  }
  function showReady() {
    overlay.style.display = 'none';
  }
  function showError(text) {
    spinner.style.display = 'none';
    barWrap.style.display = 'none';
    msgEl.textContent = text;
    msgEl.className = 'hs-msg hs-err';
    overlay.style.display = 'flex';
  }

  overlay.style.display = 'none'; // hidden until first load

  // ── Viewer ──────────────────────────────────────────────────────────────────
  const viewer = new Viewer({
    canvas,
    background,
    fov, near, far,
    splatScale,
    autoRotate,
    onProgress: p => {
      bar.style.width = `${(p * 100).toFixed(0)}%`;
      if (onProgress) onProgress(p);
    },
  });

  // ── Load ────────────────────────────────────────────────────────────────────
  async function load(url) {
    showLoading();
    bar.style.width = '0%';
    try {
      await viewer.load(url);
      showReady();
      if (onLoad) onLoad();
    } catch (err) {
      const msg = navigator.gpu
        ? err.message
        : 'WebGPU not supported. Use Chrome 113+ or Edge 113+.';
      showError(msg);
      if (onError) onError(err);
    }
  }

  // ── Boot ────────────────────────────────────────────────────────────────────
  viewer.init()
    .then(() => {
      viewer.start();
      if (src) return load(src);
    })
    .catch(err => {
      const msg = navigator.gpu
        ? err.message
        : 'WebGPU not supported. Use Chrome 113+ or Edge 113+.';
      showError(msg);
      if (onError) onError(err);
    });

  // ── Public API ───────────────────────────────────────────────────────────────
  return {
    load,
    destroy() {
      viewer.destroy();
      root.innerHTML = '';
      root.classList.remove('hs-player');
    },
    setBackground(bg)  { viewer.setBackground(bg); },
    setSplatScale(s)   { viewer.setSplatScale(s); },
    setAutoRotate(v)   { viewer.setAutoRotate(v); },
    resetCamera()      { viewer.resetCamera(); },
    get camera()       { return viewer.camera; },
  };
}

// ── Data-attribute auto-init ─────────────────────────────────────────────────

function autoInit() {
  document.querySelectorAll('[data-holosplat]').forEach(el => {
    if (el._hsPlayer) return; // already initialised
    const src = el.getAttribute('data-holosplat');
    if (src) el._hsPlayer = player(el, { src });
  });
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInit);
  } else {
    autoInit();
  }
}
