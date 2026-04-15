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
 *   load(url)              – load a new scene into the same player
 *   loadAnim(url)          – load a Blender animation JSON (camera + callouts)
 *   destroy()              – stop rendering and remove all created DOM
 *   setBackground(bg)      – '#rrggbb' | '#rrggbbaa' | 'transparent' | [r,g,b,a]
 *   setSplatScale(n)       – multiplier applied to all splat sizes
 *   setAutoRotate(bool)    – toggle slow orbit rotation (disabled while animation plays)
 *   resetCamera()          – fit camera back to the loaded scene
 *   camera                 – OrbitCamera instance for direct manipulation
 *   animation              – Animation instance (after loadAnim), or null
 *   callout(id)            – returns the HTMLElement for a named callout div
 *
 * ─── Callout styling ────────────────────────────────────────────────────────
 *
 *   Each callout exported from Blender creates a <div class="hs-callout"
 *   data-id="name"> absolutely positioned over the canvas. The library only
 *   moves it — you provide the content and styles:
 *
 *   .hs-callout[data-id="screen"] { width: 120px; background: white; ... }
 *
 *   When the callout point goes behind the camera, the class hs-callout--hidden
 *   is added (display: none by default) and removed when it comes back into view.
 *
 * ─── Data attributes ────────────────────────────────────────────────────────
 *
 *   data-holosplat="url"         – scene file (auto-init)
 *   data-holosplat-anim="url"    – animation JSON (auto-init alongside scene)
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
.hs-player .hs-callouts{position:absolute;inset:0;pointer-events:none;}
.hs-callout{position:absolute;pointer-events:auto;transform:translate(-50%,-50%);}
.hs-callout--hidden{display:none;}
.hs-callout::after{content:'';display:block;width:10px;height:10px;background:#3a7aff;border:2px solid #fff;border-radius:50%;box-shadow:0 1px 4px rgba(0,0,0,.5);}
.hs-callout:not(:empty)::after{display:none;}
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
    animation:  animSrc,
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

  const canvas    = document.createElement('canvas');
  const calloutEl = document.createElement('div');
  calloutEl.className = 'hs-callouts';
  const overlay = document.createElement('div');
  overlay.className = 'hs-overlay';
  overlay.innerHTML =
    '<div class="hs-spinner"></div>' +
    '<div class="hs-bar-wrap"><div class="hs-bar"></div></div>' +
    '<div class="hs-msg"></div>';
  root.appendChild(canvas);
  root.appendChild(calloutEl);
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

  // ── Callout DOM ──────────────────────────────────────────────────────────────
  // Map of id → HTMLElement, built when animation loads
  const calloutDivs = {};

  function buildCallouts(callouts) {
    // Remove old callout divs
    calloutEl.innerHTML = '';
    for (const key of Object.keys(calloutDivs)) delete calloutDivs[key];

    for (const c of callouts) {
      const div = document.createElement('div');
      div.className = 'hs-callout';
      div.dataset.id = c.id;
      calloutEl.appendChild(div);
      calloutDivs[c.id] = div;
    }
  }

  // Update callout positions each frame via viewer.onFrame
  viewer.onFrame = (_view, _proj, _w, _h) => {
    if (!viewer._animation?.callouts.length) return;
    const projected = viewer.projectCallouts(viewer._animation.callouts);
    for (const { id, visible, x, y } of projected) {
      const div = calloutDivs[id];
      if (!div) continue;
      if (visible) {
        div.classList.remove('hs-callout--hidden');
        div.style.left = x + 'px';
        div.style.top  = y + 'px';
      } else {
        div.classList.add('hs-callout--hidden');
      }
    }
  };

  // ── Load scene ───────────────────────────────────────────────────────────────
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

  // ── Load animation ───────────────────────────────────────────────────────────
  async function loadAnim(url) {
    try {
      const anim = await viewer.loadAnimationUrl(url);
      buildCallouts(anim.callouts);
      console.log(
        `[HoloSplat] animation loaded: ${anim.frameCount} frames @ ${anim.fps}fps, ` +
        `${anim.callouts.length} callout(s):`, anim.callouts.map(c => c.id)
      );
      return anim;
    } catch (err) {
      console.error('[HoloSplat] animation failed to load:', err);
      if (onError) onError(err);
      // Don't rethrow — leave the scene visible, just no animation
    }
  }

  // ── Boot ────────────────────────────────────────────────────────────────────
  viewer.init()
    .then(() => {
      viewer.start();
      const loads = [];
      if (src)     loads.push(load(src));
      if (animSrc) loads.push(loadAnim(animSrc));
      return Promise.all(loads);
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
    loadAnim,
    destroy() {
      viewer.destroy();
      root.innerHTML = '';
      root.classList.remove('hs-player');
    },
    setBackground(bg)  { viewer.setBackground(bg); },
    setSplatScale(s)   { viewer.setSplatScale(s); },
    setAutoRotate(v)   { viewer.setAutoRotate(v); },
    resetCamera()      { viewer.resetCamera(); },
    callout(id)        { return calloutDivs[id] ?? null; },
    get camera()       { return viewer.camera; },
    get animation()    { return viewer._animation; },
  };
}

// ── Data-attribute auto-init ─────────────────────────────────────────────────

function autoInit() {
  document.querySelectorAll('[data-holosplat]').forEach(el => {
    if (el._hsPlayer) return;
    const src  = el.getAttribute('data-holosplat')      || undefined;
    const anim = el.getAttribute('data-holosplat-anim') || undefined;
    el._hsPlayer = player(el, { src, animation: anim });
  });
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInit);
  } else {
    autoInit();
  }
}
