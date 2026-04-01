/**
 * HoloSplat Viewer — orchestrates load → sort → render.
 *
 * Usage:
 *   const viewer = new Viewer(options);
 *   await viewer.init();
 *   await viewer.load(url);
 *   viewer.start();
 */
import { Renderer }   from './renderer.js';
import { OrbitCamera } from './camera.js';
import { createSorter } from './sorter.js';
import { loadSplat }  from './loaders/splat-loader.js';
import { loadPly }    from './loaders/ply-loader.js';
import { loadSpz }    from './loaders/spz-loader.js';

export class Viewer {
  constructor(options = {}) {
    const {
      canvas,
      background = '#000000',
      fov        = 60,
      near       = 0.1,
      far        = 2000,
      splatScale = 1.0,
      autoRotate = false,
      onProgress,
      onError,
    } = options;

    this._canvas     = resolveCanvas(canvas);
    this._onProgress = onProgress;
    this._onError    = onError;
    this._autoRotate = autoRotate;
    this._splatScale = splatScale;

    this._renderer   = new Renderer(this._canvas, background);
    this._camera     = new OrbitCamera({ fov, near, far });

    this._gaussians  = null;  // Float32Array after load
    this._numSplats  = 0;
    this._depths     = null;  // pre-allocated Float32Array
    this._sort       = null;  // sorter function
    this._rafId      = null;
    this._running    = false;
    this._resizeObs  = null;
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  async init() {
    await this._renderer.init();
    this._renderer.setSplatScale(this._splatScale);
    this._camera.attach(this._canvas);
    this._observeResize();
    this._updateSize();
  }

  async load(url) {
    // Errors propagate to the caller (create() handles onError routing)
    const ext    = url.split('?')[0].split('.').pop().toLowerCase();
    const loaders = { ply: loadPly, spz: loadSpz };
    const loader  = loaders[ext] ?? loadSplat;

    const { data, count } = await loader(url, p => {
      if (this._onProgress) this._onProgress(p);
    });

    this._gaussians = data;
    this._numSplats = count;
    this._depths    = new Float32Array(count);
    this._sort      = createSorter(count);

    this._renderer.uploadGaussians(data, count);
    this._camera.fitScene(data, count);
  }

  start() {
    if (this._running) return;
    this._running = true;
    this._tick();
  }

  stop() {
    this._running = false;
    if (this._rafId) cancelAnimationFrame(this._rafId);
    this._rafId = null;
  }

  destroy() {
    this.stop();
    this._camera.detach();
    this._renderer.destroy();
    this._resizeObs?.disconnect();
  }

  // ── Configuration setters (can be called at runtime) ──────────────────────

  setBackground(bg) { this._renderer.setBackground(bg); }

  setSplatScale(s) {
    this._splatScale = s;
    this._renderer.setSplatScale(s);
  }

  setAutoRotate(v) { this._autoRotate = v; }

  resetCamera() { this._camera.fitScene(this._gaussians, this._numSplats); }

  get camera() { return this._camera; }

  // ── Render loop ────────────────────────────────────────────────────────────

  _tick() {
    if (!this._running) return;
    this._rafId = requestAnimationFrame(() => this._tick());

    if (!this._numSplats) return;

    if (this._autoRotate) this._camera.theta += 0.005;

    const w   = this._canvas.width;
    const h   = this._canvas.height;
    this._camera.update(w, h);

    const view  = this._camera.viewMatrix;
    const proj  = this._camera.projMatrix;
    const focal = this._camera.focalLength(h);

    // Compute per-Gaussian depth (z in view space)
    this._computeDepths(view);

    // Sort back-to-front
    const order = this._sort(this._depths, this._numSplats);

    // Upload uniforms and order, then draw
    this._renderer.updateUniforms({ view, proj, width: w, height: h, focal });
    this._renderer.updateOrder(order, this._numSplats);
    this._renderer.draw();
  }

  _computeDepths(view) {
    // view is column-major Float32Array(16)
    // z_view = view[2]*x + view[6]*y + view[10]*z + view[14]
    const v0 = view[2], v1 = view[6], v2 = view[10], v3 = view[14];
    const gs  = this._gaussians;
    const dep = this._depths;
    const N   = this._numSplats;
    for (let i = 0; i < N; i++) {
      const j = i * 16;
      dep[i] = v0 * gs[j] + v1 * gs[j + 1] + v2 * gs[j + 2] + v3;
    }
  }

  // ── Resize handling ────────────────────────────────────────────────────────

  _observeResize() {
    if (typeof ResizeObserver === 'undefined') return;
    this._resizeObs = new ResizeObserver(() => this._updateSize());
    this._resizeObs.observe(this._canvas);
  }

  _updateSize() {
    const dpr = window.devicePixelRatio || 1;
    const w   = Math.round(this._canvas.clientWidth  * dpr);
    const h   = Math.round(this._canvas.clientHeight * dpr);
    if (w && h && (this._canvas.width !== w || this._canvas.height !== h)) {
      this._canvas.width  = w;
      this._canvas.height = h;
    }
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────

function resolveCanvas(canvas) {
  if (!canvas) throw new Error('HoloSplat: canvas option is required');
  if (typeof canvas === 'string') {
    const el = document.querySelector(canvas);
    if (!el) throw new Error(`HoloSplat: canvas selector "${canvas}" not found`);
    return el;
  }
  return canvas;
}
