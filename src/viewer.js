/**
 * HoloSplat Viewer — orchestrates load → sort → render.
 *
 * Usage:
 *   const viewer = new Viewer(options);
 *   await viewer.init();
 *   await viewer.load(url);
 *   viewer.start();
 */
import { Renderer }    from './renderer.js';
import { OrbitCamera }  from './camera.js';
import { createSorter } from './sorter.js';
import { loadSplat }                              from './loaders/splat-loader.js';
import { loadPly, openPlyStream } from './loaders/ply-loader.js';
import { loadSpz }                                from './loaders/spz-loader.js';
import { loadAnimation } from './animation.js';

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
      flipY      = false,
      onProgress,
      onError,
    } = options;

    this._canvas     = resolveCanvas(canvas);
    this._onProgress = onProgress;
    this._onError    = onError;
    this._autoRotate = autoRotate;
    this._splatScale = splatScale;
    this._flipY      = flipY;

    this._renderer   = new Renderer(this._canvas, background);
    this._camera     = new OrbitCamera({ fov, near, far });

    this._gaussians  = null;  // Float32Array after load
    this._numSplats  = 0;
    this._depths     = null;  // pre-allocated Float32Array
    this._sort       = null;  // sorter function
    this._rafId      = null;
    this._running    = false;
    this._resizeObs  = null;

    // Multi-part support
    this._partIndex     = {};                        // id → index
    this._partTransforms = [IDENTITY_MAT4.slice()]; // Float32Array[] one per part
    this._partTransFlat  = IDENTITY_MAT4.slice();   // flattened, written to GPU each frame

    this._animation  = null;  // Animation instance, or null
    this._animPaused = false; // true → animation frozen, camera responds to user input
    this._cameraFree  = false; // true → animation does not override camera (freecamera mode)
    this._sceneReady  = false; // true → all splat data is on GPU, animation may tick
    this._lastTick    = null;  // performance.now() of previous tick (for dt)
    this.onFrame     = null;  // callback(view, proj, width, height) — called each tick
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
    this._sceneReady = false;

    const ext = url.split('?')[0].split('.').pop().toLowerCase();
    if (ext === 'ply') {
      try {
        await this._loadPlyStreamSingle(url);
        return;
      } catch (e) {
        if (!/^HTTP 4/.test(e.message)) throw e;
        // 404 → fall through to format-agnostic loadUrl
      }
    }

    const { data, count } = await loadUrl(url, p => this._onProgress?.(p));
    if (this._flipY) flipYInPlace(data, count);
    for (let i = 0; i < count; i++) data[i * 16 + 3] = 0;
    this._gaussians      = data;
    this._numSplats      = count;
    this._depths         = new Float32Array(count);
    this._sort           = createSorter(count);
    this._partIndex      = {};
    this._partTransforms = [IDENTITY_MAT4.slice()];
    this._partTransFlat  = IDENTITY_MAT4.slice();
    this._renderer.uploadGaussians(data, count);
    this._renderer.uploadTransforms(this._partTransforms);
    this._camera.fitScene(data, count);
    if (this._animation) {
      const { eye, target } = this._animation.getCameraFrame();
      this._camera.setFromLookAt(eye, this._animation.focalPoint ?? target);
    }
    this._sceneReady = true;
  }

  async _loadPlyStreamSingle(url) {
    const { numVertices, consume } = await openPlyStream(url, p => this._onProgress?.(p * 0.02));

    const zeros = new Float32Array(numVertices * 16);
    this._gaussians      = zeros;
    this._numSplats      = numVertices;
    this._depths         = new Float32Array(numVertices);
    this._sort           = createSorter(numVertices);
    this._partIndex      = {};
    this._partTransforms = [IDENTITY_MAT4.slice()];
    this._partTransFlat  = IDENTITY_MAT4.slice();
    this._renderer.uploadGaussians(zeros, numVertices);
    this._renderer.uploadTransforms(this._partTransforms);
    if (this._animation) {
      const { eye, target } = this._animation.getCameraFrame();
      this._camera.setFromLookAt(eye, this._animation.focalPoint ?? target);
    }

    let vOff = 0;
    await consume((chunk, nVerts) => {
      for (let j = 0; j < nVerts; j++) chunk[j * 16 + 3] = 0;
      if (this._flipY) flipYInPlace(chunk, nVerts);
      this._renderer.patchGaussians(chunk, vOff);
      this._gaussians.set(chunk, vOff * 16);
      vOff += nVerts;
    }, p => this._onProgress?.(p));

    if (!this._animation) this._camera.fitScene(this._gaussians, this._numSplats);
    this._sceneReady = true;
  }

  /**
   * Load multiple splat parts and merge them into a single GPU scene.
   *
   * PLY parts are streamed: the header is parsed first (getting the vertex count),
   * the GPU buffer is allocated with zeros immediately so rendering starts as soon
   * as possible, then vertex data is patched in as it downloads.
   * Non-PLY parts (spz, splat) are loaded fully before the GPU buffer is allocated.
   *
   * _sceneReady becomes true only after all streams are exhausted, at which
   * point the animation begins ticking from frame 0.
   */
  async loadParts(partsMap) {
    const partIds = Object.keys(partsMap);
    if (partIds.length === 0) throw new Error('HoloSplat: loadParts called with empty map');

    this._sceneReady = false;

    const progresses = new Array(partIds.length).fill(0);
    const reportProg = () => {
      if (this._onProgress)
        this._onProgress(progresses.reduce((a, b) => a + b, 0) / partIds.length);
    };

    // Phase 1 — open PLY streams (header only) and fully load non-PLY parts.
    // All happen in parallel; every part resolves with its vertex count.
    const parts = await Promise.all(partIds.map(async (id, idx) => {
      const url = partsMap[id];
      const ext = url.split('?')[0].split('.').pop().toLowerCase();
      if (ext === 'ply') {
        try {
          const { numVertices, consume } = await openPlyStream(url, p => {
            progresses[idx] = p * 0.03; reportProg();
          });
          return { id, idx, kind: 'stream', numVertices, consume };
        } catch (e) {
          if (!/^HTTP 4/.test(e.message)) throw e;
          // 404 — let loadUrl try other extensions
        }
      }
      const { data, count } = await loadUrl(url, p => {
        progresses[idx] = p * 0.9; reportProg();
      });
      progresses[idx] = 0.9;
      return { id, idx, kind: 'loaded', data, count };
    }));

    // Compute total vertex count and per-part offsets in the merged buffer.
    const counts     = parts.map(p => p.kind === 'stream' ? p.numVertices : p.count);
    const total      = counts.reduce((a, b) => a + b, 0);
    const vtxOffsets = [];
    { let off = 0; for (const c of counts) { vtxOffsets.push(off); off += c; } }

    // Build the merged CPU buffer.
    // Non-PLY parts: fill with real data. PLY stream parts: stay as zeros (transparent).
    const merged = new Float32Array(total * 16);
    for (const part of parts) {
      if (part.kind !== 'loaded') continue;
      const { data, count, idx } = part;
      for (let j = 0; j < count; j++) data[j * 16 + 3] = idx;
      if (this._flipY) flipYInPlace(data, count);
      merged.set(data, vtxOffsets[idx] * 16);
      progresses[idx] = 1;
    }

    // Initialise viewer state and upload the buffer (zeros for stream parts).
    this._partIndex = {};
    partIds.forEach((id, i) => { this._partIndex[id] = i; });
    this._partTransforms = counts.map(() => IDENTITY_MAT4.slice());
    this._partTransFlat  = new Float32Array(counts.length * 16);
    for (let i = 0; i < counts.length; i++) this._partTransFlat.set(IDENTITY_MAT4, i * 16);

    this._gaussians = merged;
    this._numSplats = total;
    this._depths    = new Float32Array(total);
    this._sort      = createSorter(total);

    this._renderer.uploadGaussians(merged, total);
    this._renderer.uploadTransforms(this._partTransforms);
    this._camera.fitScene(merged, total);
    if (this._animation) {
      const { eye, target } = this._animation.getCameraFrame();
      this._camera.setFromLookAt(eye, this._animation.focalPoint ?? target);
    }
    reportProg();

    // Phase 2 — stream PLY vertex data, patching the GPU buffer as chunks arrive.
    const streamParts = parts.filter(p => p.kind === 'stream');
    if (streamParts.length === 0) {
      this._sceneReady = true;
      return;
    }

    await Promise.all(streamParts.map(async (part) => {
      const { idx, consume } = part;
      let vOff = vtxOffsets[idx];

      await consume((chunk, nVerts) => {
        for (let j = 0; j < nVerts; j++) chunk[j * 16 + 3] = idx;
        if (this._flipY) flipYInPlace(chunk, nVerts);
        this._renderer.patchGaussians(chunk, vOff);
        this._gaussians.set(chunk, vOff * 16);
        vOff += nVerts;
      }, p => {
        progresses[idx] = 0.03 + 0.97 * p;
        reportProg();
      });

      progresses[idx] = 1;
      reportProg();
    }));

    if (!this._animation) this._camera.fitScene(this._gaussians, this._numSplats);
    this._sceneReady = true;
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

  setFlipY(enabled) {
    if (!!enabled === this._flipY) return;
    this._flipY = !!enabled;
    if (this._gaussians) {
      flipYInPlace(this._gaussians, this._numSplats);
      this._renderer.uploadGaussians(this._gaussians, this._numSplats);
      this._camera.fitScene(this._gaussians, this._numSplats);
    }
  }


  /** Freeze / unfreeze animation playback. When paused, camera responds to user input. */
  setAnimationPaused(paused) { this._animPaused = paused; }

  /** When true, animation no longer overrides the camera — orbit/zoom controls take over. */
  setCameraFree(v) { this._cameraFree = !!v; }

  /**
   * Read the current animation frame and set _cameraFree based on the most
   * recently passed hs-* marker (hs-free → free, hs-locked → locked).
   * hs-h{N} / hs-v{N} enable free orbit with angular constraints ±N degrees
   * from the entry angle. Markers can be combined: hs-h30-v20.
   * Only runs when the animation is self-playing (!_animPaused) and the
   * animation contains at least one hs-* marker.
   */
  _syncCameraMode() {
    const markers = this._animation.markers;
    const frame   = this._animation.frame;

    let activeName = null;
    let maxFrame   = -1;
    let hasHs      = false;

    for (const name of Object.keys(markers)) {
      if (!name.startsWith('hs-')) continue;
      hasHs = true;
      const mf = markers[name];
      if (mf <= frame && mf > maxFrame) { maxFrame = mf; activeName = name; }
    }

    if (!hasHs) return;

    const shouldBeFree = activeName !== null && activeName !== 'hs-locked';
    if (shouldBeFree === this._cameraFree) return;

    this._cameraFree = shouldBeFree;

    if (shouldBeFree) {
      // Position orbit camera at the animation eye, locked onto the focal point
      // (or the animation's look-at target if no focal point is set).
      const { eye, target } = this._animation.getCameraFrame();
      const fp = this._animation.focalPoint;
      this._camera.setFromLookAt(eye, fp ?? target);
      if (fp) this._camera.panEnabled = false;
      // Parse hs-h{N} / hs-v{N} angle restrictions from the marker name.
      const hMatch = activeName.match(/h(\d+)/);
      const vMatch = activeName.match(/v(\d+)/);
      this._camera.constrainAngles(
        hMatch ? parseInt(hMatch[1], 10) : null,
        vMatch ? parseInt(vMatch[1], 10) : null,
      );
    } else {
      if (this._animation.focalPoint) this._camera.panEnabled = true;
      this._camera.clearConstraints();
    }
  }

  resetCamera() { this._camera.fitScene(this._gaussians, this._numSplats); }
  focusCamera() { this._camera.focusScene(this._gaussians, this._numSplats); }

  /** Replace scene data (rebuilds sorter + GPU buffers). fitCamera=true re-fits the orbit camera. */
  setGaussians(data, count, fitCamera = false) {
    this._gaussians  = data;
    this._numSplats  = count;
    this._depths     = new Float32Array(count);
    this._sort       = createSorter(count);
    this._sceneReady = true;
    this._renderer.uploadGaussians(data, count);
    if (fitCamera) this._camera.fitScene(data, count);
  }

  /** Re-upload display data for the current frame without rebuilding the sorter (used for highlights). */
  uploadDisplay(data) {
    if (this._numSplats) this._renderer.uploadGaussians(data, this._numSplats);
  }

  // ── Animation ──────────────────────────────────────────────────────────────

  /** Attach a pre-loaded Animation instance. Pass null to detach. */
  setAnimation(anim) {
    this._animation = anim;
    if (!anim) return;
    if (anim.fov  != null) this._camera.fov  = anim.fov  * Math.PI / 180;
    if (anim.near != null) this._camera.near = anim.near;
    if (anim.far  != null) this._camera.far  = anim.far;
    // Apply frame 0 camera position immediately so the scene never flashes
    // the auto-fit angle while the user waits for assets to finish loading.
    const { eye, target } = anim.getCameraFrame();
    this._camera.setFromLookAt(eye, anim.focalPoint ?? target);
  }

  /** Fetch, parse, and attach an animation from a URL. */
  async loadAnimationUrl(url) {
    const anim = await loadAnimation(url);
    this.setAnimation(anim);
    return anim;
  }

  // ── Callout projection ─────────────────────────────────────────────────────

  /**
   * Project an array of { id, pos:[x,y,z] } callouts to screen coordinates.
   * Returns array of { id, visible, x, y } using the current view/proj matrices.
   * Call this after update() (i.e. inside onFrame or after a tick).
   */
  projectCallouts(callouts) {
    const view = this._camera.viewMatrix;
    const proj = this._camera.projMatrix;
    // Use CSS pixels (clientWidth/Height) so positions map directly to
    // element left/top without DPR scaling issues.
    const w    = this._canvas.clientWidth;
    const h    = this._canvas.clientHeight;
    const out  = [];

    for (const c of callouts) {
      const [px, py, pz] = c.pos;

      // Transform to view space (column-major matrix multiply)
      const vx = view[0]*px + view[4]*py + view[8]*pz  + view[12];
      const vy = view[1]*px + view[5]*py + view[9]*pz  + view[13];
      const vz = view[2]*px + view[6]*py + view[10]*pz + view[14];

      if (vz >= 0) { out.push({ id: c.id, visible: false, x: 0, y: 0 }); continue; }

      // Perspective divide using the projection matrix shortcut:
      // clip_x = proj[0]*vx,  clip_y = proj[5]*vy,  clip_w = -vz  (since proj[11] = -1)
      const cw = -vz;
      const sx = (proj[0] * vx / cw * 0.5 + 0.5) * w;
      const sy = (1 - (proj[5] * vy / cw * 0.5 + 0.5)) * h;

      out.push({ id: c.id, visible: true, x: sx, y: sy });
    }
    return out;
  }

  get camera() { return this._camera; }

  // ── Render loop ────────────────────────────────────────────────────────────

  _tick() {
    if (!this._running) return;
    this._rafId = requestAnimationFrame(() => this._tick());

    // Delta time (seconds since last tick)
    const now = performance.now();
    const dt  = this._lastTick ? Math.min((now - this._lastTick) / 1000, 0.1) : 0;
    this._lastTick = now;

    const w = this._canvas.width;
    const h = this._canvas.height;

    if (this._animation) {
      if (!this._animPaused) {
        if (this._sceneReady) this._animation.tick(dt); // don't advance until all data is on GPU
        this._syncCameraMode();
      }
      if (!this._cameraFree) {
        const { eye, target } = this._animation.getCameraFrame();
        this._camera.setFromLookAt(eye, target);
      }
      // Apply per-object transforms if the animation has them
      const objFrames = this._animation.getObjectFrames();
      if (objFrames.length > 0) {
        let dirty = false;
        for (const { id, pos, quat } of objFrames) {
          const idx = this._partIndex[id];
          if (idx !== undefined) {
            this._partTransFlat.set(quatPosToMat4(pos, quat), idx * 16);
            dirty = true;
          }
        }
        if (dirty) this._renderer.updateTransforms(this._partTransFlat);
      }
    } else if (this._autoRotate) {
      this._camera.theta += 0.005;
    }

    if (!this._numSplats) return;

    this._camera.update(w, h);

    // Notify listeners (e.g. player updating callout positions)
    if (this.onFrame) {
      this.onFrame(this._camera.viewMatrix, this._camera.projMatrix, w, h);
    }

    const view  = this._camera.viewMatrix;
    const proj  = this._camera.projMatrix;
    const focal = this._camera.focalLength(h);

    // Compute per-Gaussian depth (z in view space)
    this._computeDepths(view);

    // Sort back-to-front
    const order = this._sort(this._depths, this._numSplats);

    // Upload uniforms and order, then draw
    this._renderer.updateUniforms({ view, proj, width: w, height: h, focal, near: this._camera.near });
    this._renderer.updateOrder(order, this._numSplats);
    this._renderer.draw();
  }

  _computeDepths(view) {
    // Precompute per-part depth row: vz = a*lx + b*ly + c*lz + d
    // where (lx,ly,lz) is the splat position in part-local space.
    const v2 = view[2], v6 = view[6], v10 = view[10], v14 = view[14];
    const rows = [];
    const flat = this._partTransFlat;
    const nParts = this._partTransforms.length;
    for (let p = 0; p < nParts; p++) {
      const o = p * 16;
      rows.push([
        v2*flat[o]   + v6*flat[o+1]  + v10*flat[o+2],
        v2*flat[o+4] + v6*flat[o+5]  + v10*flat[o+6],
        v2*flat[o+8] + v6*flat[o+9]  + v10*flat[o+10],
        v2*flat[o+12]+ v6*flat[o+13] + v10*flat[o+14] + v14,
      ]);
    }

    const gs  = this._gaussians;
    const dep = this._depths;
    const N   = this._numSplats;
    for (let i = 0; i < N; i++) {
      const j = i * 16;
      const r = rows[gs[j + 3]]; // part index stored as float 0.0, 1.0, …
      dep[i] = r[0] * gs[j] + r[1] * gs[j + 1] + r[2] * gs[j + 2] + r[3];
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

// ── Constants ─────────────────────────────────────────────────────────────

const IDENTITY_MAT4 = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);

// ── Helpers ────────────────────────────────────────────────────────────────

function flipYInPlace(data, count) {
  // Applies a 180° rotation around the X axis to every Gaussian in canonical layout.
  // Converts Y-down scenes (OpenCV/COLMAP convention) to Y-up (OpenGL/HoloSplat convention).
  // Position: (x, y, z) → (x, -y, -z)
  // Quaternion pre-multiply by (1,0,0,0): (qx,qy,qz,qw) → (qw,-qz,qy,-qx)
  for (let i = 0; i < count; i++) {
    const d = i * 16;
    data[d + 1] = -data[d + 1];
    data[d + 2] = -data[d + 2];
    const qx = data[d + 12], qy = data[d + 13], qz = data[d + 14], qw = data[d + 15];
    data[d + 12] =  qw;
    data[d + 13] = -qz;
    data[d + 14] =  qy;
    data[d + 15] = -qx;
  }
}

const SPLAT_EXTS    = ['spz', 'ply', 'splat'];
const SPLAT_LOADERS = { ply: loadPly, spz: loadSpz }; // splat is the default

/**
 * Load a splat file, auto-detecting format from the extension.
 * If the file returns HTTP 4xx, tries the other known extensions in order
 * (.spz → .ply → .splat) so renaming or re-encoding a file doesn't break callers.
 * Non-4xx errors (network failures, 5xx) are rethrown immediately.
 */
async function loadUrl(url, onProgress) {
  const clean   = url.split('?')[0];
  const lastDot = clean.lastIndexOf('.');
  const rawExt  = lastDot >= 0 ? clean.slice(lastDot + 1).toLowerCase() : '';
  const hasExt  = SPLAT_EXTS.includes(rawExt);

  // Try the given extension first, then the remaining formats as fallbacks.
  const exts = hasExt ? [rawExt, ...SPLAT_EXTS.filter(e => e !== rawExt)] : SPLAT_EXTS;
  const base  = hasExt ? url.slice(0, url.lastIndexOf('.')) : url;

  let lastErr;
  for (const ext of exts) {
    const candidate = `${base}.${ext}`;
    const loader    = SPLAT_LOADERS[ext] ?? loadSplat;
    try {
      return await loader(candidate, onProgress);
    } catch (err) {
      if (!/^HTTP 4/.test(err.message)) throw err; // non-4xx → give up immediately
      lastErr = err;
    }
  }
  throw new Error(`HoloSplat: splat file not found as .spz / .ply / .splat — "${base}"`);
}

/** Build a column-major mat4 from pos=[x,y,z] and quat=[x,y,z,w]. */
function quatPosToMat4(pos, quat) {
  const [qx, qy, qz, qw] = quat;
  const [px, py, pz]      = pos;
  const x2 = qx*2, y2 = qy*2, z2 = qz*2;
  const xx = qx*x2, xy = qx*y2, xz = qx*z2;
  const yy = qy*y2, yz = qy*z2, zz = qz*z2;
  const wx = qw*x2, wy = qw*y2, wz = qw*z2;
  return new Float32Array([
    1-yy-zz,  xy+wz,   xz-wy,  0,
    xy-wz,  1-xx-zz,   yz+wx,  0,
    xz+wy,   yz-wx,  1-xx-yy,  0,
    px, py, pz, 1,
  ]);
}

function resolveCanvas(canvas) {
  if (!canvas) throw new Error('HoloSplat: canvas option is required');
  if (typeof canvas === 'string') {
    const el = document.querySelector(canvas);
    if (!el) throw new Error(`HoloSplat: canvas selector "${canvas}" not found`);
    return el;
  }
  return canvas;
}
