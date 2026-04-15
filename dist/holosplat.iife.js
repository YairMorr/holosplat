/**
 * HoloSplat v0.1.0 – WebGPU Gaussian Splat viewer | MIT
 */
var HoloSplat = (function () {
  'use strict';

// ── src/shaders.js ────────────────────────────
// WGSL shader for Gaussian splatting.
//
// GPU layout (must match CPU Float32Array layout in loaders):
//
//   struct Gaussian {         byte offset
//     pos   : vec3<f32>,   //  0  (12 bytes + 4 implicit WGSL padding = 16)
//     color : vec4<f32>,   // 16  (rgba, premultiplied ready)
//     scale : vec3<f32>,   // 32  (12 bytes + 4 implicit padding = 16)
//     quat  : vec4<f32>,   // 48  (xyzw)
//   }; // stride = 64 bytes = 16 floats/gaussian
//
// Uniform layout (160 bytes = 40 floats):
//   [0-15]  view matrix (col-major)
//   [16-31] proj matrix (col-major)
//   [32-33] viewport (width, height) in pixels
//   [34-35] focal (fx, fy) in pixels
//   [36]    splatScale multiplier
//   [37-39] padding

const SHADER = /* wgsl */`

struct Uniforms {
  view       : mat4x4<f32>,
  proj       : mat4x4<f32>,
  viewport   : vec2<f32>,
  focal      : vec2<f32>,
  params     : vec4<f32>,  // .x = splatScale
};

struct Gaussian {
  pos   : vec3<f32>,
  color : vec4<f32>,
  scale : vec3<f32>,
  quat  : vec4<f32>,
};

@group(0) @binding(0) var<uniform>       uniforms  : Uniforms;
@group(0) @binding(1) var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(2) var<storage, read> order     : array<u32>;

struct VOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) color         : vec4<f32>,
  @location(1) uv            : vec2<f32>,
  @location(2) conic         : vec3<f32>,
};

// Quaternion (xyzw) → column-major rotation matrix
fn quatToMat3(q: vec4<f32>) -> mat3x3<f32> {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0*(y*y+z*z),  2.0*(x*y+w*z),       2.0*(x*z-w*y)),
    vec3<f32>(2.0*(x*y-w*z),         1.0 - 2.0*(x*x+z*z), 2.0*(y*z+w*x)),
    vec3<f32>(2.0*(x*z+w*y),         2.0*(y*z-w*x),       1.0 - 2.0*(x*x+y*y))
  );
}

// Degenerate vertex (moved outside clip space → triangle discarded)
fn degen() -> VOut {
  var o: VOut;
  o.clipPos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
  o.color   = vec4<f32>(0.0);
  o.uv      = vec2<f32>(0.0);
  o.conic   = vec3<f32>(0.0);
  return o;
}

@vertex
fn vs_main(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> VOut {
  // Quad corners (two triangles, CCW)
  const corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0)
  );

  let gi = order[ii];
  let g  = gaussians[gi];
  let corner = corners[vi];

  // ── Transform to view space ──────────────────────────────────────────────
  let viewPos4 = uniforms.view * vec4<f32>(g.pos, 1.0);
  let t = viewPos4.xyz;

  // Discard if behind near plane
  if t.z > -0.1 { return degen(); }

  // ── 3-D covariance ────────────────────────────────────────────────────────
  let splatScale = uniforms.params.x;
  let s = g.scale * splatScale;
  let R = quatToMat3(g.quat);
  // M = R * diag(s); Cov3D = M * Mᵀ
  let M    = mat3x3<f32>(R[0]*s.x, R[1]*s.y, R[2]*s.z);
  let cov3 = M * transpose(M);

  // ── Project covariance to 2-D ─────────────────────────────────────────────
  // Perspective Jacobian at view-space point t (camera looks down –Z)
  let tz2 = t.z * t.z;
  let fx  = uniforms.focal.x;
  let fy  = uniforms.focal.y;
  let J = mat3x3<f32>(
    vec3<f32>(fx / (-t.z),  0.0,           0.0),
    vec3<f32>(0.0,          fy / (-t.z),   0.0),
    vec3<f32>(fx*t.x / tz2, fy*t.y / tz2, 0.0)
  );
  // View rotation (upper-left 3×3 of view matrix, column-major)
  let W = mat3x3<f32>(
    uniforms.view[0].xyz,
    uniforms.view[1].xyz,
    uniforms.view[2].xyz
  );
  let T    = J * W;
  let cov2 = T * cov3 * transpose(T);

  // Extract 2×2 + low-pass filter (anti-aliasing)
  let a = cov2[0][0] + 0.3;   // Σ_xx  (cov2[col][row])
  let b = cov2[0][1];          // Σ_xy
  let c = cov2[1][1] + 0.3;   // Σ_yy

  // ── Inverse covariance (conic) ────────────────────────────────────────────
  let det = a*c - b*b;
  if det < 1e-4 { return degen(); }
  let inv = 1.0 / det;
  // conic = (A, B, C) such that Mahalanobis² = A·dx² + 2B·dx·dy + C·dy²
  let conic = vec3<f32>(c*inv, -b*inv, a*inv);

  // ── Bounding radius (3σ of largest eigenvalue) ────────────────────────────
  let mid    = 0.5 * (a + c);
  let disc   = sqrt(max(0.1, mid*mid - det));
  let radius = ceil(3.0 * sqrt(mid + disc));

  // ── Screen-space quad placement ───────────────────────────────────────────
  let clip    = uniforms.proj * viewPos4;
  let ndcXY   = clip.xy / clip.w;
  let pixOff  = corner * radius;
  let ndcOff  = pixOff / (uniforms.viewport * 0.5);

  var o: VOut;
  o.clipPos = vec4<f32>(ndcXY + ndcOff, clip.z / clip.w, 1.0);
  o.color   = g.color;
  o.uv      = pixOff;
  o.conic   = conic;
  return o;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let d = in.uv;
  let power = -0.5 * (in.conic.x*d.x*d.x + 2.0*in.conic.y*d.x*d.y + in.conic.z*d.y*d.y);
  if power > 0.0 { discard; }
  let alpha = in.color.a * exp(power);
  if alpha < 1.0/255.0 { discard; }
  // Premultiplied alpha output (blend: src=ONE, dst=ONE_MINUS_SRC_ALPHA)
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
`;

// ── src/camera.js ────────────────────────────
/**
 * Orbit camera with mouse and touch controls.
 * Spherical coordinates (theta = azimuth, phi = elevation) around a target point.
 *
 * View matrix is column-major Float32Array(16) compatible with WebGPU.
 */
class OrbitCamera {
  constructor({ fov = 60, near = 0.1, far = 2000 } = {}) {
    this.fov    = fov * Math.PI / 180;  // radians
    this.near   = near;
    this.far    = far;

    // Spherical state
    this.theta  = 0;           // azimuth
    this.phi    = 0.2;         // elevation (clamped away from poles)
    this.radius = 5;
    this.target = [0, 0, 0];

    this._drag    = null;      // { x, y, button }
    this._touches = [];

    this.viewMatrix = new Float32Array(16);
    this.projMatrix = new Float32Array(16);
  }

  // ── Attach / detach input listeners ────────────────────────────────────────

  attach(canvas) {
    this._canvas = canvas;
    this._onMouseDown  = e => this._mouseDown(e);
    this._onMouseMove  = e => this._mouseMove(e);
    this._onMouseUp    = () => { this._drag = null; };
    this._onWheel      = e => this._wheel(e);
    this._onTouchStart = e => this._touchStart(e);
    this._onTouchMove  = e => this._touchMove(e);
    this._onTouchEnd   = e => this._touchEnd(e);
    this._onCtxMenu    = e => e.preventDefault();

    canvas.addEventListener('mousedown',   this._onMouseDown);
    canvas.addEventListener('mousemove',   this._onMouseMove);
    canvas.addEventListener('mouseup',     this._onMouseUp);
    canvas.addEventListener('mouseleave',  this._onMouseUp);
    canvas.addEventListener('wheel',       this._onWheel, { passive: false });
    canvas.addEventListener('touchstart',  this._onTouchStart, { passive: false });
    canvas.addEventListener('touchmove',   this._onTouchMove,  { passive: false });
    canvas.addEventListener('touchend',    this._onTouchEnd);
    canvas.addEventListener('contextmenu', this._onCtxMenu);
  }

  detach() {
    const c = this._canvas;
    if (!c) return;
    c.removeEventListener('mousedown',   this._onMouseDown);
    c.removeEventListener('mousemove',   this._onMouseMove);
    c.removeEventListener('mouseup',     this._onMouseUp);
    c.removeEventListener('mouseleave',  this._onMouseUp);
    c.removeEventListener('wheel',       this._onWheel);
    c.removeEventListener('touchstart',  this._onTouchStart);
    c.removeEventListener('touchmove',   this._onTouchMove);
    c.removeEventListener('touchend',    this._onTouchEnd);
    c.removeEventListener('contextmenu', this._onCtxMenu);
    this._canvas = null;
  }

  // ── Mouse handlers ─────────────────────────────────────────────────────────

  _mouseDown(e) {
    this._drag = { x: e.clientX, y: e.clientY, button: e.button };
    e.preventDefault();
  }

  _mouseMove(e) {
    if (!this._drag) return;
    const dx = e.clientX - this._drag.x;
    const dy = e.clientY - this._drag.y;
    this._drag.x = e.clientX;
    this._drag.y = e.clientY;

    if (this._drag.button === 2) {
      // Right-drag: pan
      this._pan(dx, dy);
    } else {
      // Left-drag: orbit
      this._orbit(dx, dy);
    }
  }

  _wheel(e) {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    this.radius = Math.max(0.01, this.radius * factor);
  }

  // ── Touch handlers ─────────────────────────────────────────────────────────

  _touchStart(e) {
    e.preventDefault();
    this._touches = Array.from(e.touches).map(t => ({ id: t.identifier, x: t.clientX, y: t.clientY }));
  }

  _touchMove(e) {
    e.preventDefault();
    const prev = this._touches;
    const curr = Array.from(e.touches).map(t => ({ id: t.identifier, x: t.clientX, y: t.clientY }));

    if (curr.length === 1 && prev.length === 1) {
      const dx = curr[0].x - prev[0].x;
      const dy = curr[0].y - prev[0].y;
      this._orbit(dx, dy);
    } else if (curr.length === 2 && prev.length === 2) {
      // Pinch to zoom
      const prevDist = Math.hypot(prev[1].x - prev[0].x, prev[1].y - prev[0].y);
      const currDist = Math.hypot(curr[1].x - curr[0].x, curr[1].y - curr[0].y);
      if (prevDist > 0) {
        this.radius = Math.max(0.01, this.radius * (prevDist / currDist));
      }
      // Two-finger pan (centroid delta)
      const prevCx = (prev[0].x + prev[1].x) * 0.5;
      const prevCy = (prev[0].y + prev[1].y) * 0.5;
      const currCx = (curr[0].x + curr[1].x) * 0.5;
      const currCy = (curr[0].y + curr[1].y) * 0.5;
      this._pan(currCx - prevCx, currCy - prevCy);
    }

    this._touches = curr;
  }

  _touchEnd(e) {
    this._touches = Array.from(e.touches).map(t => ({ id: t.identifier, x: t.clientX, y: t.clientY }));
  }

  // ── Orbit & pan helpers ────────────────────────────────────────────────────

  _orbit(dx, dy) {
    const speed = 0.005;
    this.theta -= dx * speed;
    this.phi    = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.phi + dy * speed));
  }

  _pan(dx, dy) {
    // Move target in the camera's right/up plane, scaled by radius
    const speed = this.radius * 0.001;
    const right = this._cameraRight();
    const up    = this._cameraUp();
    this.target[0] -= (right[0] * dx - up[0] * dy) * speed;
    this.target[1] -= (right[1] * dx - up[1] * dy) * speed;
    this.target[2] -= (right[2] * dx - up[2] * dy) * speed;
  }

  _cameraRight() {
    // X axis of the camera in world space = first row of view matrix
    return [this.viewMatrix[0], this.viewMatrix[4], this.viewMatrix[8]];
  }

  _cameraUp() {
    // Y axis of the camera in world space = second row of view matrix
    return [this.viewMatrix[1], this.viewMatrix[5], this.viewMatrix[9]];
  }

  // ── Matrix computation ─────────────────────────────────────────────────────

  /** Update viewMatrix and projMatrix. Must be called before getViewMatrix(). */
  update(width, height) {
    const eye = this._eye();
    lookAt(eye, this.target, [0, 1, 0], this.viewMatrix);
    perspective(this.fov, width / height, this.near, this.far, this.projMatrix);
  }

  _eye() {
    const cp = Math.cos(this.phi), sp = Math.sin(this.phi);
    const ct = Math.cos(this.theta), st = Math.sin(this.theta);
    return [
      this.target[0] + this.radius * cp * st,
      this.target[1] + this.radius * sp,
      this.target[2] + this.radius * cp * ct,
    ];
  }

  /** Focal length in pixels for a given viewport dimension and fov. */
  focalLength(height) {
    return (height * 0.5) / Math.tan(this.fov * 0.5);
  }

  /**
   * Sync orbit state from an explicit eye + target.
   * After this call, update() reproduces the same view.
   * Used by the animation system so orbit controls resume from the animated position.
   */
  setFromLookAt(eye, target) {
    this.target = [target[0], target[1], target[2]];
    const dx = eye[0] - target[0];
    const dy = eye[1] - target[1];
    const dz = eye[2] - target[2];
    this.radius = Math.hypot(dx, dy, dz) || 0.001;
    this.phi    = Math.asin(Math.max(-1, Math.min(1, dy / this.radius)));
    this.theta  = Math.atan2(dx, dz);
  }

  /** Fit camera to a scene bounding box. */
  fitScene(positions, numSplats) {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < numSplats; i++) {
      const j = i * 16;
      const x = positions[j], y = positions[j + 1], z = positions[j + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    this.target = [
      (minX + maxX) * 0.5,
      (minY + maxY) * 0.5,
      (minZ + maxZ) * 0.5,
    ];
    const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 0.5;
    this.radius = extent / Math.tan(this.fov * 0.5) * 1.2;
  }
}

// ── Math helpers (column-major, WebGPU convention) ─────────────────────────

function lookAt(eye, center, up, out) {
  const [ex, ey, ez] = eye;
  const [cx, cy, cz] = center;
  const [ux, uy, uz] = up;

  let zx = ex - cx, zy = ey - cy, zz = ez - cz;
  let zl = Math.hypot(zx, zy, zz);
  zx /= zl; zy /= zl; zz /= zl;

  let xx = uy * zz - uz * zy;
  let xy = uz * zx - ux * zz;
  let xz = ux * zy - uy * zx;
  const xl = Math.hypot(xx, xy, xz);
  xx /= xl; xy /= xl; xz /= xl;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[ 0] = xx; out[ 1] = yx; out[ 2] = zx; out[ 3] = 0;
  out[ 4] = xy; out[ 5] = yy; out[ 6] = zy; out[ 7] = 0;
  out[ 8] = xz; out[ 9] = yz; out[10] = zz; out[11] = 0;
  out[12] = -(xx*ex + xy*ey + xz*ez);
  out[13] = -(yx*ex + yy*ey + yz*ez);
  out[14] = -(zx*ex + zy*ey + zz*ez);
  out[15] = 1;
}

function perspective(fovY, aspect, near, far, out) {
  const f  = 1.0 / Math.tan(fovY * 0.5);
  const nf = near - far;
  out[ 0] = f / aspect; out[ 1] = 0; out[ 2] = 0;  out[ 3] = 0;
  out[ 4] = 0;          out[ 5] = f; out[ 6] = 0;  out[ 7] = 0;
  out[ 8] = 0;          out[ 9] = 0; out[10] = far / nf; out[11] = -1;
  out[12] = 0;          out[13] = 0; out[14] = near * far / nf; out[15] = 0;
}

// ── src/sorter.js ────────────────────────────
/**
 * 16-bit two-pass radix sort for depth ordering.
 *
 * Sorts indices ascending by depth value (smallest depth = farthest = rendered first).
 * All buffers are pre-allocated at construction time — zero allocations per frame.
 */
function createSorter(maxN) {
  const keys0   = new Uint16Array(maxN);
  const keys1   = new Uint16Array(maxN);
  const idx0    = new Uint32Array(maxN);
  const idx1    = new Uint32Array(maxN);
  const counts  = new Int32Array(256);
  const pfx     = new Int32Array(256);

  /**
   * @param {Float32Array} depths  – one depth value per Gaussian
   * @param {number}       N       – number of Gaussians
   * @returns {Uint32Array}        – sorted indices (view into pre-alloc buffer)
   */
  return function sort(depths, N) {
    // ── Quantize depths to uint16 ──────────────────────────────────────────
    let minD = depths[0], maxD = depths[0];
    for (let i = 1; i < N; i++) {
      if (depths[i] < minD) minD = depths[i];
      if (depths[i] > maxD) maxD = depths[i];
    }
    const range = maxD - minD;
    const scale = range > 0 ? 65535 / range : 0;

    for (let i = 0; i < N; i++) {
      keys0[i] = Math.round((depths[i] - minD) * scale) | 0;
      idx0[i]  = i;
    }

    // ── Pass 1: sort by low byte ───────────────────────────────────────────
    counts.fill(0);
    for (let i = 0; i < N; i++) counts[keys0[i] & 0xff]++;

    pfx[0] = 0;
    for (let i = 1; i < 256; i++) pfx[i] = pfx[i - 1] + counts[i - 1];

    for (let i = 0; i < N; i++) {
      const k   = keys0[i] & 0xff;
      const pos = pfx[k]++;
      keys1[pos] = keys0[i];
      idx1[pos]  = idx0[i];
    }

    // ── Pass 2: sort by high byte (stable → preserves low-byte order) ─────
    counts.fill(0);
    for (let i = 0; i < N; i++) counts[(keys1[i] >> 8) & 0xff]++;

    pfx[0] = 0;
    for (let i = 1; i < 256; i++) pfx[i] = pfx[i - 1] + counts[i - 1];

    for (let i = 0; i < N; i++) {
      const k   = (keys1[i] >> 8) & 0xff;
      const pos = pfx[k]++;
      idx0[pos] = idx1[i];
    }

    return idx0; // ascending depth order = back-to-front
  };
}

// ── src/loaders/fetch-utils.js ────────────────────────────
/**
 * Shared fetch helper used by all loaders.
 * Streams the response body so onProgress(0..1) can be reported.
 */
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);

  const total  = parseInt(res.headers.get('content-length') || '0', 10);
  const reader = res.body.getReader();
  const chunks = [];
  let loaded   = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    if (onProgress && total > 0) onProgress(loaded / total);
  }

  const combined = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) { combined.set(chunk, offset); offset += chunk.byteLength; }
  return combined.buffer;
}

// ── src/loaders/splat-loader.js ────────────────────────────
/**
 * Loader for the .splat binary format.
 *
 * Each Gaussian is 32 bytes:
 *   bytes  0–11 : position     (3 × float32, little-endian)
 *   bytes 12–23 : scale        (3 × float32, already linear – no exp needed)
 *   bytes 24–27 : color RGBA   (4 × uint8, 0–255)
 *   bytes 28–31 : quaternion   (4 × uint8, w/x/y/z order, decoded via (b−128)/128)
 *
 * Output: Float32Array with 16 floats per Gaussian in the canonical layout:
 *   [0-2]  pos.xyz
 *   [3]    0 (padding)
 *   [4-7]  color.rgba (0–1)
 *   [8-10] scale.xyz
 *   [11]   0 (padding)
 *   [12-15] quat.xyzw
 */
async function loadSplat(url, onProgress) {
  const buffer = await fetchWithProgress(url, onProgress);
  return parseSplat(buffer);
}

function parseSplat(buffer) {
  const STRIDE = 32;
  const N      = Math.floor(buffer.byteLength / STRIDE);
  if (N === 0) throw new Error('Empty or invalid .splat file');

  const src  = new DataView(buffer);
  const data = new Float32Array(N * 16);

  for (let i = 0; i < N; i++) {
    const s  = i * STRIDE;
    const d  = i * 16;

    // Position
    data[d + 0] = src.getFloat32(s +  0, true);
    data[d + 1] = src.getFloat32(s +  4, true);
    data[d + 2] = src.getFloat32(s +  8, true);
    // d[3] = 0 (padding)

    // Color (uint8 → 0..1)
    data[d + 4] = src.getUint8(s + 24) / 255;
    data[d + 5] = src.getUint8(s + 25) / 255;
    data[d + 6] = src.getUint8(s + 26) / 255;
    data[d + 7] = src.getUint8(s + 27) / 255;

    // Scale (already linear in .splat format)
    data[d +  8] = src.getFloat32(s + 12, true);
    data[d +  9] = src.getFloat32(s + 16, true);
    data[d + 10] = src.getFloat32(s + 20, true);
    // d[11] = 0 (padding)

    // Quaternion: bytes 28–31 = [w, x, y, z], decoded (b−128)/128, then normalize
    const qw = (src.getUint8(s + 28) - 128) / 128;
    const qx = (src.getUint8(s + 29) - 128) / 128;
    const qy = (src.getUint8(s + 30) - 128) / 128;
    const qz = (src.getUint8(s + 31) - 128) / 128;
    const ql = Math.hypot(qx, qy, qz, qw) || 1;
    data[d + 12] = qx / ql;
    data[d + 13] = qy / ql;
    data[d + 14] = qz / ql;
    data[d + 15] = qw / ql;
  }

  return { data, count: N };
}

// fetchWithProgress is defined in fetch-utils.js (shared)

// ── src/loaders/ply-loader.js ────────────────────────────
/**
 * Loader for 3D Gaussian Splatting .ply files (standard 3DGS training output).
 *
 * Parses the ASCII header to discover property names and their byte offsets,
 * then reads the binary body. Handles all the 3DGS-specific transforms:
 *   - f_dc_0/1/2 → RGB via:  0.5 + SH_C0 * f_dc  (SH_C0 = 0.28209479177387814)
 *   - opacity    → sigmoid: 1 / (1 + exp(−x))
 *   - scale_i    → linear:  exp(scale_i)
 *   - rot_0..3   → quaternion, rot_0 = w (swizzle to xyzw), normalize
 *
 * Higher-order SH coefficients (f_rest_*) are ignored.
 *
 * Output: same canonical Float32Array layout as splat-loader (16 floats/Gaussian).
 */

const SH_C0 = 0.28209479177387814;

async function loadPly(url, onProgress) {
  const buffer = await fetchWithProgress(url, onProgress);
  return parsePly(buffer);
}

function parsePly(buffer) {
  const { numVertices, properties, dataOffset } = parseHeader(buffer);
  if (numVertices === 0) throw new Error('PLY file contains no vertices');

  // Build a property lookup: name → { byteOffset, type }
  const propMap = {};
  let stride = 0;
  for (const p of properties) {
    propMap[p.name] = { offset: stride, type: p.type };
    stride += sizeOf(p.type);
  }

  // Check required fields exist
  const required = ['x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2',
                    'rot_0', 'rot_1', 'rot_2', 'rot_3', 'opacity'];
  for (const r of required) {
    if (!propMap[r]) throw new Error(`PLY missing required property: ${r}`);
  }

  const hasColor = propMap['f_dc_0'] && propMap['f_dc_1'] && propMap['f_dc_2'];

  const src  = new DataView(buffer, dataOffset);
  const data = new Float32Array(numVertices * 16);

  for (let i = 0; i < numVertices; i++) {
    const base = i * stride;
    const d    = i * 16;

    // Position
    data[d + 0] = read(src, base, propMap['x']);
    data[d + 1] = read(src, base, propMap['y']);
    data[d + 2] = read(src, base, propMap['z']);
    // d[3] = 0

    // Color (DC spherical harmonics → linear RGB) or white fallback
    if (hasColor) {
      data[d + 4] = clamp01(0.5 + SH_C0 * read(src, base, propMap['f_dc_0']));
      data[d + 5] = clamp01(0.5 + SH_C0 * read(src, base, propMap['f_dc_1']));
      data[d + 6] = clamp01(0.5 + SH_C0 * read(src, base, propMap['f_dc_2']));
    } else {
      data[d + 4] = 1; data[d + 5] = 1; data[d + 6] = 1;
    }
    // Opacity: logit-space → sigmoid
    data[d + 7] = sigmoid(read(src, base, propMap['opacity']));

    // Scale: log-space → linear
    data[d +  8] = Math.exp(read(src, base, propMap['scale_0']));
    data[d +  9] = Math.exp(read(src, base, propMap['scale_1']));
    data[d + 10] = Math.exp(read(src, base, propMap['scale_2']));
    // d[11] = 0

    // Quaternion: PLY stores (w, x, y, z) as rot_0..3; we need (x, y, z, w)
    const rw = read(src, base, propMap['rot_0']);
    const rx = read(src, base, propMap['rot_1']);
    const ry = read(src, base, propMap['rot_2']);
    const rz = read(src, base, propMap['rot_3']);
    const rl = Math.hypot(rx, ry, rz, rw) || 1;
    data[d + 12] = rx / rl;
    data[d + 13] = ry / rl;
    data[d + 14] = rz / rl;
    data[d + 15] = rw / rl;
  }

  return { data, count: numVertices };
}

// ── Header parsing ─────────────────────────────────────────────────────────

function parseHeader(buffer) {
  // Read bytes until we find "end_header\n"
  const bytes   = new Uint8Array(buffer);
  const END_TAG = 'end_header';
  let headerEnd = -1;
  let text      = '';

  for (let i = 0; i < bytes.length; i++) {
    text += String.fromCharCode(bytes[i]);
    if (text.endsWith(END_TAG)) {
      // Skip the trailing newline(s)
      headerEnd = i + 1;
      if (bytes[headerEnd] === 13) headerEnd++; // \r
      if (bytes[headerEnd] === 10) headerEnd++; // \n
      break;
    }
  }

  if (headerEnd < 0) throw new Error('Invalid PLY: end_header not found');

  const lines = text.split('\n');
  let numVertices = 0;
  const properties = [];
  let inVertex = false;

  for (const raw of lines) {
    const line  = raw.trim();
    const parts = line.split(/\s+/);
    if (parts[0] === 'element') {
      inVertex = parts[1] === 'vertex';
      if (inVertex) numVertices = parseInt(parts[2], 10);
    } else if (parts[0] === 'property' && inVertex) {
      properties.push({ type: parts[1], name: parts[2] });
    }
  }

  return { numVertices, properties, dataOffset: headerEnd };
}

// ── Binary read helpers ────────────────────────────────────────────────────

function sizeOf(type) {
  switch (type) {
    case 'float': case 'float32': case 'int':   case 'uint':   return 4;
    case 'double': case 'int64': case 'uint64': return 8;
    case 'short': case 'ushort': case 'int16': case 'uint16': return 2;
    case 'char':  case 'uchar':  case 'int8':  case 'uint8':  return 1;
    default: return 4; // safe fallback
  }
}

function read(view, base, prop) {
  const off = base + prop.offset;
  switch (prop.type) {
    case 'float': case 'float32': return view.getFloat32(off, true);
    case 'double':                return view.getFloat64(off, true);
    case 'int': case 'int32':     return view.getInt32(off, true);
    case 'uint': case 'uint32':   return view.getUint32(off, true);
    case 'short': case 'int16':   return view.getInt16(off, true);
    case 'ushort': case 'uint16': return view.getUint16(off, true);
    case 'char': case 'int8':     return view.getInt8(off);
    case 'uchar': case 'uint8':   return view.getUint8(off);
    default:                      return view.getFloat32(off, true);
  }
}

// ── Math helpers ───────────────────────────────────────────────────────────

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

// fetchWithProgress is defined in fetch-utils.js (shared)

// ── src/loaders/spz-loader.js ────────────────────────────
/**
 * Loader for the .spz Gaussian splat format (Niantic Labs).
 *
 * The file is gzip-compressed. After decompression:
 *
 * Header (16 bytes, little-endian):
 *   [0-3]  magic    uint32  = 0x5053474E  ('N','G','S','P' → LE 0x5053474E)
 *   [4-7]  version  uint32  (2 = xyz-int8 rot, 3-4 = smallest-3 rot)
 *   [8-11] numPoints uint32
 *   [12]   shDegree  uint8
 *   [13]   fractionalBits uint8  (position precision; typically 12)
 *   [14]   flags     uint8
 *   [15]   reserved  uint8
 *
 * Data sections (column-organized, all for numPoints Gaussians):
 *   Positions  : numPoints × 9 bytes  (3 × int24 LE signed, ÷ 2^fractionalBits)
 *   Alphas     : numPoints × 1 byte   (uint8 → opacity 0–1)
 *   Colors     : numPoints × 3 bytes  (uint8 RGB, 0–255 → 0–1)
 *   Scales     : numPoints × 3 bytes  (uint8, log-space, exp((b–128)/16))
 *   Rotations  : v2:   numPoints × 3 bytes  (int8 x,y,z; w = sqrt(1-x²-y²-z²))
 *                v3-4: numPoints × 4 bytes  (2-bit largestIdx + three 10-bit signed)
 *
 * Output: canonical Float32Array layout (16 floats/Gaussian) compatible with
 *         the HoloSplat renderer — see shaders.js for the GPU-side layout.
 */

const MAGIC    = 0x5053474E;
const SQRT2INV = 1 / Math.SQRT2;

async function loadSpz(url, onProgress) {
  const compressed = await fetchWithProgress(url, onProgress);
  const buffer     = await decompressGzip(compressed);
  return parseSpz(buffer);
}

function parseSpz(buffer) {
  const view = new DataView(buffer);

  // ── Header ───────────────────────────────────────────────────────────────
  const magic          = view.getUint32(0, true);
  const version        = view.getUint32(4, true);
  const numPoints      = view.getUint32(8, true);
  const shDegree       = view.getUint8(12);
  const fractionalBits = view.getUint8(13);
  // flags / reserved at 14-15 (ignored for now)

  if (magic !== MAGIC) {
    throw new Error(
      `Invalid .spz magic: 0x${magic.toString(16).toUpperCase()} ` +
      `(expected 0x${MAGIC.toString(16).toUpperCase()})`
    );
  }
  if (version < 2 || version > 4) {
    console.warn(`HoloSplat: .spz version ${version} is untested; attempting load anyway`);
  }

  const posDiv  = 1 << fractionalBits;  // position divisor
  const rotSize = version >= 3 ? 4 : 3; // bytes per rotation

  // ── Section offsets ───────────────────────────────────────────────────────
  const offPos   = 16;
  const offAlpha = offPos   + numPoints * 9;
  const offColor = offAlpha + numPoints * 1;
  const offScale = offColor + numPoints * 3;
  const offRot   = offScale + numPoints * 3;
  // SH rest (if shDegree > 0): offRot + numPoints * rotSize — not used here

  const data = new Float32Array(numPoints * 16);

  for (let i = 0; i < numPoints; i++) {
    const d = i * 16;

    // ── Position (int24 signed LE, ÷ 2^fractionalBits) ───────────────────
    const pb = offPos + i * 9;
    data[d + 0] = readInt24(view, pb + 0) / posDiv;
    data[d + 1] = readInt24(view, pb + 3) / posDiv;
    data[d + 2] = readInt24(view, pb + 6) / posDiv;
    // d[3] = 0 (padding)

    // ── Color (uint8 RGB → 0-1) ───────────────────────────────────────────
    const cb = offColor + i * 3;
    data[d + 4] = view.getUint8(cb + 0) / 255;
    data[d + 5] = view.getUint8(cb + 1) / 255;
    data[d + 6] = view.getUint8(cb + 2) / 255;

    // ── Alpha (uint8 → 0-1) ───────────────────────────────────────────────
    data[d + 7] = view.getUint8(offAlpha + i) / 255;

    // ── Scale (uint8 log-space → linear) ─────────────────────────────────
    const sb = offScale + i * 3;
    data[d +  8] = Math.exp((view.getUint8(sb + 0) - 128) / 16);
    data[d +  9] = Math.exp((view.getUint8(sb + 1) - 128) / 16);
    data[d + 10] = Math.exp((view.getUint8(sb + 2) - 128) / 16);
    // d[11] = 0 (padding)

    // ── Rotation ─────────────────────────────────────────────────────────
    const rb = offRot + i * rotSize;
    let qx, qy, qz, qw;

    if (version >= 3) {
      // Smallest-3 encoding: 4 bytes
      //   bits [ 1: 0]  largestIdx (0=x, 1=y, 2=z, 3=w)
      //   bits [11: 2]  component a (int10 signed)
      //   bits [21:12]  component b (int10 signed)
      //   bits [31:22]  component c (int10 signed)
      const u32  = view.getUint32(rb, true);
      const idx  = u32 & 3;
      const s    = SQRT2INV / 512;              // int10 → [-1/√2, 1/√2]
      const a    = signExtend10((u32 >>  2) & 0x3FF) * s;
      const b_   = signExtend10((u32 >> 12) & 0x3FF) * s;
      const c    = signExtend10((u32 >> 22) & 0x3FF) * s;
      const d_   = Math.sqrt(Math.max(0, 1 - a*a - b_*b_ - c*c));

      // Re-insert the reconstructed largest component
      switch (idx) {
        case 0: qx = d_; qy = a;  qz = b_; qw = c;  break;
        case 1: qx = a;  qy = d_; qz = b_; qw = c;  break;
        case 2: qx = a;  qy = b_; qz = d_; qw = c;  break;
        default: qx = a; qy = b_; qz = c;  qw = d_;
      }
    } else {
      // v2: int8 x,y,z; reconstruct w = sqrt(1 - x²-y²-z²)
      qx = view.getInt8(rb + 0) / 128;
      qy = view.getInt8(rb + 1) / 128;
      qz = view.getInt8(rb + 2) / 128;
      qw = Math.sqrt(Math.max(0, 1 - qx*qx - qy*qy - qz*qz));
    }

    const ql = Math.hypot(qx, qy, qz, qw) || 1;
    data[d + 12] = qx / ql;
    data[d + 13] = qy / ql;
    data[d + 14] = qz / ql;
    data[d + 15] = qw / ql;
  }

  return { data, count: numPoints };
}

// ── Binary helpers ─────────────────────────────────────────────────────────

function readInt24(view, offset) {
  const lo = view.getUint8(offset);
  const mi = view.getUint8(offset + 1);
  const hi = view.getInt8(offset + 2);  // signed → sign extension
  return lo | (mi << 8) | (hi << 16);
}

function signExtend10(v) {
  // Sign-extend 10-bit integer to JS number
  return v & 0x200 ? v | 0xFFFFFC00 : v;
}

// ── Gzip decompression (uses browser DecompressionStream API) ──────────────

async function decompressGzip(buffer) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('DecompressionStream is not available in this environment');
  }
  const stream = new DecompressionStream('gzip');
  const writer = stream.writable.getWriter();
  writer.write(buffer);
  writer.close();
  return new Response(stream.readable).arrayBuffer();
}

// fetchWithProgress is defined in fetch-utils.js (shared)

// ── src/animation.js ────────────────────────────
/**
 * HoloSplat Animation — drives the camera from Blender-exported keyframe data.
 *
 * JSON format (produced by blender/export_holosplat.py):
 * {
 *   "version"    : 1,
 *   "fps"        : 24,
 *   "frameCount" : 120,
 *   "fov"        : 60,        // optional — overrides player fov when present
 *   "frames"     : [          // flat array, 6 values per frame:
 *     ex, ey, ez,             //   eye position (viewer Y-up space)
 *     fx, fy, fz,             //   normalized forward vector
 *     ...
 *   ],
 *   "callouts" : [            // world-space points to project onto the screen
 *     { "id": "label", "pos": [x, y, z] }
 *   ]
 * }
 *
 * Coordinates are in viewer Y-up space. The Blender export script converts
 * from Blender's Z-up space automatically.
 */

class Animation {
  /**
   * @param {object} data  Parsed JSON from export_holosplat.py
   */
  constructor(data) {
    if (!data.frames || data.frames.length === 0) {
      throw new Error('HoloSplat Animation: no frame data');
    }

    this.fps        = data.fps        ?? 24;
    this.frameCount = data.frameCount ?? Math.floor(data.frames.length / 6);
    this.fov        = data.fov        ?? null;   // null = keep player fov
    this.callouts   = data.callouts   ?? [];
    this.loop       = true;

    // Typed array: 6 floats per frame [ex ey ez fx fy fz]
    this._frames  = new Float32Array(data.frames);

    this._time    = 0;
    this._playing = true;
  }

  // ── Read-only state ─────────────────────────────────────────────────────────

  get duration() { return this.frameCount / this.fps; }
  get time()     { return this._time; }
  get playing()  { return this._playing; }

  // ── Playback control ────────────────────────────────────────────────────────

  play()  { this._playing = true; }
  pause() { this._playing = false; }

  seek(seconds) {
    this._time = Math.max(0, Math.min(this.duration, seconds));
  }

  /**
   * Advance playback by dt seconds. Call once per render tick.
   */
  tick(dt) {
    if (!this._playing) return;
    this._time += dt;
    if (this._time >= this.duration) {
      this._time = this.loop ? this._time % this.duration : this.duration;
      if (!this.loop) this._playing = false;
    }
  }

  // ── Camera frame ────────────────────────────────────────────────────────────

  /**
   * Returns { eye, target } arrays for the current playback time.
   * `target` is eye + forward (1 unit ahead), suitable for lookAt.
   */
  getCameraFrame() {
    const frame = Math.min(
      Math.floor(this._time * this.fps),
      this.frameCount - 1
    );
    const i = frame * 6;
    const f = this._frames;
    return {
      eye:    [f[i],             f[i + 1],         f[i + 2]],
      target: [f[i] + f[i + 3], f[i + 1] + f[i + 4], f[i + 2] + f[i + 5]],
    };
  }
}

/**
 * Fetch and parse an animation JSON file.
 * @param {string} url
 * @returns {Promise<Animation>}
 */
async function loadAnimation(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HoloSplat: failed to load animation "${url}" (HTTP ${res.status})`);
  return new Animation(await res.json());
}

// ── src/compress.js ────────────────────────────
/**
 * HoloSplat compressor — encodes canonical Gaussian data to the .spz format
 * and gzip-compresses it for storage or transfer.
 *
 * The .spz format (Niantic Labs, v3) stores each Gaussian as:
 *   Position  9 bytes  (int24 × 3, fixed-point with auto-selected precision)
 *   Alpha     1 byte   (uint8)
 *   Color     3 bytes  (uint8 RGB)
 *   Scale     3 bytes  (uint8 log-space quantized)
 *   Rotation  4 bytes  (smallest-3 quaternion encoding)
 *   ─────────────────
 *   Total    20 bytes/Gaussian (before gzip) vs 32 bytes for .splat
 *
 * Input: canonical Float32Array (16 floats/Gaussian):
 *   [0-2]  pos.xyz
 *   [3]    pad
 *   [4-7]  color.rgba (0-1)
 *   [8-10] scale.xyz (linear)
 *   [11]   pad
 *   [12-15] quat.xyzw
 *
 * Usage:
 *   import { compressToSpz } from 'holosplat';
 *   const buffer = await compressToSpz(data, count);
 *   const blob = new Blob([buffer], { type: 'application/octet-stream' });
 */

const SPZ_MAGIC   = 0x5053474E;
const SPZ_VERSION = 3;

/**
 * Encode canonical Gaussian data and gzip-compress to a .spz ArrayBuffer.
 *
 * @param {Float32Array} data   – 16 floats/Gaussian, canonical layout
 * @param {number}       count  – number of Gaussians
 * @param {object}       [opts]
 * @param {number}       [opts.fractionalBits]  – position fixed-point precision
 *                                                (auto-detected from data range if omitted)
 * @returns {Promise<ArrayBuffer>}
 */
async function compressToSpz(data, count, opts = {}) {
  const raw = encodeSpz(data, count, opts);
  return compressGzip(raw);
}

/**
 * Encode canonical Gaussian data to a raw (uncompressed) .spz Uint8Array.
 * Useful if you want to compress with a different algorithm or inspect the bytes.
 *
 * @param {Float32Array} data
 * @param {number}       count
 * @param {object}       [opts]
 * @param {number}       [opts.fractionalBits]
 * @returns {Uint8Array}
 */
function encodeSpz(data, count, opts = {}) {
  // ── Choose position fixed-point precision ─────────────────────────────────
  // int24 range: -8,388,608 … 8,388,607  (2^23 - 1)
  // max representable position = INT24_MAX / 2^fractionalBits
  let { fractionalBits } = opts;
  if (fractionalBits == null) {
    let maxAbsPos = 0;
    for (let i = 0; i < count; i++) {
      const b = i * 16;
      if (Math.abs(data[b])     > maxAbsPos) maxAbsPos = Math.abs(data[b]);
      if (Math.abs(data[b + 1]) > maxAbsPos) maxAbsPos = Math.abs(data[b + 1]);
      if (Math.abs(data[b + 2]) > maxAbsPos) maxAbsPos = Math.abs(data[b + 2]);
    }
    const INT24_MAX = (1 << 23) - 1;
    fractionalBits = maxAbsPos > 0
      ? Math.min(20, Math.max(0, Math.floor(Math.log2(INT24_MAX / maxAbsPos))))
      : 12;
  }

  // ── Allocate buffer ───────────────────────────────────────────────────────
  const HEADER = 16;
  const total  = HEADER + count * (9 + 1 + 3 + 3 + 4);
  const buf    = new ArrayBuffer(total);
  const view   = new DataView(buf);
  const u8     = new Uint8Array(buf);

  // ── Header ────────────────────────────────────────────────────────────────
  view.setUint32( 0, SPZ_MAGIC,       true);
  view.setUint32( 4, SPZ_VERSION,     true);
  view.setUint32( 8, count,           true);
  view.setUint8 (12, 0);                    // shDegree
  view.setUint8 (13, fractionalBits);
  view.setUint8 (14, 0);                    // flags
  view.setUint8 (15, 0);                    // reserved

  // ── Section offsets (column-organised, matching the SPZ spec) ────────────
  const offPos   = HEADER;
  const offAlpha = offPos   + count * 9;
  const offColor = offAlpha + count * 1;
  const offScale = offColor + count * 3;
  const offRot   = offScale + count * 3;

  const posScale = 1 << fractionalBits;

  for (let i = 0; i < count; i++) {
    const d = i * 16;

    // Position (float → fixed-point int24 LE)
    writeInt24(view, offPos + i * 9 + 0, data[d + 0] * posScale);
    writeInt24(view, offPos + i * 9 + 3, data[d + 1] * posScale);
    writeInt24(view, offPos + i * 9 + 6, data[d + 2] * posScale);

    // Alpha
    u8[offAlpha + i] = clampU8(data[d + 7] * 255);

    // Color RGB
    u8[offColor + i * 3 + 0] = clampU8(data[d + 4] * 255);
    u8[offColor + i * 3 + 1] = clampU8(data[d + 5] * 255);
    u8[offColor + i * 3 + 2] = clampU8(data[d + 6] * 255);

    // Scale: linear → log-space uint8
    // Inverse of: Math.exp((b - 128) / 16)  →  b = log(scale) * 16 + 128
    u8[offScale + i * 3 + 0] = encodeScale(data[d +  8]);
    u8[offScale + i * 3 + 1] = encodeScale(data[d +  9]);
    u8[offScale + i * 3 + 2] = encodeScale(data[d + 10]);

    // Rotation: smallest-3 encoding (4 bytes, uint32 LE)
    view.setUint32(offRot + i * 4,
      encodeQuat(data[d + 12], data[d + 13], data[d + 14], data[d + 15]),
      true);
  }

  return u8;
}

// ── Encoding helpers ──────────────────────────────────────────────────────────

function writeInt24(view, offset, value) {
  const v = Math.max(-8388608, Math.min(8388607, Math.round(value)));
  view.setUint8(offset,     v & 0xFF);
  view.setUint8(offset + 1, (v >>  8) & 0xFF);
  view.setUint8(offset + 2, (v >> 16) & 0xFF);
}

function clampU8(v) {
  return Math.max(0, Math.min(255, Math.round(v)));
}

function encodeScale(linear) {
  return clampU8(Math.log(Math.max(1e-9, linear)) * 16 + 128);
}

/**
 * Encode a unit quaternion (x,y,z,w) using smallest-3 / "omit largest" encoding.
 *
 * The largest component is stored implicitly as sqrt(1 - a² - b² - c²).
 * Canonically it must be positive, so we flip the sign of all components when
 * the largest is negative (the rotation is identical either way).
 *
 * Packed uint32:
 *   bits [ 1: 0]  index of largest component (0=x,1=y,2=z,3=w)
 *   bits [11: 2]  a  (int10, scaled by 512*√2 to fill [-1/√2, 1/√2])
 *   bits [21:12]  b
 *   bits [31:22]  c
 */
function encodeQuat(qx, qy, qz, qw) {
  // Normalize
  const len = Math.hypot(qx, qy, qz, qw) || 1;
  const q = [qx / len, qy / len, qz / len, qw / len];

  // Find largest-magnitude component
  let maxIdx = 0;
  for (let j = 1; j < 4; j++) {
    if (Math.abs(q[j]) > Math.abs(q[maxIdx])) maxIdx = j;
  }

  // Ensure largest is positive (flip sign of whole quaternion if needed)
  const sign = q[maxIdx] < 0 ? -1 : 1;

  // The three "other" components, in the order the decoder expects:
  //   maxIdx=0 → [qy,qz,qw]  idx [1,2,3]
  //   maxIdx=1 → [qx,qz,qw]  idx [0,2,3]
  //   maxIdx=2 → [qx,qy,qw]  idx [0,1,3]
  //   maxIdx=3 → [qx,qy,qz]  idx [0,1,2]
  // [0,1,2,3].filter(j => j !== maxIdx) produces exactly these in ascending order.
  const others = [0, 1, 2, 3].filter(j => j !== maxIdx);

  // Scale factor: decoder uses s = (1/√2) / 512,  encoder uses 1/s = 512*√2
  const S = 512 * Math.SQRT2;
  const encode10 = j => Math.max(-512, Math.min(511, Math.round(q[j] * sign * S)));

  const a = encode10(others[0]);
  const b = encode10(others[1]);
  const c = encode10(others[2]);

  // Pack and return as unsigned 32-bit (>>> 0 ensures correct bit pattern)
  return ((maxIdx & 3) | ((a & 0x3FF) << 2) | ((b & 0x3FF) << 12) | ((c & 0x3FF) << 22)) >>> 0;
}

// ── Gzip compression ──────────────────────────────────────────────────────────

async function compressGzip(data) {
  if (typeof CompressionStream === 'undefined') {
    throw new Error('CompressionStream API is not available in this environment');
  }
  const stream = new CompressionStream('gzip');
  const writer = stream.writable.getWriter();
  writer.write(data);
  writer.close();
  return new Response(stream.readable).arrayBuffer();
}

// ── src/renderer.js ────────────────────────────
/**
 * WebGPU renderer for Gaussian splatting.
 *
 * Manages device, pipeline, bind groups, and GPU buffers.
 * Upload happens once (gaussians); order buffer is updated every frame.
 *
 * Buffer layout (must match src/shaders.js):
 *   Uniforms  : 160 bytes (40 × f32)
 *   Gaussians : N × 64 bytes (16 × f32 each)
 *   Order     : N × 4 bytes (u32 indices, sorted back-to-front)
 */

// Uniform buffer offsets (in float32 index units)
const U_VIEW     = 0;   // mat4 (16 floats)
const U_PROJ     = 16;  // mat4 (16 floats)
const U_VIEWPORT = 32;  // vec2 (2 floats)
const U_FOCAL    = 34;  // vec2 (2 floats)
const U_PARAMS   = 36;  // vec4 (4 floats), .x = splatScale
const U_SIZE     = 40;  // total floats → 160 bytes

class Renderer {
  constructor(canvas, background) {
    this.canvas     = canvas;
    this.background = parseBackground(background);
    this.device     = null;
    this.context    = null;
    this.pipeline   = null;
    this.bindGroup  = null;

    // GPU buffers
    this._uniformBuf  = null;
    this._gaussianBuf = null;
    this._orderBuf    = null;

    // CPU-side uniform data
    this._uniforms = new Float32Array(U_SIZE);
    this._uniforms[U_PARAMS] = 1.0; // splatScale = 1

    this._numSplats = 0;
  }

  // ── Initialise WebGPU ──────────────────────────────────────────────────────

  async init() {
    if (!navigator.gpu) throw new Error('WebGPU is not supported in this browser.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter found.');
    this.device = await adapter.requestDevice();

    this.context = this.canvas.getContext('webgpu');
    this._format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this._format,
      alphaMode: 'premultiplied',
    });

    this._createPipeline();
    this._uniformBuf = this._createBuffer(U_SIZE * 4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  }

  // ── Upload scene data (call once after load) ───────────────────────────────

  uploadGaussians(data, count) {
    this._numSplats = count;

    // Gaussian storage buffer (read-once upload)
    if (this._gaussianBuf) this._gaussianBuf.destroy();
    this._gaussianBuf = this.device.createBuffer({
      size:  data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this._gaussianBuf, 0, data);

    // Order buffer (rewritten every frame)
    if (this._orderBuf) this._orderBuf.destroy();
    this._orderBuf = this.device.createBuffer({
      size:  count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this._rebuildBindGroup();
  }

  // ── Per-frame updates ──────────────────────────────────────────────────────

  updateUniforms({ view, proj, width, height, focal }) {
    const u = this._uniforms;
    u.set(view,   U_VIEW);
    u.set(proj,   U_PROJ);
    u[U_VIEWPORT]     = width;
    u[U_VIEWPORT + 1] = height;
    u[U_FOCAL]        = focal;
    u[U_FOCAL + 1]    = focal;
    this.device.queue.writeBuffer(this._uniformBuf, 0, u);
  }

  updateOrder(sortedIndices, count) {
    this.device.queue.writeBuffer(
      this._orderBuf, 0,
      sortedIndices.buffer, 0, count * 4
    );
  }

  setSplatScale(s) {
    this._uniforms[U_PARAMS] = s;
  }

  setBackground(bg) {
    this.background = parseBackground(bg);
  }

  // ── Draw ───────────────────────────────────────────────────────────────────

  draw() {
    if (!this._numSplats || !this.bindGroup) return;

    const encoder = this.device.createCommandEncoder();
    const pass    = encoder.beginRenderPass({
      colorAttachments: [{
        view:       this.context.getCurrentTexture().createView(),
        clearValue: this.background,
        loadOp:     'clear',
        storeOp:    'store',
      }],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(6, this._numSplats, 0, 0); // 6 verts/instance, N instances
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────

  destroy() {
    this._uniformBuf?.destroy();
    this._gaussianBuf?.destroy();
    this._orderBuf?.destroy();
    this.context?.unconfigure();
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  _createPipeline() {
    const module = this.device.createShaderModule({ code: SHADER });

    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module, entryPoint: 'vs_main' },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{
          format: this._format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      // No depth/stencil — Gaussians are manually sorted back-to-front
    });
  }

  _createBuffer(size, usage) {
    return this.device.createBuffer({ size, usage });
  }

  _rebuildBindGroup() {
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._uniformBuf  } },
        { binding: 1, resource: { buffer: this._gaussianBuf } },
        { binding: 2, resource: { buffer: this._orderBuf    } },
      ],
    });
  }
}

// ── Background colour parsing ──────────────────────────────────────────────

function parseBackground(bg) {
  if (!bg || bg === 'transparent') return { r: 0, g: 0, b: 0, a: 0 };
  if (Array.isArray(bg))           return { r: bg[0], g: bg[1], b: bg[2], a: bg[3] ?? 1 };
  if (typeof bg === 'string') {
    const hex = bg.replace('#', '');
    if (hex.length === 6) {
      return {
        r: parseInt(hex.slice(0, 2), 16) / 255,
        g: parseInt(hex.slice(2, 4), 16) / 255,
        b: parseInt(hex.slice(4, 6), 16) / 255,
        a: 1,
      };
    }
    if (hex.length === 8) {
      return {
        r: parseInt(hex.slice(0, 2), 16) / 255,
        g: parseInt(hex.slice(2, 4), 16) / 255,
        b: parseInt(hex.slice(4, 6), 16) / 255,
        a: parseInt(hex.slice(6, 8), 16) / 255,
      };
    }
  }
  return { r: 0, g: 0, b: 0, a: 1 };
}

// ── src/viewer.js ────────────────────────────
/**
 * HoloSplat Viewer — orchestrates load → sort → render.
 *
 * Usage:
 *   const viewer = new Viewer(options);
 *   await viewer.init();
 *   await viewer.load(url);
 *   viewer.start();
 */







class Viewer {
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

    this._animation  = null;  // Animation instance, or null
    this._animPaused = false; // true → animation frozen, camera responds to user input
    this._lastTick   = null;  // performance.now() of previous tick (for dt)
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

  /** Freeze / unfreeze animation playback. When paused, camera responds to user input. */
  setAnimationPaused(paused) { this._animPaused = paused; }

  resetCamera() { this._camera.fitScene(this._gaussians, this._numSplats); }

  // ── Animation ──────────────────────────────────────────────────────────────

  /** Attach a pre-loaded Animation instance. Pass null to detach. */
  setAnimation(anim) {
    this._animation = anim;
    if (anim?.fov != null) {
      this._camera.fov = anim.fov * Math.PI / 180;
    }
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

    if (this._animation && !this._animPaused) {
      this._animation.tick(dt);
      const { eye, target } = this._animation.getCameraFrame();
      this._camera.setFromLookAt(eye, target);
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

// ── src/player.js ────────────────────────────
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
function player(container, opts = {}) {
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
    setBackground(bg)        { viewer.setBackground(bg); },
    setSplatScale(s)         { viewer.setSplatScale(s); },
    setAutoRotate(v)         { viewer.setAutoRotate(v); },
    setAnimationPaused(v)    { viewer.setAnimationPaused(v); },
    resetCamera()            { viewer.resetCamera(); },
    callout(id)              { return calloutDivs[id] ?? null; },
    get camera()             { return viewer.camera; },
    get animation()          { return viewer._animation; },
    get animationPaused()    { return viewer._animPaused; },
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

// ── src/index.js ────────────────────────────
/**
 * HoloSplat – WebGPU Gaussian Splat viewer library
 *
 * ── Quick start ──────────────────────────────────────────────────────────────
 *
 *   ESM (bundler / native module):
 *     import { create } from 'holosplat';
 *
 *   IIFE (Webflow / plain script tag):
 *     <script src="holosplat.iife.js"></script>
 *     HoloSplat.create({ ... });
 *
 * ── create(options) ──────────────────────────────────────────────────────────
 *
 *   canvas     {string|HTMLCanvasElement}  – required; CSS selector or element
 *   src        {string}                   – URL of .splat or .ply file
 *   background {string|number[]}          – '#rrggbb', '#rrggbbaa', 'transparent',
 *                                           or [r,g,b,a] (0–1). Default '#000000'
 *   fov        {number}                   – vertical field of view, degrees. Default 60
 *   near       {number}                   – near clip plane. Default 0.1
 *   far        {number}                   – far clip plane. Default 2000
 *   splatScale {number}                   – global scale multiplier. Default 1
 *   autoRotate {boolean}                  – slow continuous orbit. Default false
 *   onLoad     {function}                 – called after scene is fully loaded
 *   onProgress {function(0..1)}           – called during fetch with progress 0–1
 *   onError    {function(Error)}          – called on any error
 *
 * ── Returns ───────────────────────────────────────────────────────────────────
 *
 *   A controller object with:
 *     destroy()            – stop rendering, release GPU resources
 *     setBackground(bg)    – change background colour
 *     setSplatScale(s)     – change global splat scale
 *     setAutoRotate(bool)  – toggle auto-rotate
 *     resetCamera()        – fit camera back to scene
 *     camera               – OrbitCamera instance (for direct manipulation)
 */






async function create(options = {}) {
  const { onLoad, onError, src, ...viewerOpts } = options;

  const viewer = new Viewer({ ...viewerOpts });

  const noop = {
    destroy() {}, setBackground() {}, setSplatScale() {},
    setAutoRotate() {}, resetCamera() {}, camera: null,
  };

  try {
    await viewer.init();
    if (src) await viewer.load(src);
  } catch (err) {
    viewer.destroy();
    if (onError) { onError(err); return noop; }
    throw err;
  }

  viewer.start();
  onLoad?.();

  return {
    destroy()           { viewer.destroy(); },
    setBackground(bg)   { viewer.setBackground(bg); },
    setSplatScale(s)    { viewer.setSplatScale(s); },
    setAutoRotate(v)    { viewer.setAutoRotate(v); },
    resetCamera()       { viewer.resetCamera(); },
    get camera()        { return viewer.camera; },
  };
}

// Also expose Viewer class, animation, parsers, and compression utilities


  return { create, player, Viewer, Animation, loadAnimation, compressToSpz, encodeSpz, parseSplat, parsePly };
})();
