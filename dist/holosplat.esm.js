/**
 * HoloSplat v0.1.0 – WebGPU Gaussian Splat viewer
 * https://github.com/your-org/holosplat
 * License: MIT
 */
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

// Also expose Viewer class for advanced usage


export { create, Viewer };
