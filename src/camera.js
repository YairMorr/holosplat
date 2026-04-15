/**
 * Orbit camera with mouse and touch controls.
 * Spherical coordinates (theta = azimuth, phi = elevation) around a target point.
 *
 * View matrix is column-major Float32Array(16) compatible with WebGPU.
 */
export class OrbitCamera {
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
