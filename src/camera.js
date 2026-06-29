/**
 * Orbit camera with mouse and touch controls.
 * Spherical coordinates (theta = azimuth, phi = elevation) around a target point.
 *
 * View matrix is column-major Float32Array(16) compatible with WebGPU.
 */
export class OrbitCamera {
  constructor({ fov = 60, near = 0.01, far = 2000 } = {}) {
    this.fov    = fov * Math.PI / 180;  // radians
    this.near   = near;
    this.far    = far;

    // Spherical state
    this.theta  = 0;           // azimuth
    this.phi    = 0.2;         // elevation (clamped away from poles)
    this.radius = 5;
    this.target = [0, 0, 0];

    // Set to false to disable all input handling (e.g. when scroll-scene owns playback).
    this.enabled    = true;
    // Set to false to disable drag-orbit specifically. When that happens,
    // left-drag pans instead of orbiting (if panEnabled), so pan is
    // reachable without requiring a right-click drag.
    this.orbitEnabled = true;
    this.panEnabled = true;
    this.panSpeed   = 1;     // sensitivity multiplier (0..1, derived from damping)
    this.panButton  = 2;     // mouse button that starts a pan-drag: 0 = left, 2 = right
    this.panRadius  = null;  // max distance from panOrigin; null = unlimited
    this.panOrigin  = null;  // world-space [x,y,z] centre for limited pan

    // Angular constraints (radians). null = unconstrained.
    // Set by constrainAngles(); cleared by clearConstraints().
    this.thetaMin = null;
    this.thetaMax = null;
    this.phiMin   = null;
    this.phiMax   = null;

    // Zoom (radius) control. zoomEnabled=false blocks wheel/pinch entirely.
    this.zoomEnabled = true;
    this.radiusMin   = null;
    this.radiusMax   = null;

    // Overlay callbacks — set by viewer.js when hs-* markers are active.
    // orbitDeltaCallback(dTheta, dPhi)  — drag input routed as deltas instead of direct state mutation
    // dragStartCallback()               — fired on left-button mousedown / touch
    // dragEndCallback()                 — fired on mouseup / mouseleave / touch end
    // panDeltaCallback(dx, dy, dz)      — right-drag/two-finger pan routed as a world-space
    //                                      delta instead of mutating target directly (the
    //                                      animation overwrites target every frame)
    // zoomDeltaCallback(factor)         — wheel/pinch routed as a radius multiplier instead
    //                                      of mutating radius directly (animation overwrites
    //                                      radius every frame)
    this.orbitDeltaCallback = null;
    this.dragStartCallback  = null;
    this.dragEndCallback    = null;
    this.panDeltaCallback   = null;
    this.zoomDeltaCallback  = null;

    this._drag    = null;      // { x, y, button }
    this._touches = [];

    // When true, single-finger touch is left alone (no preventDefault, no
    // orbit/pan) so the page can scroll natively — set by player.js
    // for scroll-driven scenes, where swipe-to-scroll is the primary mobile
    // interaction. Two-finger pinch/pan still works regardless. Cleared
    // during freecamera "explore" acts so single-finger touch can orbit.
    // Using the setter so touch-action is updated on the document root.
    this._allowTouchScroll = false;

    this.viewMatrix = new Float32Array(16);
    this.projMatrix = new Float32Array(16);
  }

  get allowTouchScroll() { return this._allowTouchScroll; }
  set allowTouchScroll(v) {
    this._allowTouchScroll = !!v;
    // pan-y: browser handles vertical scroll natively; horizontal touches reach JS
    // uninterrupted (no touchcancel), so horizontal-swipe orbit still works.
    // Resetting to '' restores default `auto` when scroll mode is off.
    if (typeof document !== 'undefined') {
      document.documentElement.style.touchAction = v ? 'pan-y' : '';
    }
  }

  // ── Attach / detach input listeners ────────────────────────────────────────

  attach(canvas) {
    this._canvas = canvas;

    // Check whether a client point is actually over the canvas — not just
    // within its bounding rect, since overlays (e.g. the holosplat editor
    // panel) are positioned on top of the canvas via z-index while the
    // canvas itself still spans the full viewport underneath them. A rect-
    // only check would treat clicks on such overlays as canvas clicks,
    // calling preventDefault() on their mousedown and silently blocking the
    // browser's default click-to-focus behavior on any input inside them.
    const overCanvas = (cx, cy) => canvas.contains(document.elementFromPoint(cx, cy));

    // Mouse events: listen on document so overlapping elements don't block them.
    this._onMouseDown  = e => { if (overCanvas(e.clientX, e.clientY)) this._mouseDown(e); };
    this._onMouseMove  = e => this._mouseMove(e);
    this._onMouseUp    = () => { this._drag = null; this.dragEndCallback?.(); };
    // Wheel must stay on canvas so we can preventDefault without the passive restriction.
    this._onWheel      = e => this._wheel(e);
    // Touch: listen on document for same reason.
    this._onTouchStart = e => { if (overCanvas(e.touches[0]?.clientX, e.touches[0]?.clientY)) this._touchStart(e); };
    this._onTouchMove  = e => this._touchMove(e);
    this._onTouchEnd   = e => this._touchEnd(e);
    this._onCtxMenu    = e => { if (overCanvas(e.clientX, e.clientY)) e.preventDefault(); };
    // Mouse leaves the browser window entirely: relatedTarget is null only when
    // the pointer exits the document (not when moving to another element).
    // Ends any active drag even when the cursor is dragged or jumps
    // off-window without a mouseup.
    this._onMouseOutDoc = e => {
      if (e.relatedTarget !== null) return;
      if (this._drag) { this._drag = null; this.dragEndCallback?.(); }
    };

    document.addEventListener('mousedown',   this._onMouseDown);
    document.addEventListener('mousemove',   this._onMouseMove);
    document.addEventListener('mouseup',     this._onMouseUp);
    document.addEventListener('mouseout',    this._onMouseOutDoc);
    canvas.addEventListener('wheel',         this._onWheel, { passive: false });
    document.addEventListener('touchstart',  this._onTouchStart, { passive: false });
    document.addEventListener('touchmove',   this._onTouchMove,  { passive: false });
    document.addEventListener('touchend',    this._onTouchEnd);
    document.addEventListener('contextmenu', this._onCtxMenu);
  }

  detach() {
    const c = this._canvas;
    if (!c) return;
    document.removeEventListener('mousedown',   this._onMouseDown);
    document.removeEventListener('mousemove',   this._onMouseMove);
    document.removeEventListener('mouseup',     this._onMouseUp);
    document.removeEventListener('mouseout',    this._onMouseOutDoc);
    c.removeEventListener('wheel',              this._onWheel);
    document.removeEventListener('touchstart',  this._onTouchStart);
    document.removeEventListener('touchmove',   this._onTouchMove);
    document.removeEventListener('touchend',    this._onTouchEnd);
    document.removeEventListener('contextmenu', this._onCtxMenu);
    this._canvas = null;
  }

  // ── Mouse handlers ─────────────────────────────────────────────────────────

  _mouseDown(e) {
    if (!this.enabled) return;
    if (e.button === 2 && !this.panEnabled) return;
    this._drag = { x: e.clientX, y: e.clientY, button: e.button };
    if (e.button === 0) this.dragStartCallback?.();
    e.preventDefault();
  }

  _mouseMove(e) {
    if (this._drag) {
      const dx = e.clientX - this._drag.x;
      const dy = e.clientY - this._drag.y;
      this._drag.x = e.clientX;
      this._drag.y = e.clientY;

      const isPanButton = this._drag.button === this.panButton;
      if (this.panEnabled && (isPanButton || (!this.orbitEnabled && this._drag.button === 0))) {
        // pan.button drag, or left-drag when orbit is off but pan is on: pan.
        this._pan(dx, dy);
        return;
      }
      if (this._drag.button === 0 && !isPanButton) {
        // Left-drag (when not bound to pan): orbit
        this._orbit(dx, dy);
      }
    }
  }

  _wheel(e) {
    if (!this.enabled || !this.zoomEnabled) return;
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    if (this.zoomDeltaCallback) { this.zoomDeltaCallback(factor); return; }
    this.radius = Math.max(0.01, this.radius * factor);
    if (this.radiusMin !== null) this.radius = Math.max(this.radiusMin, this.radius);
    if (this.radiusMax !== null) this.radius = Math.min(this.radiusMax, this.radius);
  }

  // ── Touch handlers ─────────────────────────────────────────────────────────

  _touchStart(e) {
    if (!this.enabled) return;
    if (this.allowTouchScroll && e.touches.length < 2) {
      return; // no preventDefault — page scrolls freely
    }
    e.preventDefault();
    this._touches = Array.from(e.touches).map(t => ({ id: t.identifier, x: t.clientX, y: t.clientY }));
  }

  _touchMove(e) {
    if (!this.enabled) return;
    if (this.allowTouchScroll && e.touches.length < 2) {
      this._touches = [];
      return;
    }
    e.preventDefault();
    const prev = this._touches;
    const curr = Array.from(e.touches).map(t => ({ id: t.identifier, x: t.clientX, y: t.clientY }));

    if (curr.length === 1 && prev.length === 1) {
      const dx = curr[0].x - prev[0].x;
      const dy = curr[0].y - prev[0].y;
      // Single-finger touch acts as a left-click drag.
      if (this.panEnabled && (this.panButton === 0 || !this.orbitEnabled)) {
        this._pan(dx, dy);
      } else {
        this._orbit(dx, dy);
      }
    } else if (curr.length === 2 && prev.length === 2) {
      // Pinch to zoom
      const prevDist = Math.hypot(prev[1].x - prev[0].x, prev[1].y - prev[0].y);
      const currDist = Math.hypot(curr[1].x - curr[0].x, curr[1].y - curr[0].y);
      if (prevDist > 0 && currDist > 0 && this.zoomEnabled) {
        const factor = prevDist / currDist;
        if (this.zoomDeltaCallback) {
          this.zoomDeltaCallback(factor);
        } else {
          this.radius = Math.max(0.01, this.radius * factor);
          if (this.radiusMin !== null) this.radius = Math.max(this.radiusMin, this.radius);
          if (this.radiusMax !== null) this.radius = Math.min(this.radiusMax, this.radius);
        }
      }
      // Two-finger pan (centroid delta)
      if (this.panEnabled) {
        const prevCx = (prev[0].x + prev[1].x) * 0.5;
        const prevCy = (prev[0].y + prev[1].y) * 0.5;
        const currCx = (curr[0].x + curr[1].x) * 0.5;
        const currCy = (curr[0].y + curr[1].y) * 0.5;
        this._pan(currCx - prevCx, currCy - prevCy);
      }
    }

    this._touches = curr;
  }

  _touchEnd(e) {
    this._touches = Array.from(e.touches).map(t => ({ id: t.identifier, x: t.clientX, y: t.clientY }));
    if (this._touches.length === 0) this.dragEndCallback?.();
  }

  // ── Orbit & pan helpers ────────────────────────────────────────────────────

  _orbit(dx, dy) {
    const speed  = 0.005;
    const dTheta = -dx * speed;
    const dPhi   =  dy * speed;
    if (this.orbitDeltaCallback) {
      this.orbitDeltaCallback(dTheta, dPhi);
      return;
    }
    this.theta += dTheta;
    this.phi    = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.phi + dPhi));
    if (this.thetaMin !== null) this.theta = Math.max(this.thetaMin, this.theta);
    if (this.thetaMax !== null) this.theta = Math.min(this.thetaMax, this.theta);
    if (this.phiMin   !== null) this.phi   = Math.max(this.phiMin,   this.phi);
    if (this.phiMax   !== null) this.phi   = Math.min(this.phiMax,   this.phi);
  }

  /**
   * Restrict orbit to ±hDeg horizontal and ±vDeg vertical from the current
   * theta/phi. Call after setFromLookAt() so the current angles are the center.
   * Pass null for either axis to leave it unconstrained.
   */
  constrainAngles(hDeg, vDeg) {
    if (hDeg !== null) {
      const r = hDeg * Math.PI / 180;
      this.thetaMin = this.theta - r;
      this.thetaMax = this.theta + r;
    } else {
      this.thetaMin = null;
      this.thetaMax = null;
    }
    if (vDeg !== null) {
      const r = vDeg * Math.PI / 180;
      this.phiMin = Math.max(-Math.PI / 2 + 0.01, this.phi - r);
      this.phiMax = Math.min( Math.PI / 2 - 0.01, this.phi + r);
    } else {
      this.phiMin = null;
      this.phiMax = null;
    }
  }

  clearConstraints() {
    this.thetaMin = null;
    this.thetaMax = null;
    this.phiMin   = null;
    this.phiMax   = null;
  }

  /** Disable scroll/pinch zoom entirely (used when entering free-camera mode without hs-zoom). */
  disableZoom() {
    this.zoomEnabled = false;
    this.radiusMin   = null;
    this.radiusMax   = null;
  }

  /** Enable zoom constrained to ±rangePct% of baseRadius. */
  constrainZoom(baseRadius, rangePct) {
    const r = rangePct / 100;
    this.zoomEnabled = true;
    this.radiusMin   = Math.max(0.01, baseRadius * (1 - r));
    this.radiusMax   = baseRadius * (1 + r);
  }

  /** Re-enable unrestricted zoom (used when leaving free-camera mode). */
  enableZoom() {
    this.zoomEnabled = true;
    this.radiusMin   = null;
    this.radiusMax   = null;
  }

  _pan(dx, dy) {
    const speed = this.radius * 0.001 * this.panSpeed;
    const right = this._cameraRight();
    const up    = this._cameraUp();
    const ddx = -(right[0] * dx - up[0] * dy) * speed;
    const ddy = -(right[1] * dx - up[1] * dy) * speed;
    const ddz = -(right[2] * dx - up[2] * dy) * speed;
    if (this.panDeltaCallback) { this.panDeltaCallback(ddx, ddy, ddz); return; }
    this.target[0] += ddx;
    this.target[1] += ddy;
    this.target[2] += ddz;
    if (this.panRadius !== null && this.panOrigin) {
      const [ox, oy, oz] = this.panOrigin;
      const dist = Math.hypot(this.target[0]-ox, this.target[1]-oy, this.target[2]-oz);
      if (dist > this.panRadius) {
        const f = this.panRadius / dist;
        this.target[0] = ox + (this.target[0]-ox) * f;
        this.target[1] = oy + (this.target[1]-oy) * f;
        this.target[2] = oz + (this.target[2]-oz) * f;
      }
    }
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

  get eye() { return this._eye(); }

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

  /** Fit camera to a scene bounding box, resetting angle to default. */
  fitScene(positions, numSplats) {
    this._sceneBounds(positions, numSplats);
    this.theta = 0;
    this.phi   = 0.2;
  }

  /** Fit camera to a scene bounding box, preserving current angle. */
  focusScene(positions, numSplats) {
    this._sceneBounds(positions, numSplats);
  }

  _sceneBounds(positions, numSplats) {
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
    this.radius  = extent / Math.tan(this.fov * 0.5) * 1.2;
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
