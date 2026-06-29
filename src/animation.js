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
 *   ],
 *   "markers" : {             // Blender timeline markers → frame numbers (0-based)
 *     "intro"       : 0,      // used by scroll-scene data-from / data-to attributes
 *     "desk_reveal" : 72,
 *     "end"         : 119
 *   },
 *   "stateCalls" : [          // "state: <asset>.<axis>=<value>" markers
 *     { "frame": 80, "asset": "headphones", "axis": "fold", "value": "folded" }
 *   ]
 * }
 *
 * Coordinates are in viewer Y-up space. The Blender export script converts
 * from Blender's Z-up space automatically.
 */

// Quaternion slerp — takes the shortest arc and normalizes at the boundary.
export function slerpQuat(ax, ay, az, aw, bx, by, bz, bw, t) {
  let dot = ax * bx + ay * by + az * bz + aw * bw;
  if (dot < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; dot = -dot; }
  if (dot > 0.9995) {
    // Quaternions nearly identical — safe to lerp + normalize.
    const x = ax + (bx - ax) * t, y = ay + (by - ay) * t;
    const z = az + (bz - az) * t, w = aw + (bw - aw) * t;
    const n = Math.sqrt(x * x + y * y + z * z + w * w);
    return [x / n, y / n, z / n, w / n];
  }
  const theta0 = Math.acos(dot), theta = theta0 * t;
  const st0 = Math.sin(theta0), st = Math.sin(theta);
  const sa = Math.cos(theta) - dot * st / st0, sb = st / st0;
  return [sa * ax + sb * bx, sa * ay + sb * by, sa * az + sb * bz, sa * aw + sb * bw];
}

/**
 * Extract the splat file stem from a Blender empty name.
 *
 * Old convention — control prefixes followed by a single name:
 *   "fork-left"               → "fork-left"
 *   "ctrl.fork-left"          → "fork-left"
 *   "ctrl.fork-left.001"      → "fork-left"
 *   "hs-part.ctrl.fork-left"  → "fork-left"
 *
 * New hierarchical convention — dotted name IS the file stem:
 *   "headphones.headband"     → "headphones.headband"
 *   "headphones.cup.l"        → "headphones.cup.l"
 *   "headphones.cup.l.001"    → "headphones.cup.l"
 *
 * Control prefixes "hs-part." and "ctrl." are always stripped first,
 * then trailing all-digit Blender duplicate suffixes (e.g. ".001") are removed.
 */
export function splatNameFromId(id) {
  let s = id.replace(/^hs-part\./, '').replace(/^ctrl\./, '');
  s = s.replace(/(\.\d+)+$/, '');
  return s;
}

export class Animation {
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
    this.near       = data.near       ?? null;   // null = keep player near
    this.far        = data.far        ?? null;   // null = keep player far
    this.callouts   = data.callouts   ?? [];
    this.focalPoint  = data.focalPoint ?? null;
    this._focalFrames = data.focalFrames ? new Float32Array(data.focalFrames) : null;
    this.loop       = true;

    // Named timeline markers: { markerName: frameNumber }
    // Accepts either object { name: frame } or array [{ name, frame }]
    if (Array.isArray(data.markers)) {
      this.markers = Object.fromEntries(data.markers.map(m => [m.name, m.frame]));
    } else {
      this.markers = data.markers ?? {};
    }

    // "state: <asset>.<axis>=<value>" markers — calls into an asset's own
    // state axis (see export_holosplat_asset.py). Resolved every tick by
    // Viewer#_syncAssetStates, same "most recently passed" rule as markers
    // above, so seeking/scrubbing the main timeline lands on the right value
    // without needing edge-triggered crossing detection.
    this.stateCalls = data.stateCalls ?? [];

    // Typed array: 6 floats per frame [ex ey ez fx fy fz]
    this._frames  = new Float32Array(data.frames);

    // Per-object animated transforms: 7 floats per frame [px py pz qx qy qz qw]
    this._objects = (data.objects ?? []).map(obj => ({
      id:       obj.id,
      frames:   new Float32Array(obj.frames),
      variants: obj.variants,
    }));

    // Mask volumes: each has a name prefix (for CPU-side part matching) and
    // 16 floats per frame (world-space col-major mat4).
    this._volumes = (data.volumes ?? []).map(v => ({
      name:     v.name,
      softEdge: v.softEdge ?? 0.05,
      matrices: new Float32Array(v.matrices),
    }));

    // Asset anchors: a pure parent transform (7 floats per frame, same
    // layout as _objects) for an externally-loaded asset's parts — see
    // Viewer#_tick, which composes this onto any loaded part whose id is
    // namespaced under the anchor's asset id ("hs-anchor.<assetId>" in
    // Blender). Not bind-pose relative like parts — there's no rest pose to
    // subtract, the asset's own parts already carry their own.
    this._anchors = (data.anchors ?? []).map(a => ({
      asset:  a.asset,
      frames: new Float32Array(a.frames),
    }));

    // Clips (independent, button-triggered per-object animations — product
    // customization) live in their own separate asset JSON files now, loaded
    // via Viewer#loadClips — not part of the main Animation/timeline at all.
    // See export_holosplat_asset.py and Viewer#playClip.

    this._time    = 0;
    this._playing = true;
    this.direction = 1;     // 1 = forward, -1 = reverse
    this.pingPong  = false; // bounce direction at each end instead of looping
  }

  // ── Read-only state ─────────────────────────────────────────────────────────

  get duration() { return this.frameCount / this.fps; }
  get time()     { return this._time; }
  get playing()  { return this._playing; }
  /** Array of tracked objects: [{ id, frames, variants? }]. Empty for v1 animations. */
  get objects()  { return this._objects; }
  /** Array of mask volumes: [{ name, softEdge, matrices }]. Empty if none exported. */
  get volumes()  { return this._volumes; }
  /** Array of asset anchors: [{ asset, frames }]. Empty if none exported. */
  get anchors()  { return this._anchors; }

  // ── Playback control ────────────────────────────────────────────────────────

  play()  { this._playing = true; }
  pause() { this._playing = false; }

  seek(seconds) {
    this._time = Math.max(0, Math.min(this.duration, seconds));
  }

  /** Seek to an exact frame number (0-based). */
  seekFrame(frame) {
    this._time = Math.max(0, Math.min(this.frameCount - 1, frame)) / this.fps;
  }

  /** Current playback position as a frame number (may be fractional). */
  get frame() {
    return this._time * this.fps;
  }

  /**
   * Advance playback by dt seconds. Call once per render tick.
   */
  tick(dt) {
    if (!this._playing) return;
    this._time += dt * this.direction;
    if (this.direction >= 0) {
      if (this._time >= this.duration) {
        if (this.pingPong) {
          this.direction = -1;
          this._time = 2 * this.duration - this._time; // reflect
        } else {
          this._time = this.loop ? this._time % this.duration : this.duration;
          if (!this.loop) this._playing = false;
        }
      }
    } else {
      if (this._time <= 0) {
        if (this.pingPong) {
          this.direction = 1;
          this._time = -this._time; // reflect
        } else if (this.loop) {
          this._time = this.duration + this._time; // wrap (time was negative)
        } else {
          this._time = 0;
          this._playing = false;
        }
      }
    }
  }

  // ── Camera frame ────────────────────────────────────────────────────────────

  /**
   * Returns [{ id, pos:[x,y,z], quat:[x,y,z,w] }] for every tracked object at
   * the current playback time, or at `frameOverride` if given — lets callers
   * sample a pose at an arbitrary frame without touching actual playback
   * state (see player.js's scene-blend, which needs to peek at an adjacent
   * scene's frame while still playing its own).
   * Empty array for v1 animations with no objects.
   */
  getObjectFrames(frameOverride) {
    const rawFrame = frameOverride != null
      ? Math.max(0, Math.min(frameOverride, this.frameCount - 1))
      : Math.min(this._time * this.fps, this.frameCount - 1);
    const frameA   = Math.min(Math.floor(rawFrame), this.frameCount - 1);
    const frameB   = Math.min(frameA + 1, this.frameCount - 1);
    const t        = rawFrame - frameA;
    return this._objects.map(obj => {
      const f = obj.frames;
      const ia = frameA * 7, ib = frameB * 7;
      return {
        id:   obj.id,
        pos:  [
          f[ia]     + (f[ib]     - f[ia])     * t,
          f[ia + 1] + (f[ib + 1] - f[ia + 1]) * t,
          f[ia + 2] + (f[ib + 2] - f[ia + 2]) * t,
        ],
        quat: slerpQuat(f[ia+3], f[ia+4], f[ia+5], f[ia+6], f[ib+3], f[ib+4], f[ib+5], f[ib+6], t),
      };
    });
  }

  /**
   * Returns [{ asset, pos:[x,y,z], quat:[x,y,z,w] }] for every asset anchor
   * at the current playback time. Same interpolation as getObjectFrames —
   * see there for frameOverride. Empty array if none exported.
   */
  getAnchorFrames(frameOverride) {
    const rawFrame = frameOverride != null
      ? Math.max(0, Math.min(frameOverride, this.frameCount - 1))
      : Math.min(this._time * this.fps, this.frameCount - 1);
    const frameA = Math.min(Math.floor(rawFrame), this.frameCount - 1);
    const frameB = Math.min(frameA + 1, this.frameCount - 1);
    const t      = rawFrame - frameA;
    return this._anchors.map(a => {
      const f = a.frames;
      const ia = frameA * 7, ib = frameB * 7;
      return {
        asset: a.asset,
        pos:  [
          f[ia]     + (f[ib]     - f[ia])     * t,
          f[ia + 1] + (f[ib + 1] - f[ia + 1]) * t,
          f[ia + 2] + (f[ib + 2] - f[ia + 2]) * t,
        ],
        quat: slerpQuat(f[ia+3], f[ia+4], f[ia+5], f[ia+6], f[ib+3], f[ib+4], f[ib+5], f[ib+6], t),
      };
    });
  }

  /**
   * Returns [{ name, softEdge, matrix: Float32Array(16) }] for every mask volume
   * at the current playback time, or at `frameOverride` if given — see
   * getObjectFrames for why (player.js's scene-blend needs to peek at an
   * adjacent scene's mask state too, not just camera/object pose, otherwise
   * masks snap instantly at a scene handoff while everything else eases in).
   * Sub-frame interpolated, same as the other getters, so a mask doesn't
   * step abruptly between integer frames during normal playback either.
   */
  getVolumeFrames(frameOverride) {
    const rawFrame = frameOverride != null
      ? Math.max(0, Math.min(frameOverride, this.frameCount - 1))
      : Math.min(this._time * this.fps, this.frameCount - 1);
    const frameA = Math.min(Math.floor(rawFrame), this.frameCount - 1);
    const frameB = Math.min(frameA + 1, this.frameCount - 1);
    const t      = rawFrame - frameA;
    return this._volumes.map(v => {
      const m  = v.matrices;
      const ia = frameA * 16, ib = frameB * 16;
      const matrix = new Float32Array(16);
      for (let i = 0; i < 16; i++) matrix[i] = m[ia + i] + (m[ib + i] - m[ia + i]) * t;
      return { name: v.name, softEdge: v.softEdge, matrix };
    });
  }

  /**
   * Returns the focal point world position at the current frame,
   * or null if no focal point is defined.
   * Uses per-frame data (focalFrames) when available — handles the case where
   * the focal-point Empty is a child of an animated parent.
   */
  getFocalPoint() {
    if (!this.focalPoint) return null;
    if (!this._focalFrames) return this.focalPoint;
    const rawFrame = Math.min(this._time * this.fps, this.frameCount - 1);
    const frameA   = Math.min(Math.floor(rawFrame), this.frameCount - 1);
    const frameB   = Math.min(frameA + 1, this.frameCount - 1);
    const t        = rawFrame - frameA;
    const f        = this._focalFrames;
    const ia = frameA * 3, ib = frameB * 3;
    return [
      f[ia]     + (f[ib]     - f[ia])     * t,
      f[ia + 1] + (f[ib + 1] - f[ia + 1]) * t,
      f[ia + 2] + (f[ib + 2] - f[ia + 2]) * t,
    ];
  }

  /**
   * Returns { eye, target } arrays for the current playback time, or at
   * `frameOverride` if given — see getObjectFrames for why.
   * `target` is eye + forward (1 unit ahead), suitable for lookAt.
   */
  getCameraFrame(frameOverride) {
    const rawFrame = frameOverride != null
      ? Math.max(0, Math.min(frameOverride, this.frameCount - 1))
      : Math.min(this._time * this.fps, this.frameCount - 1);
    const frameA   = Math.min(Math.floor(rawFrame), this.frameCount - 1);
    const frameB   = Math.min(frameA + 1, this.frameCount - 1);
    const t        = rawFrame - frameA;
    const f        = this._frames;
    const ia = frameA * 6, ib = frameB * 6;
    const ex = f[ia]     + (f[ib]     - f[ia])     * t;
    const ey = f[ia + 1] + (f[ib + 1] - f[ia + 1]) * t;
    const ez = f[ia + 2] + (f[ib + 2] - f[ia + 2]) * t;
    const fx = f[ia + 3] + (f[ib + 3] - f[ia + 3]) * t;
    const fy = f[ia + 4] + (f[ib + 4] - f[ia + 4]) * t;
    const fz = f[ia + 5] + (f[ib + 5] - f[ia + 5]) * t;
    return {
      eye:    [ex,      ey,      ez],
      target: [ex + fx, ey + fy, ez + fz],
    };
  }
}

/**
 * Returns [{ id, pos:[x,y,z], quat:[x,y,z,w] }] for every object animated by
 * `clip`, at `frame` (clamped to the clip's own [0, frameCount-1] range and
 * linearly/slerp-interpolated, same scheme as Animation#getObjectFrames).
 * Clips aren't tied to an Animation instance's own playback clock — Viewer
 * manages each clip's local frame position itself (see Viewer#playClip) —
 * so this is a standalone function rather than an Animation method.
 */
export function getClipObjectFrames(clip, frame) {
  const frameA = Math.max(0, Math.min(Math.floor(frame), clip.frameCount - 1));
  const frameB = Math.min(frameA + 1, clip.frameCount - 1);
  const t      = Math.max(0, Math.min(1, frame - frameA));
  return clip.objects.map(obj => {
    const f = obj.frames;
    const ia = frameA * 7, ib = frameB * 7;
    return {
      id:   obj.id,
      pos:  [
        f[ia]     + (f[ib]     - f[ia])     * t,
        f[ia + 1] + (f[ib + 1] - f[ia + 1]) * t,
        f[ia + 2] + (f[ib + 2] - f[ia + 2]) * t,
      ],
      quat: slerpQuat(f[ia+3], f[ia+4], f[ia+5], f[ia+6], f[ib+3], f[ib+4], f[ib+5], f[ib+6], t),
    };
  });
}

/**
 * Returns [{ name, softEdge, matrix:Float32Array(16) }] for every mask
 * volume animated by `clip`, at `frame` (clamped + linearly interpolated,
 * same scheme as getClipObjectFrames). A clip's masks are independent of
 * its objects — a clip may animate masks only (e.g. fading between color
 * variants while the parts themselves stay put).
 */
export function getClipMaskFrames(clip, frame) {
  const masks = clip.masks;
  if (!masks || !masks.length) return [];
  const frameA = Math.max(0, Math.min(Math.floor(frame), clip.frameCount - 1));
  const frameB = Math.min(frameA + 1, clip.frameCount - 1);
  const t      = Math.max(0, Math.min(1, frame - frameA));
  return masks.map(({ name, softEdge, matrices: m }) => {
    const ia = frameA * 16, ib = frameB * 16;
    const matrix = new Float32Array(16);
    for (let k = 0; k < 16; k++) matrix[k] = m[ia + k] + (m[ib + k] - m[ia + k]) * t;
    return { name, softEdge, matrix };
  });
}

/**
 * Returns { pos:[x,y,z], quat:[x,y,z,w] } interpolated from a flat 7-floats-
 * per-frame array at `frame` (clamped to [0, frameCount-1]). Used by
 * Viewer's axis-transition playback (see Viewer#playVariant), where each
 * value's part data is its own standalone array, not grouped into a clip.
 */
export function interpPosQuat(frames, frameCount, frame) {
  const frameA = Math.max(0, Math.min(Math.floor(frame), frameCount - 1));
  const frameB = Math.min(frameA + 1, frameCount - 1);
  const t      = Math.max(0, Math.min(1, frame - frameA));
  const ia = frameA * 7, ib = frameB * 7;
  return {
    pos: [
      frames[ia]     + (frames[ib]     - frames[ia])     * t,
      frames[ia + 1] + (frames[ib + 1] - frames[ia + 1]) * t,
      frames[ia + 2] + (frames[ib + 2] - frames[ia + 2]) * t,
    ],
    quat: slerpQuat(frames[ia+3], frames[ia+4], frames[ia+5], frames[ia+6], frames[ib+3], frames[ib+4], frames[ib+5], frames[ib+6], t),
  };
}

/**
 * Returns a Float32Array(16) interpolated from a flat 16-floats-per-frame
 * matrix array at `frame` (clamped to [0, frameCount-1]). See interpPosQuat.
 */
export function interpMat4Frames(matrices, frameCount, frame) {
  const frameA = Math.max(0, Math.min(Math.floor(frame), frameCount - 1));
  const frameB = Math.min(frameA + 1, frameCount - 1);
  const t      = Math.max(0, Math.min(1, frame - frameA));
  const ia = frameA * 16, ib = frameB * 16;
  const out = new Float32Array(16);
  for (let k = 0; k < 16; k++) out[k] = matrices[ia + k] + (matrices[ib + k] - matrices[ia + k]) * t;
  return out;
}

/**
 * Fetch and parse an animation JSON file.
 * @param {string} url
 * @returns {Promise<Animation>}
 */
export async function loadAnimation(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HoloSplat: failed to load animation "${url}" (HTTP ${res.status})`);
  return new Animation(await res.json());
}
