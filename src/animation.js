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
 *   }
 * }
 *
 * Coordinates are in viewer Y-up space. The Blender export script converts
 * from Blender's Z-up space automatically.
 */

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
    this.loop       = true;

    // Named timeline markers: { markerName: frameNumber }
    // Accepts either object { name: frame } or array [{ name, frame }]
    if (Array.isArray(data.markers)) {
      this.markers = Object.fromEntries(data.markers.map(m => [m.name, m.frame]));
    } else {
      this.markers = data.markers ?? {};
    }

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
export async function loadAnimation(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HoloSplat: failed to load animation "${url}" (HTTP ${res.status})`);
  return new Animation(await res.json());
}
