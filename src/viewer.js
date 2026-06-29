/**
 * HoloSplat Viewer — orchestrates load → sort → render.
 *
 * Usage:
 *   const viewer = new Viewer(options);
 *   await viewer.init();
 *   await viewer.load(url);
 *   viewer.start();
 */
import { Renderer, invertMat4 } from './renderer.js';
import { OrbitCamera }  from './camera.js';
import { createSorter } from './sorter.js';
import { loadSplat }                              from './loaders/splat-loader.js';
import { loadPly, openPlyStream } from './loaders/ply-loader.js';
import { loadSpz }                                from './loaders/spz-loader.js';
import { loadSpzv }                               from './loaders/spzv-loader.js';
import { loadAnimation, getClipObjectFrames, getClipMaskFrames, interpPosQuat, interpMat4Frames, slerpQuat } from './animation.js';
import { resolvePartsLod, resolveLodUrl } from './device-tier.js';


export class Viewer {
  constructor(options = {}) {
    const {
      canvas,
      background = '#000000',
      fov        = 60,
      near       = 0.01,
      far        = 2000,
      splatScale = 1.08,
      autoRotate = false,
      flipY      = false,
      shDegree   = 0,
      aaDilation = 0.3,
      maxPixelRatio = 2,
      adaptiveQuality = true,
      prefetchVariants = true,
      gpuSort = false,
      tier = null, // 'low'|'medium'|'high'|null — see device-tier.js; surfaced via getStats() for the ?hs mobile overlay
      onProgress,
      onError,
    } = options;

    this._canvas     = resolveCanvas(canvas);
    this._onProgress = onProgress;
    this._onError    = onError;
    this._autoRotate = autoRotate;
    this._splatScale  = splatScale;
    this._maxPixelRatio = maxPixelRatio;
    this._flipY       = flipY;
    this._shDegree    = shDegree;
    this._aaDilation  = aaDilation;
    this._tier        = tier;

    // Adaptive quality: dynamically scales render resolution between
    // _minPixelRatio and _maxPixelRatio based on a running average of frame
    // time — see _updateAdaptiveQuality. _effectivePixelRatio is the value
    // _updateSize() actually uses.
    this._adaptiveQuality    = adaptiveQuality;
    this._prefetchVariantsEnabled = prefetchVariants;
    // GPU compute-shader radix sort (see src/sort-shaders.js), opt-in. Only
    // used when activeCount === null (no per-part cull active this frame) —
    // see _tick. Default false: zero behavior change unless explicitly enabled.
    this._gpuSort = gpuSort;
    this._effectivePixelRatio = maxPixelRatio;
    this._minPixelRatio      = 0.5;
    this._frameTimeEMA       = null;
    this._lastQualityCheck   = 0;
    this._wasSceneReady      = false;
    this._qualityWarmupUntil = 0; // see the _sceneReady rising-edge check in _tick

    // Screen-space splat radius cap (fraction of viewport.y — see shaders.js
    // `maxRadius`), eased between _minRadiusCap and _maxRadiusCap by
    // _updateAdaptiveQuality alongside _effectivePixelRatio. Tightening this
    // under load reduces the overdraw cost of large close-up splats.
    // _maxRadiusCap is a safety net against runaway/degenerate splat scales,
    // not a routine constraint — it must stay well above what normal close-up
    // framing produces (0.25 was too tight: legitimate close-up shots of a
    // surface routinely have visible splats with screen radii > 25% of the
    // viewport height, which made them visibly fade away while zooming in).
    this._effectiveRadiusCap = 1.0;
    this._minRadiusCap       = 0.4;
    this._maxRadiusCap       = 1.0;

    this._renderer   = new Renderer(this._canvas, background);
    this._camera     = new OrbitCamera({ fov, near, far });

    this._gaussians  = null;  // Float32Array after load
    this._numSplats  = 0;
    this._depths     = null;  // pre-allocated Float32Array
    this._sort       = null;  // sorter function
    this._rafId      = null;
    this._running    = false;
    this._resizeObs   = null;
    this._resizeTimer = null;

    // Multi-part support
    this._partIndex     = {};                        // id → transform slot index(es)
    this._fileNames      = [];                       // slot → splat name, for mask matching
    this._partVariants   = {};                       // slot → 'spzv' { variants:[{name,palette}], active } | 'file' { names, active, baseUrl, ext, cache }
    this._partTransforms = [IDENTITY_MAT4.slice()]; // Float32Array[] one per slot
    this._partTransFlat  = IDENTITY_MAT4.slice();   // flattened, written to GPU each frame
    this._partLocalPose  = {};                       // id → local mat4 set this tick (clip/transition/state/anim object), for anchors to compose against

    // Mask-aware active-subset rendering (skip fully-hidden files when masks are present)
    this._fileRanges = null;  // slot → [start, end) gaussian index range
    this._fileAABB   = null;  // slot → { min: [x,y,z], max: [x,y,z] } local-space bounds
    this._fileMasks  = null;  // slot → u32 bitmask of mask volumes affecting this file
    this._activeIdx  = null;  // Uint32Array(_numSplats) — packed gaussian indices, active subset
    this._lastActiveCount = 0; // splats actually drawn last frame — see getStats()
    this._currentSceneName = null; // set by player.js's scroll-playback override — see getStats()

    this._animation   = null;  // Animation instance, or null
    this._animPaused  = false; // true → animation frozen, camera responds to user input
    this._cameraFree  = false; // true → scroll-scene freecamera act owns the camera
    this._blendBack   = null;  // { fromEye, fromTarget, t, duration } — blend from freecamera back to animation
    // { otherEye, otherTarget, otherObjects: {id:{pos,quat}}, bf } | null — set
    // externally by player.js's scroll-scene blend. Crossfades the live
    // animation pose (this scene's own frame) with `other` at weight bf (bf=1
    // → fully live, bf=0 → fully other). See _tick.
    this._sceneBlend  = null;
    this._sceneReady  = false; // true → all splat data is on GPU, animation may tick
    this._lastUrl     = null;  // last URL passed to load() — used by setShDegree to reload
    this._lastParts   = null;  // last parts map passed to loadParts() — used by setShDegree to reload
    this._lastTick    = null;  // performance.now() of previous tick (for dt)
    this.onFrame     = null;  // callback(view, proj, width, height) — called each tick
    this._lastSortView = null; // Float32Array(16) — view matrix used for the last sort
    this._sortDirty   = true;  // force sort+render on first frame and after data/uniform changes
    this._lastRenderMs = 0;    // performance.now() of last sort+render — for 60fps cap

    // Pan/zoom overlays: applied on top of the animation-driven look-at each
    // tick since the animation overwrites both every frame.
    this._panOffset  = [0, 0, 0];
    this._panLimit   = null; // max |panOffset| (world units), or null = unlimited
    this._zoomFactor = 1;    // radius multiplier
    this._zoomLimit  = null; // max deviation from 1 (e.g. 0.5 = 50%..150%), or null = unlimited
    this._camMode          = null;
    this._camModeType      = null;

    // Per-volume softEdge ("feather") overrides set by the editor/player config.
    // Keyed by mask volume name; applied on top of the Blender-exported default.
    this._maskFeather = {};

    // Clips (product customization) — independent of the main Animation/
    // timeline entirely, loaded from separate asset files via loadClips().
    // See export_holosplat_asset.py and playClip().
    this._clips = {}; // clipId -> { id, fps, frameCount, holdFrame, objects }
    // Keyed by "product" (a clip id's prefix before its last hyphen, e.g.
    // "headphones" for "headphones-blue"), so each product's customization
    // plays independently of every other product's.
    this._clipPlaybacks = {}; // product -> { clip, dir:'in'|'out', frame, nextClipId }
    this._clipHeld      = {}; // product -> currently-held clip id, or undefined
    // Resting/current world matrix + softEdge for every mask volume declared
    // by any loaded clip — independent of this._animation's own volumes
    // (which only exist for the main scroll timeline). Updated by
    // _applyClipFrame while a clip plays; otherwise holds whatever its
    // clip last left it at (its "out" pose by default — see loadClips()).
    this._clipMaskState      = {}; // mask name -> Float32Array(16)
    this._clipMaskSoftEdge   = {}; // mask name -> number

    // loadParts() always replaces the WHOLE merged scene (see its docstring) —
    // fine for a single call, but loadClips() now makes two per asset (default
    // variant first, then the full set once the background fetch finishes —
    // see loadClips()) and multiple assets can each have their own splatsDir
    // parts. Without accumulating into one registry and always loading the
    // combined map, every one of those calls would wipe out whatever any
    // other asset/phase had already loaded — which is what was causing the
    // flicker/stutter (parts repeatedly vanishing and reappearing) and the
    // resulting adaptive-quality resolution drop.
    this._clipPartsRegistry  = {}; // part id -> url|url[]|{url,variants}, accumulated across loadClips() calls
    this._clipPartsLoadQueue = Promise.resolve(); // serializes the loadParts() calls below
    this._clipPartsFlush     = null; // pending debounced flush — see _mergeClipParts

    // Per-URL cache of loadUrl() results. Every queued loadParts() call below
    // re-loads the FULL accumulated registry from scratch (it's the only way
    // loadParts() knows how to load anything — see its docstring), so without
    // this, each new asset/phase landing would re-fetch every splat file
    // every earlier call had already loaded, snowballing into the "lots of
    // 404s" / repeated downloads seen with multiple clip assets.
    this._urlCache = new Map(); // url#shDegree -> Promise<loadUrl() result>

    // Axis transitions (button-triggered color switches) — one shared
    // in/hold/out timeline per axis, not one per value (contrast with
    // clips above). See loadClips()/playVariant().
    this._transitions        = {}; // axis -> { fps, frameCount, holdFrame, parts, masks }
    this._transitionPlaybacks = {}; // axis -> { value, prevValue, elapsed }
    this._axisActive         = {}; // axis -> currently active value, or undefined

    // Asset states ("state: <axis>=<value>" markers) — one continuous
    // timeline per axis spanning every named value, as opposed to the
    // two-value crossfade transitions above. Switching values seeks along
    // the shared timeline, forward or backward, through whatever frames lie
    // between. See loadClips()/playState()/_syncAssetStates().
    this._states          = {}; // axis -> { fps, frameCount, markers, default, parts, masks }
    this._stateFrame      = {}; // axis -> current resting/in-flight frame
    this._stateActive     = {}; // axis -> value currently at rest, or undefined
    this._stateTarget     = {}; // axis -> value currently targeted (resting or in-flight)
    this._statePlaybacks  = {}; // axis -> { dir, frame, toFrame, value }
  }

  /** Override a mask volume's soft-edge falloff distance (scene units). */
  setMaskFeather(name, value) {
    this._maskFeather[name] = value;
  }

  // ── Clips (product customization) ───────────────────────────────────────────

  /** Merges `partsMap` into every part contributed so far by loadClips() (see
   *  _clipPartsRegistry above), then loads the combined map — never the
   *  partial one alone, so this call can't wipe out another asset's (or this
   *  same asset's earlier-phase) parts. The actual loadParts() call is
   *  debounced a few ms: every loadParts() call resets each part's transform
   *  to identity for a frame (until the next animation tick re-applies it),
   *  so several calls landing close together (e.g. two assets' default-phase
   *  loads at boot, or a background variant fetch finishing mid-scroll) was
   *  what caused the visible flicker — coalescing them into one call using
   *  whatever's accumulated by the time it fires fixes that. Queued onto
   *  _clipPartsLoadQueue so non-coalesced calls still run strictly in order. */
  _mergeClipParts(partsMap) {
    Object.assign(this._clipPartsRegistry, partsMap);
    if (!this._clipPartsFlush) {
      this._clipPartsFlush = new Promise((resolve, reject) => {
        setTimeout(() => {
          this._clipPartsFlush = null;
          const snapshot = { ...this._clipPartsRegistry };
          // Run after whatever's ahead in the queue regardless of whether
          // that succeeded or failed — a single bad file 404ing must not
          // permanently stall every later merge for the rest of the session.
          const run = this._clipPartsLoadQueue.then(() => this.loadParts(snapshot), () => this.loadParts(snapshot));
          // The shared queue itself must never reject, or the *next* flush's
          // `.then` above would skip straight past its loadParts() call too.
          this._clipPartsLoadQueue = run.catch(() => {});
          run.then(resolve, reject);
        }, 30);
      });
    }
    return this._clipPartsFlush;
  }

  /**
   * Load an asset's clip file (see export_holosplat_asset.py) and merge its
   * clips into the live registry. Independent of the main Animation/timeline
   * entirely — can be called before or after the main animation loads, and
   * multiple asset files can be loaded (clip ids are expected to be unique
   * across all of them). Returns the list of clip ids this call added, so a
   * caller (e.g. the editor's asset list) can show what a given file
   * contains without needing the registry to track per-file origin itself.
   *
   * @param {string} url
   * @param {{splatsDir?: string, defaults?: Object<string,string>, lod?: number}} [opts]
   *   If splatsDir is given, this also resolves and loads the asset's own
   *   geometry (the JSON's "parts" field) via loadParts — defaults picks
   *   which variant value to use per axis (e.g. {color: "blue"}); a part
   *   with no variants of its own just loads its bare splat name. `lod`
   *   (see device-tier.js) probes for a `<name>.lods/<name>.lod{N}.spz`
   *   sibling of each part/variant and uses it instead when found.
   *   NOTE: loadParts() replaces the whole merged scene rather than
   *   appending to it, so this currently assumes the main scene has no
   *   parts of its own loaded separately — combining both isn't supported
   *   yet.
   */
  async loadClips(url, opts = {}) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HoloSplat: failed to load clips "${url}" (HTTP ${res.status})`);
    const data = await res.json();
    const ids = [];
    for (const c of data.clips ?? []) {
      const masks = (c.masks ?? []).map(m => ({
        name: m.name, softEdge: m.softEdge ?? 0.05, matrices: new Float32Array(m.matrices),
      }));
      this._clips[c.id] = {
        id:         c.id,
        fps:        c.fps ?? data.fps ?? 24,
        frameCount: c.frameCount,
        holdFrame:  c.holdFrame,
        objects:    (c.objects ?? []).map(o => ({ id: o.id, frames: new Float32Array(o.frames) })),
        masks,
      };
      ids.push(c.id);
      // Rest each of this clip's masks at their "out" (hidden) pose by
      // default — resolving the asset's default variant below (or a later
      // playClip()) promotes whichever one should actually be visible.
      for (const m of masks) {
        if (this._clipMaskState[m.name]) continue;
        const lastFrame = m.matrices.length / 16 - 1;
        this._clipMaskState[m.name]    = m.matrices.slice(lastFrame * 16, lastFrame * 16 + 16);
        this._clipMaskSoftEdge[m.name] = m.softEdge;
      }
    }

    // Axis transitions — one shared in/hold/out timeline per axis (see
    // playVariant()), as opposed to clips above (one timeline per clip id).
    for (const [axis, t] of Object.entries(data.transitions ?? {})) {
      const masks = (t.masks ?? []).map(m => ({
        name: m.name, value: m.value, softEdge: m.softEdge ?? 0.05, matrices: new Float32Array(m.matrices),
      }));
      this._transitions[axis] = {
        fps:        t.fps ?? data.fps ?? 24,
        frameCount: t.frameCount,
        holdFrame:  t.holdFrame,
        parts:      (t.parts ?? []).map(o => ({ id: o.id, value: o.value, frames: new Float32Array(o.frames) })),
        masks,
      };
      // Rest every value's masks at their "out" (hidden) pose by default —
      // resolving the asset's default variant below (or a later
      // playVariant()) promotes whichever value should actually be visible.
      for (const m of masks) {
        if (this._clipMaskState[m.name]) continue;
        const lastFrame = m.matrices.length / 16 - 1;
        this._clipMaskState[m.name]    = m.matrices.slice(lastFrame * 16, lastFrame * 16 + 16);
        this._clipMaskSoftEdge[m.name] = m.softEdge;
      }
    }

    // Asset states — one continuous timeline per axis (see playState()),
    // as opposed to transitions above (two values crossfading on a shared
    // in/hold/out range).
    for (const [axis, s] of Object.entries(data.states ?? {})) {
      const masks = (s.masks ?? []).map(m => ({
        name: m.name, softEdge: m.softEdge ?? 0.05, matrices: new Float32Array(m.matrices),
      }));
      this._states[axis] = {
        fps:        s.fps ?? data.fps ?? 24,
        frameCount: s.frameCount,
        markers:    s.markers ?? {},
        default:    s.default,
        parts:      (s.parts ?? []).map(o => ({ id: o.id, frames: new Float32Array(o.frames) })),
        masks,
      };
    }

    if (opts.splatsDir && data.parts && Object.keys(data.parts).length) {
      const dir = opts.splatsDir.replace(/\/?$/, '/');

      // Phase 1 — load only each part's default variant (per opts.defaults),
      // so the scene becomes visible without paying for every color up front.
      // LOD tiers are always .spz (see device-tier.js), so the explicit
      // .spz here is what resolveLodUrl needs to match against — it falls
      // back through .ply/.splat downstream (loadUrl) if no .spz exists.
      let defaultPartsMap = {};
      for (const [id, part] of Object.entries(data.parts)) {
        defaultPartsMap[id] = `${dir}${part.splatName}${defaultVariantSuffix(part, opts.defaults)}.spz`;
      }
      if (opts.lod) defaultPartsMap = await resolvePartsLod(defaultPartsMap, opts.lod);
      await this._mergeClipParts(defaultPartsMap);

      // Phase 2 — fetch every other variant in the background, then swap in
      // the full multi-slot scene so axis-transition masks (see playVariant)
      // have every color's geometry to fade between. Not awaited: this must
      // not block the caller (or the boot sequence) on color variants nobody
      // may ever select. _clipMaskState (set below/by playVariant) survives
      // the swap untouched, so whichever color is already showing stays put.
      // Delayed a few seconds rather than firing immediately: starting a
      // dozen more downloads + decompressions the instant the default scene
      // becomes visible competes with it for bandwidth and main-thread time
      // right when the page most needs to settle, which read as startup
      // stutter even though the default variant alone had already loaded.
      if (Object.values(data.parts).some(p => p.variants.length > 1)) {
        setTimeout(() => { (async () => {
          // Per-file mask culling is disabled (see _computeActiveRanges) —
          // every variant a part loads gets sorted and drawn every frame
          // forever, whether or not it's the currently visible color. The
          // active variant already loaded at full quality in Phase 1, so
          // the other (mostly-hidden, only ever partially revealed mid
          // color-swap transition) variants are capped at a low LOD tier
          // regardless of device tier — pure dead weight otherwise.
          const fullPartsMap = {};
          for (const [id, part] of Object.entries(data.parts)) {
            if (!part.variants.length) { fullPartsMap[id] = `${dir}${part.splatName}.spz`; continue; }
            const activeVariant = defaultVariantSuffix(part, opts.defaults).slice(1);
            fullPartsMap[id] = await Promise.all(part.variants.map(v => {
              const lod = v === activeVariant ? (opts.lod || 0) : Math.max(opts.lod || 0, BACKGROUND_VARIANT_LOD);
              return resolveLodUrl(`${dir}${part.splatName}.${v}.spz`, lod);
            }));
          }
          try { await this._mergeClipParts(fullPartsMap); }
          catch (e) { console.error(`[HoloSplat] background variant load failed for "${url}":`, e); }
        })(); }, CLIP_VARIANT_PREFETCH_DELAY_MS);
      }
    }

    // Resolve the initial active value per axis from opts.defaults, so the
    // default-selected variant's masks/pose start fully revealed instead of
    // at their resting "out" state — applied instantly (no animation).
    for (const [axis, value] of Object.entries(opts.defaults ?? {})) {
      if (this._transitions[axis]) this._setVariantInstant(axis, value);
      if (this._states[axis])      this._setStateInstant(axis, value);
    }
    // Any state axis with no explicit default in opts.defaults still needs
    // resolving to its own exported default value, so its parts/masks start
    // at a defined pose rather than wherever loadParts() last left them.
    for (const [axis, s] of Object.entries(this._states)) {
      if (this._stateActive[axis] === undefined && s.default != null) this._setStateInstant(axis, s.default);
    }

    return ids;
  }

  /** Remove specific clips from the registry by id (e.g. when an asset's row is removed in the editor). */
  unloadClips(ids) {
    for (const id of ids) delete this._clips[id];
  }

  /**
   * Trigger a clip by id ("[productName]-[variant]" — same string as the
   * triggering button's element id). Radio-group behavior is implicit: the
   * "product" (everything before the last hyphen) groups variants, so
   * selecting a new variant for a product currently holding a different one
   * first plays that old variant's "out" segment, then this one's "in" —
   * matching the in/hold/out marker convention (see export_holosplat_asset.py).
   */
  playClip(clipId) {
    const clip = this._clips[clipId];
    if (!clip) {
      console.warn(`[HoloSplat] playClip: unknown clip "${clipId}"`);
      return;
    }
    const cut = clipId.lastIndexOf('-');
    if (cut <= 0) {
      console.warn(`[HoloSplat] playClip: "${clipId}" doesn't match "productName-variant"`);
      return;
    }
    const product = clipId.slice(0, cut);
    const held    = this._clipHeld[product];
    if (held === clipId) return; // already showing this variant

    const heldClip = held ? this._clips[held] : null;
    this._clipPlaybacks[product] = heldClip
      ? { clip: heldClip, dir: 'out', frame: heldClip.holdFrame, nextClipId: clipId }
      : { clip,           dir: 'in',  frame: 0,                  nextClipId: null   };
    this._clipHeld[product] = clipId;
  }

  /** Advance every in-progress clip playback by dt seconds. */
  _tickClips(dt) {
    for (const product in this._clipPlaybacks) {
      const pb = this._clipPlaybacks[product];
      const { clip, dir } = pb;
      const endFrame = dir === 'in' ? clip.holdFrame : clip.frameCount - 1;
      pb.frame += dt * clip.fps;
      const done = pb.frame >= endFrame;
      if (done) pb.frame = endFrame;
      this._applyClipFrame(clip, pb.frame);
      if (!done) continue;
      if (dir === 'out' && pb.nextClipId) {
        this._clipPlaybacks[product] = { clip: this._clips[pb.nextClipId], dir: 'in', frame: 0, nextClipId: null };
      } else {
        delete this._clipPlaybacks[product];
      }
    }
  }

  /** Write a clip's interpolated object transforms + mask volumes at `frame` into the live state. */
  _applyClipFrame(clip, frame) {
    let dirty = false;
    for (const { id, pos, quat } of getClipObjectFrames(clip, frame)) {
      const slots = this._partIndex[id];
      if (slots && slots.length) {
        const m = quatPosToMat4(pos, quat);
        this._partLocalPose[id] = m;
        for (const slot of slots) this._partTransFlat.set(m, slot * 16);
        dirty = true;
      }
    }
    for (const { name, softEdge, matrix } of getClipMaskFrames(clip, frame)) {
      this._clipMaskState[name]    = matrix;
      this._clipMaskSoftEdge[name] = softEdge;
      this._sortDirty = true; // mask change affects active-subset culling
    }
    if (dirty) {
      this._renderer.updateTransforms(this._partTransFlat);
      this._sortDirty = true;
    }
  }

  /**
   * Switch an axis to `value` (e.g. playVariant('color', 'blue')) — every
   * part/mask tagged "<axis>=<value>" plays its .in→.hold segment while
   * whatever value was previously active for this axis plays .hold→.out,
   * simultaneously (see export_holosplat_asset.py's axis-transition export).
   * No-op if `value` is already active for `axis`.
   */
  playVariant(axis, value) {
    const t = this._transitions[axis];
    if (!t) {
      console.warn(`[HoloSplat] playVariant: unknown axis "${axis}"`);
      return;
    }
    const prevValue = this._axisActive[axis];
    if (prevValue === value) return;
    this._axisActive[axis] = value;
    this._transitionPlaybacks[axis] = { value, prevValue, elapsed: 0 };
  }

  /** Apply a value instantly at its .hold frame, with no previous value to
   *  play out — used to silently resolve an asset's default variant on load. */
  _setVariantInstant(axis, value) {
    const t = this._transitions[axis];
    if (!t) return;
    this._axisActive[axis] = value;
    this._applyTransitionValue(t, value, t.holdFrame);
  }

  /** Advance every in-progress axis transition by dt seconds. */
  _tickTransitions(dt) {
    for (const axis in this._transitionPlaybacks) {
      const pb = this._transitionPlaybacks[axis];
      const t  = this._transitions[axis];
      pb.elapsed += dt * t.fps;
      const enterFrame = Math.min(pb.elapsed, t.holdFrame);
      const exitFrame  = Math.min(t.holdFrame + pb.elapsed, t.frameCount - 1);
      this._applyTransitionValue(t, pb.value, enterFrame);
      if (pb.prevValue) this._applyTransitionValue(t, pb.prevValue, exitFrame);
      const done = enterFrame >= t.holdFrame && exitFrame >= t.frameCount - 1;
      if (done) delete this._transitionPlaybacks[axis];
    }
  }

  /** Write one value's interpolated part transforms + mask volumes at `frame` into the live state. */
  _applyTransitionValue(t, value, frame) {
    let dirty = false;
    for (const obj of t.parts) {
      if (obj.value !== value) continue;
      const slots = this._partIndex[obj.id];
      if (slots && slots.length) {
        const { pos, quat } = interpPosQuat(obj.frames, t.frameCount, frame);
        const m = quatPosToMat4(pos, quat);
        this._partLocalPose[obj.id] = m;
        for (const slot of slots) this._partTransFlat.set(m, slot * 16);
        dirty = true;
      }
    }
    for (const m of t.masks) {
      if (m.value !== value) continue;
      this._clipMaskState[m.name]    = interpMat4Frames(m.matrices, t.frameCount, frame);
      this._clipMaskSoftEdge[m.name] = m.softEdge;
      this._sortDirty = true;
    }
    if (dirty) {
      this._renderer.updateTransforms(this._partTransFlat);
      this._sortDirty = true;
    }
  }

  /**
   * Switch a state axis to `value` (e.g. playState('fold', 'folded')) —
   * seeks the axis's own timeline from wherever it currently sits to the
   * named marker's frame, forward or backward as needed (see
   * export_holosplat_asset.py's "state: <axis>=<value>" markers). No-op if
   * `value` is already the resting/targeted value for `axis`.
   */
  playState(axis, value) {
    const t = this._states[axis];
    if (!t) {
      console.warn(`[HoloSplat] playState: unknown state axis "${axis}"`);
      return;
    }
    if (!(value in t.markers)) {
      console.warn(`[HoloSplat] playState: axis "${axis}" has no value "${value}"`);
      return;
    }
    if (this._stateTarget[axis] === value) return;
    const toFrame   = t.markers[value];
    const fromFrame = this._stateFrame[axis] ?? toFrame;
    this._stateTarget[axis] = value;
    if (fromFrame === toFrame) {
      this._stateActive[axis] = value;
      delete this._statePlaybacks[axis];
      return;
    }
    this._statePlaybacks[axis] = { dir: toFrame > fromFrame ? 1 : -1, frame: fromFrame, toFrame, value };
  }

  /** Apply a value instantly at its marker frame, with no seek — used to
   *  silently resolve an asset's default state on load. */
  _setStateInstant(axis, value) {
    const t = this._states[axis];
    if (!t || !(value in t.markers)) return;
    const frame = t.markers[value];
    this._stateFrame[axis]  = frame;
    this._stateActive[axis] = value;
    this._stateTarget[axis] = value;
    delete this._statePlaybacks[axis];
    this._applyStateFrame(t, frame);
  }

  /** Advance every in-progress state seek by dt seconds. */
  _tickStates(dt) {
    for (const axis in this._statePlaybacks) {
      const pb = this._statePlaybacks[axis];
      const t  = this._states[axis];
      pb.frame += dt * t.fps * pb.dir;
      const done = pb.dir > 0 ? pb.frame >= pb.toFrame : pb.frame <= pb.toFrame;
      if (done) pb.frame = pb.toFrame;
      this._applyStateFrame(t, pb.frame);
      this._stateFrame[axis] = pb.frame;
      if (done) {
        this._stateActive[axis] = pb.value;
        delete this._statePlaybacks[axis];
      }
    }
  }

  /** Write a state axis's interpolated part transforms + mask volumes at `frame` into the live state. */
  _applyStateFrame(t, frame) {
    let dirty = false;
    for (const obj of t.parts) {
      const slots = this._partIndex[obj.id];
      if (slots && slots.length) {
        const { pos, quat } = interpPosQuat(obj.frames, t.frameCount, frame);
        const m = quatPosToMat4(pos, quat);
        this._partLocalPose[obj.id] = m;
        for (const slot of slots) this._partTransFlat.set(m, slot * 16);
        dirty = true;
      }
    }
    for (const m of t.masks) {
      this._clipMaskState[m.name]    = interpMat4Frames(m.matrices, t.frameCount, frame);
      this._clipMaskSoftEdge[m.name] = m.softEdge;
      this._sortDirty = true;
    }
    if (dirty) {
      this._renderer.updateTransforms(this._partTransFlat);
      this._sortDirty = true;
    }
  }

  /**
   * Resolve every state axis's target value from the main timeline's
   * "state: <asset>.<axis>=<value>" calls — same "most recently passed"
   * rule as _syncCameraMode, so seeking/scrubbing lands on the right value
   * without needing edge-triggered crossing detection. Axes with no call
   * yet passed fall back to their own exported default.
   */
  _syncAssetStates() {
    const calls = this._animation.stateCalls;
    if (!calls || !calls.length) return;
    const frame = this._animation.frame;
    const byAxis = {};
    for (const c of calls) {
      if (!this._states[c.axis]) continue;
      if (c.frame <= frame && (!byAxis[c.axis] || c.frame > byAxis[c.axis].frame)) byAxis[c.axis] = c;
    }
    for (const axis in this._states) {
      const call   = byAxis[axis];
      const target = call ? call.value : this._states[axis].default;
      if (target != null) this.playState(axis, target);
    }
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  async init() {
    await this._renderer.init();
    this._renderer.setSplatScale(this._splatScale);
    this._renderer.setAaDilation(this._aaDilation);
    this._camera.attach(this._canvas);
    this._observeResize();
    this._updateSize();
  }

  async load(url) {
    this._lastUrl   = url;
    this._lastParts = null;
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

    const { data, count, variants, shData, numSHBases } = await loadUrl(url, p => this._onProgress?.(p), this._shDegree);
    if (this._flipY) { flipYInPlace(data, count); flipYInPlaceSH(shData, numSHBases, count); }
    for (let i = 0; i < count; i++) data[i * 16 + 3] = 0;
    // Cached so setFlipY() (live-toggled, e.g. the examples/viewer.html
    // checkbox) can correctly re-flip SH data too, not just position/quat.
    this._lastShData    = shData;
    this._lastNumSHBases = numSHBases;
    this._gaussians      = data;
    this._numSplats      = count;
    this._depths         = new Float32Array(count);
    this._sort           = createSorter(count);
    this._partIndex      = {};
    this._partLocalPose  = {};
    this._partVariants   = variants ? { 0: { kind: 'spzv', variants, active: variants[0]?.name } } : {};
    this._fileNames      = [basenameNoExt(url)];
    this._partTransforms = [IDENTITY_MAT4.slice()];
    this._partTransFlat  = IDENTITY_MAT4.slice();
    // Single-file scenes never benefit from active-subset masking (there's
    // nothing else to render if this file is hidden) — leave _activeIdx
    // null so the render loop always takes the fast (full-scene) path.
    this._fileRanges = [[0, count]];
    this._fileAABB   = [null];
    this._activeIdx  = null;
    this._renderer.uploadGaussians(data, count);
    this._renderer.uploadTransforms(this._partTransforms);
    this._renderer.uploadSH(shData || null, numSHBases || 0);
    this._renderer.setShDegree(numSHBases > 0 ? this._shDegree : 0);
    this._sortDirty = true;
    this._camera.fitScene(data, count);
    if (this._animation) {
      const { eye, target } = this._animation.getCameraFrame();
      this._camera.setFromLookAt(eye, target);
    }
    this._buildPartVolumeMask();
    this._sceneReady = true;
  }

  async _loadPlyStreamSingle(url) {
    const { numVertices, numSHBases, consume } = await openPlyStream(url, p => this._onProgress?.(p * 0.02), this._shDegree);

    const zeros = new Float32Array(numVertices * 16);
    this._gaussians      = zeros;
    this._numSplats      = numVertices;
    this._depths         = new Float32Array(numVertices);
    this._sort           = createSorter(numVertices);
    this._partIndex      = {};
    this._partLocalPose  = {};
    this._partVariants   = {};
    this._fileNames      = [basenameNoExt(url)];
    this._partTransforms = [IDENTITY_MAT4.slice()];
    this._partTransFlat  = IDENTITY_MAT4.slice();
    this._fileRanges = [[0, numVertices]];
    this._fileAABB   = [null];
    this._activeIdx  = null;
    this._renderer.uploadGaussians(zeros, numVertices);
    this._renderer.uploadTransforms(this._partTransforms);
    this._renderer.allocateSH(numVertices, numSHBases);
    this._renderer.setShDegree(numSHBases > 0 ? this._shDegree : 0);
    this._buildPartVolumeMask();
    if (this._animation) {
      const { eye, target } = this._animation.getCameraFrame();
      this._camera.setFromLookAt(eye, target);
    }

    // Accumulated alongside this._gaussians so setFlipY() (live-toggled —
    // see examples/viewer.html's checkbox) can re-flip SH data too, the same
    // way it already re-flips this._gaussians.
    this._lastShData     = numSHBases > 0 ? new Float32Array(numVertices * numSHBases * 3) : null;
    this._lastNumSHBases = numSHBases;

    let vOff = 0;
    await consume((chunk, nVerts, shChunk) => {
      for (let j = 0; j < nVerts; j++) chunk[j * 16 + 3] = 0;
      if (this._flipY) { flipYInPlace(chunk, nVerts); flipYInPlaceSH(shChunk, numSHBases, nVerts); }
      this._renderer.patchGaussians(chunk, vOff);
      this._renderer.patchSH(shChunk, vOff);
      this._gaussians.set(chunk, vOff * 16);
      if (this._lastShData && shChunk) this._lastShData.set(shChunk, vOff * numSHBases * 3);
      vOff += nVerts;
    }, p => this._onProgress?.(p));

    if (!this._animation) this._camera.fitScene(this._gaussians, this._numSplats);
    this._sceneReady = true;
  }

  /** loadUrl(), but cached per URL+shDegree on this viewer instance — see
   *  _urlCache above. Callers (loadParts()'s task loop) mutate the returned
   *  data in place (flipY, slot tagging), so a cache hit must hand back a
   *  fresh copy, never the same typed-array instance twice. */
  async _loadUrlCached(url, onProgress, shDegree) {
    const key = `${url}#${shDegree}`;
    let entry = this._urlCache.get(key);
    if (!entry) {
      entry = loadUrl(url, onProgress, shDegree);
      this._urlCache.set(key, entry);
      entry.catch(() => this._urlCache.delete(key)); // don't cache failures
    } else {
      onProgress?.(1);
    }
    const result = await entry;
    return {
      ...result,
      data: result.data.slice(),
      shData: result.shData ? result.shData.slice() : result.shData,
    };
  }

  /**
   * Load multiple splat parts and merge them into a single GPU scene.
   *
   * @param {Object<string, string|string[]|{url: string, variants: string[]}>} partsMap
   *   Map of object id → splat file URL, an array of URLs, or
   *   `{ url, variants }`. All files for a part share that part's animated
   *   transform.
   *   - A plain URL array loads every file as its own slot, rendered
   *     together — useful for unmasked color variants until per-color masks
   *     select between them.
   *   - `{ url, variants }` loads only `url` (one of `<base>.<v>.<ext>` for
   *     `v` in `variants`); the others are fetched lazily by setVariant() —
   *     use this for independently-trained per-color variants whose geometry
   *     differs, where loading every variant up front would be wasteful.
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
    this._lastParts = partsMap;
    this._lastUrl   = null;
    const partIds = Object.keys(partsMap);
    if (partIds.length === 0) throw new Error('HoloSplat: loadParts called with empty map');

    this._sceneReady = false;

    // Flatten to one task per splat file, each getting its own transform
    // slot. A part may reference multiple files (e.g. unmasked color
    // variants, all rendered together until per-color masks are set up) —
    // slots belonging to the same part are kept in sync every animation
    // tick, so they animate together.
    const tasks = [];
    partIds.forEach((id) => {
      const entry = partsMap[id];
      // { url, variants } — single file loaded now, with sibling
      // "<base>.<variant>.<ext>" files fetched lazily by setVariant().
      if (entry && typeof entry === 'object' && !Array.isArray(entry)) {
        tasks.push({ id, slot: tasks.length, url: entry.url, variantNames: entry.variants });
        return;
      }
      const urls = Array.isArray(entry) ? entry : [entry];
      for (const url of urls) tasks.push({ id, slot: tasks.length, url });
    });

    const progresses = new Array(tasks.length).fill(0);
    const reportProg = () => {
      if (this._onProgress)
        this._onProgress(progresses.reduce((a, b) => a + b, 0) / tasks.length);
    };

    // Phase 1 — open PLY streams (header only) and fully load non-PLY parts.
    // All happen in parallel; every task resolves with its vertex count.
    const parts = await Promise.all(tasks.map(async ({ id, slot, url }, i) => {
      const ext = url.split('?')[0].split('.').pop().toLowerCase();
      if (ext === 'ply') {
        try {
          const { numVertices, numSHBases, consume } = await openPlyStream(url, p => {
            progresses[i] = p * 0.03; reportProg();
          }, this._shDegree);
          return { id, slot, kind: 'stream', numVertices, numSHBases, consume };
        } catch (e) {
          if (!/^HTTP 4/.test(e.message)) throw e;
          // 404 — let loadUrl try other extensions
        }
      }
      const { data, count, variants, shData, numSHBases } = await this._loadUrlCached(url, p => {
        progresses[i] = p * 0.9; reportProg();
      }, this._shDegree);
      progresses[i] = 0.9;
      return { id, slot, kind: 'loaded', data, count, variants, shData, numSHBases };
    }));

    // Compute total vertex count and per-task offsets in the merged buffer.
    const counts     = parts.map(p => p.kind === 'stream' ? p.numVertices : p.count);
    const total      = counts.reduce((a, b) => a + b, 0);
    const vtxOffsets = [];
    { let off = 0; for (const c of counts) { vtxOffsets.push(off); off += c; } }

    // Build the merged CPU buffer.
    // Non-PLY parts: fill with real data. PLY stream parts: stay as zeros (transparent).
    const merged = new Float32Array(total * 16);
    const fileAABB = new Array(tasks.length).fill(null);
    parts.forEach((part, i) => {
      if (part.kind !== 'loaded') return;
      const { data, count, slot, shData, numSHBases } = part;
      for (let j = 0; j < count; j++) data[j * 16 + 3] = slot;
      if (this._flipY) { flipYInPlace(data, count); flipYInPlaceSH(shData, numSHBases, count); }
      merged.set(data, vtxOffsets[i] * 16);
      fileAABB[slot] = computeAABB(data, count);
      progresses[i] = 1;
    });

    // Initialise viewer state and upload the buffer (zeros for stream parts).
    // Each loaded file gets its own transform slot; _partIndex maps a part id
    // to every slot that should follow that part's animated transform (more
    // than one when a part has multiple color variants loaded). _fileNames
    // gives each slot's splat name, used by _buildPartVolumeMask to match
    // mask-volume naming conventions.
    this._partIndex = {};
    partIds.forEach(id => { this._partIndex[id] = []; });
    tasks.forEach(({ id, slot }) => { this._partIndex[id].push(slot); });
    this._fileNames = tasks.map(t => basenameNoExt(t.url));

    // Two ways a slot gets runtime-swappable variants — see setVariant():
    //  - 'spzv': a packed file carries every variant's color/alpha palette
    //    alongside one shared geometry — swap is instant, no fetch.
    //  - 'file': the partsMap entry was { url, variants }, where `url` is
    //    "<base>.<active>.<ext>" and the other names are sibling files with
    //    their own geometry, fetched (and cached) on first use.
    this._partVariants = {};
    parts.forEach(part => {
      if (part.variants) {
        this._partVariants[part.slot] = { kind: 'spzv', variants: part.variants, active: part.variants[0]?.name };
      }
    });
    tasks.forEach(({ slot, url, variantNames }) => {
      if (!variantNames?.length || this._partVariants[slot]) return;
      const clean = url.split('?')[0];
      const m = clean.match(/^(.*)\.([^./]+)\.(spz|ply|splat)$/i);
      if (!m || !variantNames.includes(m[2])) return;
      this._partVariants[slot] = {
        kind: 'file', names: variantNames, active: m[2],
        baseUrl: m[1], ext: m[3], cache: {},
      };
    });

    this._partTransforms = tasks.map(() => IDENTITY_MAT4.slice());
    this._partTransFlat  = new Float32Array(tasks.length * 16);
    for (let i = 0; i < tasks.length; i++) this._partTransFlat.set(IDENTITY_MAT4, i * 16);

    // Per-file gaussian ranges and bounding boxes, for the active-subset
    // AABB-vs-mask-volume test in _computeActiveRanges (skips fully-masked
    // files entirely so they cost nothing in sort/draw).
    this._fileRanges = vtxOffsets.map((off, i) => [off, off + counts[i]]);
    this._fileAABB   = fileAABB;
    this._activeIdx  = new Uint32Array(total);

    this._gaussians = merged;
    this._numSplats = total;
    this._depths    = new Float32Array(total);
    this._sort      = createSorter(total);

    this._renderer.uploadGaussians(merged, total);
    this._renderer.uploadTransforms(this._partTransforms);

    // SH: find the max numSHBases across all parts; pre-allocate the GPU SH
    // buffer (zeroed), then fill in the non-streaming parts' data immediately.
    const maxSHBases = parts.reduce((m, p) => Math.max(m, p.numSHBases || 0), 0);
    this._renderer.allocateSH(total, maxSHBases);
    this._renderer.setShDegree(maxSHBases > 0 ? this._shDegree : 0);
    parts.forEach((part, i) => {
      if (part.kind === 'loaded' && part.shData && part.numSHBases === maxSHBases) {
        this._renderer.patchSH(part.shData, vtxOffsets[i]);
      }
    });

    this._buildPartVolumeMask();
    this._camera.fitScene(merged, total);
    if (this._animation) {
      const { eye, target } = this._animation.getCameraFrame();
      this._camera.setFromLookAt(eye, target);
    }
    reportProg();

    // Phase 2 — stream PLY vertex data, patching the GPU buffer as chunks arrive.
    const streamParts = parts
      .map((part, i) => ({ part, i }))
      .filter(({ part }) => part.kind === 'stream');
    if (streamParts.length === 0) {
      this._sceneReady = true;
      this._prefetchVariants();
      return;
    }

    await Promise.all(streamParts.map(async ({ part, i }) => {
      const { slot, consume } = part;
      let vOff = vtxOffsets[i];

      await consume((chunk, nVerts, shChunk) => {
        for (let j = 0; j < nVerts; j++) chunk[j * 16 + 3] = slot;
        if (this._flipY) { flipYInPlace(chunk, nVerts); flipYInPlaceSH(shChunk, part.numSHBases, nVerts); }
        this._renderer.patchGaussians(chunk, vOff);
        if (shChunk && part.numSHBases === maxSHBases) this._renderer.patchSH(shChunk, vOff);
        this._gaussians.set(chunk, vOff * 16);
        this._fileAABB[slot] = extendAABB(this._fileAABB[slot], chunk, nVerts);
        vOff += nVerts;
      }, p => {
        progresses[i] = 0.03 + 0.97 * p;
        reportProg();
      });

      progresses[i] = 1;
      reportProg();
    }));

    if (!this._animation) this._camera.fitScene(this._gaussians, this._numSplats);
    this._sceneReady = true;
    this._prefetchVariants();
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
    clearTimeout(this._resizeTimer);
  }

  // ── Configuration setters (can be called at runtime) ──────────────────────

  setBackground(bg) { this._renderer.setBackground(bg); }

  setSplatScale(s) {
    this._splatScale = s;
    this._renderer.setSplatScale(s);
  }

  setGamma(g) {
    this._renderer.setGamma(g);
  }

  async setShDegree(n) {
    if (n === this._shDegree) return;
    this._shDegree = n;
    if (this._lastParts) await this.loadParts(this._lastParts);
    else if (this._lastUrl) await this.load(this._lastUrl);
  }

  setAaDilation(v) {
    this._aaDilation = v;
    this._renderer.setAaDilation(v);
    this._sortDirty = true;
  }

  /** TEMP debug instrumentation — see src/shaders.js binding(7). */
  setDebugIndex(i) { this._renderer.setDebugIndex(i); }
  async readDebug() { return this._renderer.readDebug(); }

  setAutoRotate(v) { this._autoRotate = v; }

  setFlipY(enabled) {
    if (!!enabled === this._flipY) return;
    this._flipY = !!enabled;
    if (this._gaussians) {
      flipYInPlace(this._gaussians, this._numSplats);
      this._renderer.uploadGaussians(this._gaussians, this._numSplats);
      this._camera.fitScene(this._gaussians, this._numSplats);
      this._sortDirty = true;
      // Only covers the single-file load path (loadParts doesn't cache SH
      // data per-part) — see _lastShData.
      if (this._lastShData) {
        flipYInPlaceSH(this._lastShData, this._lastNumSHBases, this._numSplats);
        this._renderer.uploadSH(this._lastShData, this._lastNumSHBases);
      }
    }
  }

  /**
   * Snapshot of live performance/quality numbers, for the `?hs` mobile stats
   * overlay (see holosplat/stats.js) and any other diagnostics UI.
   * @returns {{fps: number, numSplats: number, activeSplats: number,
   *            shDegree: number, pixelRatio: number, sceneName: string|null,
   *            gpuSort: boolean, gpuSortFailed: boolean, tier: string|null}}
   */
  getStats() {
    return {
      fps:           this._frameTimeEMA ? 1 / this._frameTimeEMA : 0,
      numSplats:     this._numSplats,
      activeSplats:  this._lastActiveCount || this._numSplats,
      shDegree:      this._shDegree,
      pixelRatio:    this._effectivePixelRatio,
      sceneName:     this._currentSceneName,
      // _gpuSort is the requested config; _gpuSortFailed flips true if a GPU
      // validation error forced a silent fallback to CPU sort (see
      // renderer.js's uncapturederror handler) — surface both so "is it
      // actually using the GPU" is answerable from the stats overlay alone.
      gpuSort:       this._gpuSort && !this._renderer._gpuSortFailed,
      gpuSortFailed: this._gpuSort && this._renderer._gpuSortFailed,
      tier:          this._tier,
    };
  }

  /** TEMP debug helper — returns the CPU-decoded canonical layout for one
   *  gaussian, for cross-checking against hand-computed values from the raw
   *  source file. See src/viewer.js's 16-floats/gaussian layout comment. */
  getSplatDebug(index) {
    if (!this._gaussians || index < 0 || index >= this._numSplats) return null;
    const d = index * 16;
    const g = this._gaussians;
    return {
      pos:   [g[d], g[d+1], g[d+2]],
      part:  g[d+3],
      color: [g[d+4], g[d+5], g[d+6], g[d+7]],
      scale: [g[d+8], g[d+9], g[d+10]],
      quat:  [g[d+12], g[d+13], g[d+14], g[d+15]],
    };
  }

  /**
   * List the color/material variant names available for a part (or the
   * whole scene, for a single-file load — pass no id). Empty if the part
   * has no variants.
   * @param {string} [id]  part id, as passed to loadParts()
   * @returns {string[]}
   */
  getVariants(id) {
    const slots = id == null ? [0] : this._partIndex[id];
    if (!slots) return [];
    const pv = this._partVariants[slots[0]];
    if (!pv) return [];
    return pv.kind === 'spzv' ? pv.variants.map(v => v.name) : pv.names;
  }

  /**
   * Switch the active color/material variant for a part.
   *
   * - For parts loaded from a packed `.spzv` file (shared geometry, one
   *   palette per variant — see examples/pack.html), the swap is instant:
   *   no fetch, no reload, geometry is stored once.
   * - For parts loaded as `{ url, variants }` (each variant is its own file
   *   with its own geometry — the common case for independently-trained
   *   color variants), the target variant's file is fetched on first use
   *   (cached after that) and that part's slice of the scene is rebuilt and
   *   re-uploaded. Other variants are prefetched in the background after the
   *   scene loads, so repeat swaps are instant.
   *
   * @param {string} id     part id, as passed to loadParts() (omit for a single-file load)
   * @param {string} name   one of the names returned by getVariants()
   * @returns {Promise<boolean>}  true if the variant was found and applied
   */
  async setVariant(id, name) {
    const slots = id == null ? [0] : this._partIndex[id];
    if (!slots) return false;

    let applied = false;
    for (const slot of slots) {
      const pv = this._partVariants[slot];
      if (!pv) continue;

      if (pv.kind === 'spzv') {
        const variant = pv.variants.find(v => v.name === name);
        if (!variant) continue;

        const [start, end] = this._fileRanges[slot];
        for (let i = start; i < end; i++) {
          const d = i * 16, p = (i - start) * 4;
          this._gaussians[d + 4] = variant.palette[p + 0];
          this._gaussians[d + 5] = variant.palette[p + 1];
          this._gaussians[d + 6] = variant.palette[p + 2];
          this._gaussians[d + 7] = variant.palette[p + 3];
        }
        this._renderer.patchGaussians(this._gaussians.subarray(start * 16, end * 16), start);
        pv.active = name;
        applied = true;
        continue;
      }

      // kind === 'file' — each variant is its own file with its own geometry.
      if (pv.active === name) { applied = true; continue; }
      if (!pv.names.includes(name)) continue;
      await this._swapPartVariant(slot, pv, name);
      applied = true;
    }
    return applied;
  }

  /** Fetch (or reuse a cached) "<base>.<name>.<ext>" file for a 'file'-kind
   *  variant slot, replace that slot's geometry+color in the merged scene,
   *  and re-upload. Handles a different splat count than the slot currently
   *  has by resizing the merged buffer and shifting later slots' ranges. */
  async _swapPartVariant(slot, pv, name) {
    let entry = pv.cache[name];
    if (!entry) {
      entry = await loadUrl(`${pv.baseUrl}.${name}.${pv.ext}`);
      pv.cache[name] = entry;
    }

    const [start, end] = this._fileRanges[slot];
    const oldCount = end - start;
    const newCount = entry.count;
    const delta    = newCount - oldCount;

    const slotData = entry.data.slice();
    for (let j = 0; j < newCount; j++) slotData[j * 16 + 3] = slot;
    if (this._flipY) flipYInPlace(slotData, newCount);

    if (delta === 0) {
      this._gaussians.set(slotData, start * 16);
      this._renderer.patchGaussians(slotData, start);
    } else {
      const newTotal = this._numSplats + delta;
      const merged = new Float32Array(newTotal * 16);
      merged.set(this._gaussians.subarray(0, start * 16), 0);
      merged.set(slotData, start * 16);
      merged.set(this._gaussians.subarray(end * 16), (start + newCount) * 16);

      for (let s = 0; s < this._fileRanges.length; s++) {
        const [a, b] = this._fileRanges[s];
        if (s === slot) this._fileRanges[s] = [start, start + newCount];
        else if (a >= end) this._fileRanges[s] = [a + delta, b + delta];
      }

      this._gaussians = merged;
      this._numSplats = newTotal;
      this._depths    = new Float32Array(newTotal);
      this._sort      = createSorter(newTotal);
      this._activeIdx = new Uint32Array(newTotal);
      this._renderer.uploadGaussians(this._gaussians, this._numSplats);
    }

    this._fileAABB[slot]  = computeAABB(slotData, newCount);
    this._fileNames[slot] = basenameNoExt(`${pv.baseUrl}.${name}.${pv.ext}`);
    pv.active = name;
    this._buildPartVolumeMask();
  }

  /** Background-fetch every non-active 'file'-kind variant after the scene is
   *  ready, so later setVariant() calls don't pay a network round trip.
   *  Fetched sequentially (low priority); failures are logged, not thrown. */
  async _prefetchVariants() {
    if (!this._prefetchVariantsEnabled) return;
    for (const pv of Object.values(this._partVariants)) {
      if (pv.kind !== 'file') continue;
      for (const name of pv.names) {
        if (name === pv.active || pv.cache[name]) continue;
        try {
          pv.cache[name] = await loadUrl(`${pv.baseUrl}.${name}.${pv.ext}`);
        } catch (e) {
          console.warn(`[HoloSplat] variant prefetch failed for "${pv.baseUrl}.${name}.${pv.ext}": ${e.message}`);
        }
      }
    }
  }


  /** Freeze / unfreeze animation playback. When paused, camera responds to user input. */
  setAnimationPaused(paused) { this._animPaused = paused; }

  /** Used by scroll-scene freecamera acts to hand the camera fully to the user.
   *  On exit, smoothly blends back to the animation path.
   *  (hs-* markers use the overlay system instead — see _syncCameraMode.) */
  setCameraFree(v) {
    const wasActive   = this._cameraFree;
    this._cameraFree  = !!v;
    if (v) {
      this._blendBack = null;
      this._camera.disableZoom();
      // Explore acts own single-finger touch for orbiting — stop passing it
      // through to page scroll while active.
      if (this.animTickOverride) this._camera.allowTouchScroll = false;
    } else {
      this._camera.enableZoom();
      if (this.animTickOverride) this._camera.allowTouchScroll = true;
      if (wasActive && this._animation) {
        this._blendBack = {
          fromEye:    this._camera.eye.slice(),
          fromTarget: this._camera.target.slice(),
          t:          0,
          duration:   0.5,
        };
      }
    }
  }

  /**
   * Detect the active hs-* marker and apply its scene config's pan/zoom
   * overlay settings on top of the animation base.
   *
   * NOTE: orbit/follow-mouse support was removed here (cfg.orbit) as part of
   * a deliberate cleanup — it will be re-added slowly later.
   */
  _syncCameraMode() {
    const markers = this._animation.markers;
    const frame   = this._animation.frame;

    // Find the most recently passed marker.
    let activeMarker = null;
    let maxFrame     = -1;
    for (const [name, mf] of Object.entries(markers)) {
      if (mf <= frame && mf > maxFrame) { maxFrame = mf; activeMarker = name; }
    }

    // Read scene config — editor overlay takes priority, then linked DOM element.
    // Fall back to parsing the marker name for backwards compatibility.
    let cfg = null;
    if (activeMarker) {
      // Live editor config (set by editor overlay; available without a linked element)
      cfg = window.__hsSceneConfigs?.[activeMarker] ?? null;

      // Legacy fallback: parse hs-* tokens from marker name
      if (!cfg) {
        const hsMode = extractHsMode(activeMarker);
        if (hsMode && hsMode !== 'hs-locked') {
          const zm = hsMode.match(/zoom-(\d+)/);
          cfg = {
            zoom: { enabled: !!zm, mode: 'limited', range: zm ? +zm[1] : 25 },
          };
        }
      }
    }

    // Build a stable mode key for change detection
    const newMode = cfg ? JSON.stringify(cfg) : null;
    if (newMode === this._camMode) return;
    this._camMode = newMode;

    // Structural type key — pan/zoom on-off only.
    const newModeType = cfg
      ? `${+!!(cfg.pan?.enabled)}:${+!!(cfg.zoom?.enabled)}`
      : null;
    const typeChanged = newModeType !== this._camModeType;
    this._camModeType = newModeType;

    if (cfg) {
      const pan  = cfg.pan  || {};
      const zoom = cfg.zoom || {};

      if (typeChanged) {
        this._panOffset  = [0, 0, 0];
        this._zoomFactor = 1;
      }

      // Zoom — overlay-based (see _zoomFactor): the animation drives radius every
      // frame, so wheel/pinch input is routed through zoomDeltaCallback as a
      // multiplier applied on top of the animation radius (see _tick).
      if (zoom.enabled) {
        this._camera.zoomEnabled  = true;
        this._zoomLimit = zoom.limited ? Math.max(0, (zoom.range ?? 500) / 100) : null;
        // Bring the current zoom back inside the new bounds immediately
        // (e.g. when "limited" is turned on while already zoomed out further).
        if (this._zoomLimit !== null) {
          this._zoomFactor = Math.max(1 - this._zoomLimit, Math.min(1 + this._zoomLimit, this._zoomFactor));
        }
        this._camera.zoomDeltaCallback = (factor) => {
          let f = this._zoomFactor * factor;
          if (this._zoomLimit !== null) f = Math.max(1 - this._zoomLimit, Math.min(1 + this._zoomLimit, f));
          this._zoomFactor = Math.max(0.01, f);
        };
      } else {
        this._camera.zoomEnabled = false;
        this._camera.zoomDeltaCallback = null;
        this._zoomFactor = 1;
      }

      // Pan — overlay-based (see _panOffset): the animation drives target every
      // frame, so drag input is routed through panDeltaCallback as a world-space
      // delta accumulated on top of the animation target (see _tick).
      if (pan.enabled) {
        this._camera.panEnabled = true;
        this._camera.panSpeed   = 1 - Math.min(100, Math.max(0, pan.damping ?? 0)) / 100;
        this._camera.panButton  = pan.button === 'left' ? 0 : 2;
        // pan.radius is a percentage of the camera's orbit radius (like zoom.range),
        // so the limit scales with the scene instead of being a fixed world-unit value.
        this._panLimit = pan.limited
          ? Math.max(0, (pan.radius ?? 500) / 100) * this._camera.radius
          : null;
        // Bring the current pan offset back inside the new bounds immediately
        // (e.g. when "limited" is turned on while already panned further out).
        if (this._panLimit !== null) {
          const off = this._panOffset;
          const mag = Math.hypot(off[0], off[1], off[2]);
          if (mag > this._panLimit) {
            const f = this._panLimit / mag;
            off[0] *= f; off[1] *= f; off[2] *= f;
          }
        }
        this._camera.panDeltaCallback = (dx, dy, dz) => {
          const off = this._panOffset;
          off[0] += dx; off[1] += dy; off[2] += dz;
          if (this._panLimit !== null) {
            const mag = Math.hypot(off[0], off[1], off[2]);
            if (mag > this._panLimit) {
              const f = this._panLimit / mag;
              off[0] *= f; off[1] *= f; off[2] *= f;
            }
          }
        };
      } else {
        this._camera.panEnabled = false;
        this._camera.panDeltaCallback = null;
        this._panOffset = [0, 0, 0];
      }
    } else {
      this._panOffset  = [0, 0, 0];
      this._zoomFactor = 1;
      this._camera.panDeltaCallback   = null;
      this._camera.zoomDeltaCallback  = null;
      this._camera.panEnabled  = !this._animation.focalPoint;
      this._camera.panSpeed    = 1;
      this._camera.panButton   = 2;
      this._camera.panRadius   = null;
      this._camera.panOrigin   = null;
      this._camera.enableZoom();
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

  /** Build per-file volume bitmasks and upload to GPU.
   *  Call after loading parts or after attaching an animation with volumes.
   *
   *  Mask volume names follow one of two conventions (matched against each
   *  loaded splat file's name, e.g. "headphones.headband.yellow"):
   *    - "<prefix>.mask"            → affects any file equal to <prefix> or
   *                                    starting with "<prefix>.". E.g.
   *                                    "headphones.mask" affects every
   *                                    "headphones.*" file (all parts, all
   *                                    variants).
   *    - "<prefix>..<suffix>.mask"  → affects any file starting with
   *                                    "<prefix>." AND ending with
   *                                    ".<suffix>". E.g. "headphones..yellow.mask"
   *                                    affects "headphones.headband.yellow",
   *                                    "headphones.cup.left.yellow", etc. —
   *                                    every "yellow" variant across all parts.
   */
  _buildPartVolumeMask() {
    // Clip masks and axis-transition masks (see loadClips()) are independent
    // of this._animation's own volumes — they exist for an asset's
    // button-triggered transitions, not the main scroll timeline — but use
    // the same file-name matching, so they're merged into the same bitmask.
    const clipMaskNames = new Set();
    for (const clip of Object.values(this._clips)) {
      for (const m of clip.masks ?? []) clipMaskNames.add(m.name);
    }
    for (const t of Object.values(this._transitions)) {
      for (const m of t.masks ?? []) clipMaskNames.add(m.name);
    }
    for (const t of Object.values(this._states)) {
      for (const m of t.masks ?? []) clipMaskNames.add(m.name);
    }
    const vols = [...(this._animation?.volumes ?? []), ...[...clipMaskNames].map(name => ({ name }))];
    const fileNames = this._fileNames ?? [];
    const count = Math.max(fileNames.length, 1);
    const masks = new Uint32Array(count);
    vols.forEach((vol, vi) => {
      const prefix = vol.name;
      let matched = 0;
      for (let fi = 0; fi < fileNames.length; fi++) {
        if (matchesMaskPrefix(fileNames[fi], prefix)) {
          masks[fi] |= (1 << vi);
          matched++;
        }
      }
      if (matched === 0 && fileNames.length > 0) {
        console.warn(`[HoloSplat] mask volume "${prefix}" matched 0 of ${fileNames.length} file(s) — check naming convention`);
      }
    });
    this._fileMasks = masks;
    this._renderer.uploadPartVolumeMask(masks);
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
    this._camera.setFromLookAt(eye, target);
    this._buildPartVolumeMask();
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

    // Rising edge of _sceneReady: the page can stay genuinely busy for
    // several seconds *after* the scene lands too — decoding/prefetching
    // additional assets (clips, color variants), first-time draw/sort of
    // the full splat count, etc. None of that reflects steady-state device
    // performance, so it shouldn't get baked into "this device is slow" and
    // trigger a pixelRatio drop — ramp-down is 0.85x/s but ramp-up is only
    // 1.05x/s (see _updateAdaptiveQuality), so a drop earned in 2-3s of
    // startup churn takes 15-20s to claw back, which reads as sustained
    // blur. Reset the EMA and hold quality at max for a grace window once
    // ready; a longer preload-feeling startup beats a visibly degrading
    // render.
    if (this._sceneReady && !this._wasSceneReady) {
      this._frameTimeEMA      = null;
      this._qualityWarmupUntil = now + QUALITY_WARMUP_MS;
    }
    this._wasSceneReady = this._sceneReady;

    // PLY parts stream in progressively: the merged buffer (real data for
    // non-PLY parts, zeros for streaming ones) is uploaded and drawn every
    // frame from the start, well before _sceneReady — which only flips once
    // every stream has fully finished decoding (see loadParts). That decode
    // (chunked across many files, each yielding to the main thread) runs
    // concurrently with this same render loop for the entire streaming
    // window, which is the real source of sustained slow frames the warm-up
    // above never sees because it only starts counting once _sceneReady
    // goes true. Suspend degradation for the whole !_sceneReady window too.
    const warmingUp = !this._sceneReady || now < this._qualityWarmupUntil;
    this._updateAdaptiveQuality(dt, warmingUp);

    const w = this._canvas.width;
    const h = this._canvas.height;

    // Mask-volume frames for this tick, used by _computeActiveRanges below
    // to restrict sort/draw to the active subset (mask hiding + frustum cull).
    let volFrames = [];
    let transformsDirty = false;

    // Clip playback (product customization) is user-triggered and fully
    // independent of the main Animation/timeline — it doesn't require one
    // to exist at all, and ticks regardless of _animPaused/_sceneReady.
    if (Object.keys(this._clipPlaybacks).length > 0) this._tickClips(dt);
    if (Object.keys(this._transitionPlaybacks).length > 0) this._tickTransitions(dt);
    if (Object.keys(this._statePlaybacks).length > 0) this._tickStates(dt);

    if (this._animation) {
      if (!this._animPaused && this._sceneReady) {
        if (this.animTickOverride) this.animTickOverride(dt);
        else                       this._animation.tick(dt);
      }

      // Update camera from animation BEFORE _syncCameraMode so that when a
      // mode transition fires (e.g. scrubbing into an hs zone), the camera is
      // already at the correct frame position — no stale-frame snap.
      if (!this._cameraFree) {
        const { eye, target } = this._animation.getCameraFrame();
        if (this._blendBack) {
          // Smooth blend from scroll-scene freecamera position back to animation
          this._blendBack.t += dt;
          const alpha = smoothstep(Math.min(this._blendBack.t / this._blendBack.duration, 1));
          this._camera.setFromLookAt(
            lerp3(this._blendBack.fromEye,    eye,    alpha),
            lerp3(this._blendBack.fromTarget, target, alpha),
          );
          if (this._blendBack.t >= this._blendBack.duration) this._blendBack = null;
        } else {
          // NOTE: orbit/follow-mouse + focal-point look-target blending was
          // removed here as part of a deliberate cleanup — it will be
          // re-added slowly later. This currently just follows the
          // animation's own eye/target (optionally crossfaded with an
          // adjacent scene via _sceneBlend), then applies pan/zoom overlays.
          let finalEye = eye, finalTarget = target;
          if (this._sceneBlend) {
            const { otherEye, otherTarget, bf } = this._sceneBlend;
            finalEye    = lerp3(otherEye,    eye,    bf);
            finalTarget = lerp3(otherTarget, target, bf);
          }
          this._camera.setFromLookAt(finalEye, finalTarget);
          // Pan/zoom overlays: user-dragged target offset and scroll-wheel radius
          // multiplier on top of the animation base (see panDeltaCallback/zoomDeltaCallback).
          if (this._panOffset[0] !== 0 || this._panOffset[1] !== 0 || this._panOffset[2] !== 0) {
            this._camera.target[0] += this._panOffset[0];
            this._camera.target[1] += this._panOffset[1];
            this._camera.target[2] += this._panOffset[2];
          }
          if (this._zoomFactor !== 1) {
            this._camera.radius *= this._zoomFactor;
          }
        }
      }

      // Runs every tick (cheap — early-returns when mode is unchanged).
      // Running during scrub lets the camera respond immediately when
      // the playhead crosses an hs-* marker.
      this._syncCameraMode();
      // Same reasoning — resolve asset state calls every tick so scrubbing
      // (not just forward playback) lands assets on the right state.
      this._syncAssetStates();

      const objFrames = this._animation.getObjectFrames();
      if (objFrames.length > 0) {
        let dirty = false;
        for (const { id, pos, quat } of objFrames) {
          const slots = this._partIndex[id];
          if (slots && slots.length) {
            let fpos = pos, fquat = quat;
            const other = this._sceneBlend?.otherObjects?.[id];
            if (other) {
              const bf = this._sceneBlend.bf;
              fpos  = lerp3(other.pos, pos, bf);
              fquat = slerpQuat(
                other.quat[0], other.quat[1], other.quat[2], other.quat[3],
                quat[0],       quat[1],       quat[2],       quat[3], bf,
              );
            }
            const m = quatPosToMat4(fpos, fquat);
            this._partLocalPose[id] = m;
            for (const slot of slots) this._partTransFlat.set(m, slot * 16);
            dirty = true;
          }
        }
        if (dirty) { this._renderer.updateTransforms(this._partTransFlat); transformsDirty = true; }
      }

      // ── Asset anchors ────────────────────────────────────────────────────────
      // A pure parent transform for an externally-loaded asset's parts (see
      // Animation#getAnchorFrames) — any loaded part whose id is namespaced
      // under "ctrl.<assetId>" gets the anchor's world transform composed
      // with whatever local pose this tick already wrote for that part
      // (e.g. a state/transition/clip pose, or an anim object frame above —
      // see _partLocalPose) — anchorMat alone for parts with no local pose
      // of their own. Composing rather than overwriting is required for
      // assets like headphones that are both anchored on the main timeline
      // AND have their own state axis (e.g. "fold") animating sub-parts;
      // overwriting wiped out the state's pose every single tick.
      const anchorFrames = this._animation.getAnchorFrames();
      if (anchorFrames.length) {
        let anchored = false;
        for (const { asset, pos, quat } of anchorFrames) {
          const anchorMat = quatPosToMat4(pos, quat);
          const prefix = `ctrl.${asset}`;
          for (const id in this._partIndex) {
            if (!id.startsWith(prefix)) continue;
            const local = this._partLocalPose[id];
            const m = local ? mat4Mul(anchorMat, local) : anchorMat;
            for (const slot of this._partIndex[id]) this._partTransFlat.set(m, slot * 16);
            anchored = true;
          }
        }
        if (anchored) { this._renderer.updateTransforms(this._partTransFlat); transformsDirty = true; }
      }

      volFrames = this._animation.getVolumeFrames();
      // Cross-fade mask-volume matrices the same way object/camera poses are
      // blended across a scene handoff (see player.js's updateSceneBlend) —
      // otherwise a mask snaps instantly to its new-scene state on the tick
      // the handoff fires while everything else eases in, a visible flicker.
      if (this._sceneBlend?.otherVolumes) {
        const { otherVolumes, bf } = this._sceneBlend;
        for (const v of volFrames) {
          const other = otherVolumes[v.name];
          if (other) v.matrix = lerpMat4(other, v.matrix, bf);
        }
      }
      // A mask volume can be animated (e.g. a reveal wipe during an intro)
      // independently of the camera — _computeActiveRanges below only runs
      // when something is flagged dirty, so without this, a part hidden by
      // a moving mask stays stuck hidden/shown (using a stale active-range
      // snapshot) until the next camera-driven render forces a recompute —
      // visible as a part vanishing, then popping back on the next scroll.
      const animFrame = this._animation.frame;
      if (volFrames.length > 0 && animFrame !== this._lastVolAnimFrame) this._sortDirty = true;
      this._lastVolAnimFrame = animFrame;
    } else if (this._autoRotate) {
      this._camera.theta += 0.005;
    }

    // Clip masks (asset color-transition fades — see loadClips()/playClip())
    // are independent of this._animation and apply regardless of whether a
    // main animation is attached at all.
    for (const name in this._clipMaskState) {
      volFrames.push({ name, matrix: this._clipMaskState[name], softEdge: this._clipMaskSoftEdge[name] });
    }
    if (volFrames.length > 0) {
      for (const v of volFrames) {
        const feather = this._maskFeather[v.name];
        if (feather != null) v.softEdge = feather;
      }
      this._renderer.updateMaskVolumes(volFrames);
    }

    if (!this._numSplats) return;

    this._camera.update(w, h);

    const view  = this._camera.viewMatrix;
    const proj  = this._camera.projMatrix;

    // Skip sort+render when nothing has changed — keeps GPU idle on static scenes.
    let viewChanged = !this._lastSortView;
    if (!viewChanged) {
      for (let i = 0; i < 16; i++) {
        if (view[i] !== this._lastSortView[i]) { viewChanged = true; break; }
      }
    }
    if (!viewChanged && !transformsDirty && !this._sortDirty) return;

    // Frame rate cap: limit GPU work to ~60fps on high-refresh displays.
    // The animation tick (above) still runs every RAF for smooth timing.
    if (now - this._lastRenderMs < 15) return;
    this._lastRenderMs = now;

    if (!this._lastSortView) this._lastSortView = new Float32Array(16);
    this._lastSortView.set(view);
    this._sortDirty = false;

    // Notify listeners (e.g. player updating callout positions)
    if (this.onFrame) {
      this.onFrame(view, proj, w, h);
    }

    const focal = this._camera.focalLength(h);

    // Upload uniforms before sorting — the GPU sort's cs_depth_key kernel
    // reads uniforms.view for this frame's depth computation.
    this._renderer.updateUniforms({
      view, proj, width: w, height: h, focal,
      near: this._camera.near, radiusCap: this._effectiveRadiusCap,
    });

    // null = render the full scene (fast path). When mask volumes hide one or
    // more files entirely, or a part's bounds fall entirely outside the view
    // frustum, this is the number of active gaussians and this._activeIdx
    // holds their packed indices — see _computeActiveRanges.
    const activeCount = this._computeActiveRanges(volFrames, view, proj);

    // Sort back-to-front. When some parts are hidden/culled (activeCount !==
    // null), restrict sort/draw to the active subset — sort cost and
    // instance count both scale with N, not _numSplats.
    const N = activeCount === null ? this._numSplats : activeCount;
    this._lastActiveCount = N;

    // Per-splat covariance/eigen/SH preprocess (see shaders.js cs_preprocess) —
    // must run before both the sort (GPU sort's depth_key reads gaussian data
    // independently, but vs_main needs this frame's geometry either way) and
    // draw(). Runs every frame: view/transforms/masks can all change frame to
    // frame.
    this._renderer.preprocess(N);

    if (this._gpuSort && activeCount === null && !this._renderer._gpuSortFailed) {
      // GPU radix sort writes the permutation directly into _orderBuf (idxA) —
      // no CPU readback. Not used with per-part culling yet (v1 keeps GPU
      // sort and _activeIdx compaction mutually exclusive — see plan).
      // Falls back to the CPU path below if _gpuSortFailed gets set (a GPU
      // validation error was reported via uncapturederror — see renderer init).
      this._renderer.runGpuSort(N);
    } else {
      this._computeDepths(view);
      const order = this._sort(this._depths, N, activeCount === null ? null : this._activeIdx);
      this._renderer.updateOrder(order, N);
    }

    this._renderer.draw(N);
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

  /** Build the set of gaussian indices to sort/draw this frame, writing them
   *  into this._activeIdx. A part's range is excluded when it's fully hidden
   *  by mask volumes (_isFileHidden) or its bounds lie entirely outside the
   *  view frustum (_isFileOutsideView). Returns null (fast path — render
   *  everything) for single-part scenes or when nothing is excluded;
   *  otherwise returns the active count. */
  _computeActiveRanges(volFrames, view, proj) {
    // Per-part culling (mask/frustum) forces the renderer onto the slower
    // CPU sort fallback (see _tick — GPU sort only runs when activeCount is
    // null), which has turned out to cause visible flicker/instability and
    // softer image quality versus the GPU radix sort's fast path. Until that
    // CPU-sort-path issue is root-caused, skip per-file culling entirely —
    // costs some performance (everything gets sorted/drawn every frame,
    // masked-out splats included) but the shader's own per-splat maskFade
    // still hides them correctly; this only forgoes the CPU-side shortcut.
    return null;
    const ranges = this._fileRanges;
    if (!this._activeIdx || !ranges || ranges.length < 2) return null;

    const masks = this._fileMasks;
    let anyMasked = false;
    if (masks) {
      for (let i = 0; i < masks.length; i++) {
        if (masks[i]) { anyMasked = true; break; }
      }
    }
    const invMats = anyMasked ? volFrames.map(v => invertMat4(v.matrix)) : null;

    const tanHalfFovX = 1 / proj[0];
    const tanHalfFovY = 1 / proj[5];

    const idx = this._activeIdx;
    let count = 0;
    let anyExcluded = false;
    for (let slot = 0; slot < ranges.length; slot++) {
      const range = ranges[slot];
      if (!range) continue;
      if (anyMasked && this._isFileHidden(slot, volFrames, invMats)) { anyExcluded = true; continue; }
      if (this._isFileOutsideView(slot, view, proj, tanHalfFovX, tanHalfFovY)) { anyExcluded = true; continue; }
      for (let g = range[0]; g < range[1]; g++) idx[count++] = g;
    }
    return anyExcluded ? count : null;
  }

  /** True if `slot`'s world-space bounding box lies entirely outside the
   *  camera's view frustum (behind the near plane, or off to one side), so
   *  its splat range can be skipped from sort/draw. Conservative: a corner
   *  that can't be unambiguously placed on one side prevents the cull. */
  _isFileOutsideView(slot, view, proj, tanHalfFovX, tanHalfFovY) {
    const aabb = this._fileAABB[slot];
    if (!aabb) return false;

    const { min, max } = aabb;
    const m    = this._partTransFlat.subarray(slot * 16, slot * 16 + 16);
    const near = this._camera.near;

    let allBehindNear = true;
    let allLeft = true, allRight = true, allAbove = true, allBelow = true;

    for (let c = 0; c < 8; c++) {
      const lx = (c & 1) ? max[0] : min[0];
      const ly = (c & 2) ? max[1] : min[1];
      const lz = (c & 4) ? max[2] : min[2];

      const wx = m[0]*lx + m[4]*ly + m[8]*lz  + m[12];
      const wy = m[1]*lx + m[5]*ly + m[9]*lz  + m[13];
      const wz = m[2]*lx + m[6]*ly + m[10]*lz + m[14];

      const vx = view[0]*wx + view[4]*wy + view[8]*wz  + view[12];
      const vy = view[1]*wx + view[5]*wy + view[9]*wz  + view[13];
      const vz = view[2]*wx + view[6]*wy + view[10]*wz + view[14];

      if (vz < -near) {
        allBehindNear = false;
        const limX = -vz * tanHalfFovX;
        const limY = -vz * tanHalfFovY;
        if (vx > -limX) allLeft  = false;
        if (vx <  limX) allRight = false;
        if (vy > -limY) allAbove = false;
        if (vy <  limY) allBelow = false;
      } else {
        allLeft = allRight = allAbove = allBelow = false;
      }
    }

    return allBehindNear || allLeft || allRight || allAbove || allBelow;
  }

  /** True if `slot`'s bounding box lies entirely outside any mask volume
   *  that affects it — i.e. that volume's maskFade is 0 across the whole
   *  file, making the file invisible regardless of other volumes (maskFade
   *  is a product across affecting volumes). */
  _isFileHidden(slot, volFrames, invMats) {
    const maskBits = this._fileMasks[slot];
    if (!maskBits) return false;
    const aabb = this._fileAABB[slot];
    if (!aabb) return false;

    const partMat = this._partTransFlat.subarray(slot * 16, slot * 16 + 16);
    for (let vi = 0; vi < volFrames.length; vi++) {
      if (!((maskBits >> vi) & 1)) continue;
      const inv = invMats[vi];
      if (!inv) continue; // degenerate volume — shader treats it as a no-op too

      const m = mat4Mul(inv, partMat);
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let c = 0; c < 8; c++) {
        const x = (c & 1) ? aabb.max[0] : aabb.min[0];
        const y = (c & 2) ? aabb.max[1] : aabb.min[1];
        const z = (c & 4) ? aabb.max[2] : aabb.min[2];
        const px = m[0]*x + m[4]*y + m[8]*z  + m[12];
        const py = m[1]*x + m[5]*y + m[9]*z  + m[13];
        const pz = m[2]*x + m[6]*y + m[10]*z + m[14];
        if (px < minX) minX = px; if (px > maxX) maxX = px;
        if (py < minY) minY = py; if (py > maxY) maxY = py;
        if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;
      }

      const lim = 0.5 + (volFrames[vi].softEdge ?? 0.05);
      if (maxX < -lim || minX > lim || maxY < -lim || minY > lim || maxZ < -lim || minZ > lim) {
        return true;
      }
    }
    return false;
  }

  // ── Resize handling ────────────────────────────────────────────────────────

  _observeResize() {
    if (typeof ResizeObserver === 'undefined') return;
    this._resizeObs = new ResizeObserver(() => {
      // Debounce so mobile browser bar show/hide (which fires a continuous stream
      // of resize events as the bar animates) only triggers one canvas resize
      // after the animation settles, not one per pixel of bar movement.
      clearTimeout(this._resizeTimer);
      this._resizeTimer = setTimeout(() => this._updateSize(), 150);
    });
    this._resizeObs.observe(this._canvas);
  }

  _updateSize() {
    const dpr = Math.min(window.devicePixelRatio || 1, this._effectivePixelRatio);
    const w   = Math.round(this._canvas.clientWidth  * dpr);
    const h   = Math.round(this._canvas.clientHeight * dpr);
    if (w && h && (this._canvas.width !== w || this._canvas.height !== h)) {
      this._canvas.width  = w;
      this._canvas.height = h;
    }
  }

  // ── Adaptive quality ───────────────────────────────────────────────────────

  /**
   * Tracks a smoothed frame time and, once per second, nudges
   * _effectivePixelRatio down when frames are running slow (<27fps) or back
   * up toward _maxPixelRatio when comfortably fast (>50fps), resizing the
   * canvas to match. No-op if `dt` is 0 (first frame) or adaptiveQuality is off.
   *
   * @param {boolean} warmingUp  while true (see the _sceneReady rising-edge
   *   check in _tick), the EMA still gets tracked but the scale-down branch
   *   is skipped — startup churn raises frame times without representing
   *   steady-state device performance, and scale-down recovers far slower
   *   than it triggers (0.85x/s down vs 1.05x/s up).
   */
  _updateAdaptiveQuality(dt, warmingUp) {
    if (!this._adaptiveQuality || !dt) return;

    const alpha = 1 - Math.exp(-dt / 0.5); // ~0.5s smoothing window
    this._frameTimeEMA = this._frameTimeEMA == null
      ? dt
      : this._frameTimeEMA + (dt - this._frameTimeEMA) * alpha;

    const now = performance.now();
    if (now - this._lastQualityCheck < 1000) return;
    this._lastQualityCheck = now;

    const SLOW = 1 / 27; // scale down below ~27fps
    const FAST = 1 / 50; // scale back up above ~50fps

    if (!warmingUp && this._frameTimeEMA > SLOW && this._effectivePixelRatio > this._minPixelRatio) {
      this._effectivePixelRatio = Math.max(this._minPixelRatio, this._effectivePixelRatio * 0.85);
      this._updateSize();
    } else if (this._frameTimeEMA < FAST && this._effectivePixelRatio < this._maxPixelRatio) {
      this._effectivePixelRatio = Math.min(this._maxPixelRatio, this._effectivePixelRatio * 1.05);
      this._updateSize();
    }

    if (!warmingUp && this._frameTimeEMA > SLOW && this._effectiveRadiusCap > this._minRadiusCap) {
      this._effectiveRadiusCap = Math.max(this._minRadiusCap, this._effectiveRadiusCap * 0.85);
    } else if (this._frameTimeEMA < FAST && this._effectiveRadiusCap < this._maxRadiusCap) {
      this._effectiveRadiusCap = Math.min(this._maxRadiusCap, this._effectiveRadiusCap * 1.05);
    }
  }
}

// ── Constants ─────────────────────────────────────────────────────────────

const IDENTITY_MAT4 = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);

// How long after _sceneReady flips true to hold adaptive quality at max
// before letting it scale down — see the _sceneReady rising-edge check in
// _tick. Covers startup churn (clip/variant decode, first full-scene
// draw+sort) that isn't representative of steady-state frame time.
const QUALITY_WARMUP_MS = 8000;

// How long loadClips() waits after the default color variant is showing
// before fetching the other color variants in the background — see the
// Phase 2 comment in loadClips(). Long enough for the page to settle past
// its own startup churn before adding more network/decompression load.
const CLIP_VARIANT_PREFETCH_DELAY_MS = 4000;

// LOD tier forced on a clip part's non-active color variants (see
// loadClips() Phase 2) regardless of device tier — per-file mask culling is
// disabled (_computeActiveRanges always returns null), so every variant a
// part loads costs sort/draw time every frame forever, not just while
// visible. 2 is the most aggressive tier prune.html generates.
const BACKGROUND_VARIANT_LOD = 2;

// ── Helpers ────────────────────────────────────────────────────────────────

/** Strip query string, directory, known splat extension, and a trailing
 *  `.lod{N}` LOD-tier suffix (see device-tier.js's resolveLodUrl — mask
 *  matching below must key off the original splat name regardless of which
 *  LOD tier actually got fetched, or every "<prefix>..<suffix>.mask" lookup
 *  silently matches 0 files once LOD substitution is in play) from a URL,
 *  leaving the bare splat name (e.g. "headphones.headband.yellow"). */
function basenameNoExt(url) {
  const base = url.split('?')[0].split('/').pop();
  return base.replace(/\.(spz|ply|splat)$/i, '').replace(/\.lod\d+$/i, '');
}

/** Picks which of a part's "<axis>=<value>" variant suffixes (see
 *  export_holosplat_asset.py) matches `defaults`, returning it as a leading
 *  "." segment ready to append to the part's base splat name — or '' if the
 *  part has no variants. Falls back to the first variant if none of them
 *  match (e.g. defaults omits that part's axis). */
function defaultVariantSuffix(part, defaults) {
  if (!part.variants.length) return '';
  const match = part.variants.find(v => {
    const eq = v.indexOf('=');
    if (eq < 0) return false;
    return defaults?.[v.slice(0, eq)] === v.slice(eq + 1);
  });
  return '.' + (match ?? part.variants[0]);
}

/** Multiplies two column-major mat4s: returns a * b. */
function mat4Mul(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[row]      * b[col * 4]     +
        a[row + 4]  * b[col * 4 + 1] +
        a[row + 8]  * b[col * 4 + 2] +
        a[row + 12] * b[col * 4 + 3];
    }
  }
  return out;
}

// Half-extent multiplier applied to each axis's largest splat scale when
// computing a file's local-space AABB — a coarse but cheap bound on how far
// a Gaussian's visible footprint extends past its center.
const MASK_AABB_MARGIN = 3;

/** Local-space AABB (position ± scale margin) over `count` Gaussians,
 *  or null if count is 0. */
function computeAABB(data, count) {
  if (count === 0) return null;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    const j = i * 16;
    const x = data[j], y = data[j + 1], z = data[j + 2];
    const m = MASK_AABB_MARGIN * Math.max(data[j + 8], data[j + 9], data[j + 10]);
    if (x - m < minX) minX = x - m; if (x + m > maxX) maxX = x + m;
    if (y - m < minY) minY = y - m; if (y + m > maxY) maxY = y + m;
    if (z - m < minZ) minZ = z - m; if (z + m > maxZ) maxZ = z + m;
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

/** Merges `aabb` (or null) with the AABB of another chunk of Gaussians. */
function extendAABB(aabb, data, count) {
  const next = computeAABB(data, count);
  if (!next) return aabb;
  if (!aabb) return next;
  return {
    min: [Math.min(aabb.min[0], next.min[0]), Math.min(aabb.min[1], next.min[1]), Math.min(aabb.min[2], next.min[2])],
    max: [Math.max(aabb.max[0], next.max[0]), Math.max(aabb.max[1], next.max[1]), Math.max(aabb.max[2], next.max[2])],
  };
}

/**
 * Tests whether a loaded splat file's name matches a mask-volume prefix.
 *
 * "<a>"     → fileName === a || fileName.startsWith(a + '.')
 * "<a>..<b>" → both of the above for `a`, AND
 *              fileName === b || fileName.endsWith('.' + b)
 *              (an empty `a` or `b` skips that half of the check, so
 *              "..yellow" matches any file ending in ".yellow")
 */
function matchesMaskPrefix(fileName, prefix) {
  const sep   = prefix.indexOf('..');
  const name1 = sep === -1 ? prefix : prefix.slice(0, sep);
  const name2 = sep === -1 ? ''     : prefix.slice(sep + 2);
  const pre = name1 === '' || fileName === name1 || fileName.startsWith(name1 + '.');
  const suf = name2 === '' || fileName === name2 || fileName.endsWith('.' + name2);
  return pre && suf;
}

/**
 * Extract the camera-mode token from a marker name.
 * Handles both pure hs-* markers ("hs-h30") and scene markers with embedded
 * hs-* tokens ("feature-01 hs-h30", "intro hs-h30 hs-v20").
 * Returns the hs-* portion as a single string, or null for non-camera markers.
 *
 *   "hs-h30"              → "hs-h30"
 *   "feature-01 hs-h30"   → "hs-h30"
 *   "intro hs-h30 hs-v20" → "hs-h30-v20"   (merged for regex matching)
 *   "hs-locked"           → "hs-locked"
 *   "feature-01 hs-locked"→ "hs-locked"
 *   "intro"               → null
 */
function extractHsMode(name) {
  if (name.startsWith('hs-')) return name;
  const hs = name.trim().split(/\s+/).filter(t => t.startsWith('hs-'));
  if (!hs.length) return null;
  return hs.length === 1 ? hs[0] : 'hs-' + hs.map(t => t.slice(3)).join('-');
}

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

// Sign flip per SH-rest basis (indices 0-14) under the same 180°-about-X
// rotation flipYInPlace applies to position/quaternion: y→-y, z→-z, x
// unchanged. Each real-SH basis function is a polynomial in x,y,z, so its
// sign under this transform is just its parity in y and z combined — e.g.
// basis 0 ∝ y flips, basis 2 ∝ x doesn't, basis 4 ∝ yz doesn't (both factors
// flip, signs cancel). Derived directly from the basis list in shaders.js.
// SH-rest coefficients are NOT rotation-invariant — flipYInPlace rotates
// every splat's position/orientation but never touched this data, so SH
// evaluation was silently using coefficients meant for the original,
// unrotated frame against directions computed in the flipped frame.
const SH_FLIP_Y_SIGN = [
  -1, -1, 1,           // degree 1: y, z, x
  -1, 1, 1, -1, 1,     // degree 2: xy, yz, (2zz-xx-yy), xz, (xx-yy)
  -1, 1, -1, -1, 1, -1, 1, // degree 3
];

function flipYInPlaceSH(shData, numBases, count) {
  if (!shData || !numBases) return;
  const n = Math.min(numBases, SH_FLIP_Y_SIGN.length);
  for (let i = 0; i < count; i++) {
    const base = i * numBases * 3;
    for (let b = 0; b < n; b++) {
      if (SH_FLIP_Y_SIGN[b] !== -1) continue;
      const o = base + b * 3;
      shData[o]     = -shData[o];
      shData[o + 1] = -shData[o + 1];
      shData[o + 2] = -shData[o + 2];
    }
  }
}

const SPLAT_EXTS    = ['spz', 'ply', 'splat'];
// .ply and .spz are handled explicitly below (both need shDegree); this map
// only covers the remaining formats (currently just .splat, via loadSplat).
const SPLAT_LOADERS = {};

/**
 * Load a splat file, auto-detecting format from the extension.
 * If the file returns HTTP 4xx, tries the other known extensions in order
 * (.spz → .ply → .splat) so renaming or re-encoding a file doesn't break callers.
 * Non-4xx errors (network failures, 5xx) are rethrown immediately.
 */
async function loadUrl(url, onProgress, shDegree = 0) {
  const clean   = url.split('?')[0];
  const lastDot = clean.lastIndexOf('.');
  const rawExt  = lastDot >= 0 ? clean.slice(lastDot + 1).toLowerCase() : '';

  // .spzv (packed variants) is a deliberate choice, not a fallback target —
  // load it directly with no .spz/.ply/.splat fallback chain.
  if (rawExt === 'spzv') return loadSpzv(url, onProgress);

  const hasExt  = SPLAT_EXTS.includes(rawExt);

  // Try the given extension first, then the remaining formats as fallbacks.
  const exts = hasExt ? [rawExt, ...SPLAT_EXTS.filter(e => e !== rawExt)] : SPLAT_EXTS;
  const base  = hasExt ? url.slice(0, url.lastIndexOf('.')) : url;

  let lastErr;
  for (const ext of exts) {
    const candidate = `${base}.${ext}`;
    const loader    = ext === 'ply'  ? (u, p) => loadPly(u, p, shDegree)
                     : ext === 'spz' ? (u, p) => loadSpz(u, p, shDegree)
                     : (SPLAT_LOADERS[ext] ?? loadSplat);
    try {
      return await loader(candidate, onProgress);
    } catch (err) {
      if (!/^HTTP 4/.test(err.message)) throw err; // non-4xx → give up immediately
      lastErr = err;
    }
  }
  throw new Error(`HoloSplat: splat file not found as .spz / .ply / .splat — "${base}"`);
}

function smoothstep(t) { return t * t * (3 - 2 * t); }
function lerp3(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}
function lerpMat4(a, b, t) {
  const out = new Float32Array(16);
  for (let i = 0; i < 16; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  return out;
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
