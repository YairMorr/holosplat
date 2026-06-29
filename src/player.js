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
import { splatNameFromId } from './animation.js';
import { detectDeviceTier, qualityForTier, resolveLodUrl, resolvePartsLod } from './device-tier.js';

// ── CSS (injected once per page) ──────────────────────────────────────────────

const PLAYER_CSS = `
.hs-player{position:relative;overflow:hidden;}
.hs-player canvas{position:absolute;inset:0;width:100%;height:100%;display:block;}
.hs-player .hs-overlay{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:10px;
  pointer-events:none;font-family:system-ui,sans-serif;
}
.hs-player .hs-pct{font-size:1.1rem;font-weight:600;color:rgba(255,255,255,.85);letter-spacing:.02em;}
.hs-player .hs-bar-wrap{width:140px;height:3px;background:rgba(255,255,255,.1);border-radius:2px;overflow:hidden;}
.hs-player .hs-bar{height:100%;background:#3a7aff;width:0%;transition:width .1s;}
.hs-player .hs-msg{font-size:.78rem;color:rgba(255,255,255,.4);text-align:center;max-width:260px;line-height:1.5;padding:0 16px;}
.hs-player .hs-msg.hs-err{color:#f87;}
.hs-player .hs-callouts{position:absolute;inset:0;pointer-events:none;}
.hs-lines{position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none;}
.hs-dot{fill:#3a7aff;stroke:#fff;stroke-width:2;}
.hs-line{stroke:rgba(255,255,255,.55);stroke-width:1.5;}
.hs-callout{position:absolute;pointer-events:auto;}
.hs-callout--hidden{display:none;}
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
 * @param {number}  [opts.near=0.01]
 * @param {number}  [opts.far=2000]
 * @param {number}  [opts.splatScale=1.08]
 * @param {boolean} [opts.autoRotate=false]
 * @param {'auto'|'low'|'medium'|'high'} [opts.quality='auto']  device-tier presets
 *        for maxPixelRatio/shDegree caps and LOD file selection — see device-tier.js
 * @param {boolean} [opts.adaptiveQuality=true]  scale render resolution down when
 *        frame time exceeds ~37ms (27fps) and back up toward maxPixelRatio when
 *        comfortably under ~20ms (50fps)
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

  const dataSrc  = root.getAttribute('data-holosplat')      || undefined;
  const dataAnim = root.getAttribute('data-holosplat-anim') || undefined;

  const {
    scene,               // preferred name
    src:      srcAlias,  // kept for backward compat
    parts,
    animation:  animSrc = dataAnim,
    clips,               // string URL, or array of URLs — asset clip file(s); see export_holosplat_asset.py
    partsDir,
    partsExt    = '',
    scenes,              // markerName → scene config object
    masks,               // mask volume name → { feather } config object
    sh,                  // global SH degree (0–3); overrides per-scene sh if set
    background  = 'transparent',
    fov         = 60,
    near        = 0.01,
    far         = 2000,
    splatScale  = 1.08,
    autoRotate  = false,
    flipY       = false,
    maxPixelRatio,
    quality     = 'auto', // 'auto'|'low'|'medium'|'high' — see device-tier.js
    adaptiveQuality = true, // dynamically scale render resolution based on frame time
    prefetchVariants, // background-fetch inactive color variants; default depends on device tier
    aaDilation  = 0.3,  // anti-aliasing covariance dilation (default 0.3, matches PlayCanvas/SuperSplat)
    gpuSort     = false, // opt-in GPU compute-shader radix sort — see src/sort-shaders.js
    zIndex      = 5,
    onLoad, onProgress, onError,
  } = opts;

  const src = scene ?? srcAlias ?? dataSrc;

  const tier = quality === 'auto' ? detectDeviceTier() : quality;
  const caps = qualityForTier(tier);
  const effectiveMaxPixelRatio = maxPixelRatio ?? caps.maxPixelRatio;

  root.style.zIndex = String(zIndex);

  // ── Build DOM ───────────────────────────────────────────────────────────────
  root.classList.add('hs-player');

  const canvas    = document.createElement('canvas');
  const calloutEl = document.createElement('div');
  calloutEl.className = 'hs-callouts';
  const overlay = document.createElement('div');
  overlay.className = 'hs-overlay';
  overlay.innerHTML =
    '<div class="hs-pct">0%</div>' +
    '<div class="hs-bar-wrap"><div class="hs-bar"></div></div>' +
    '<div class="hs-msg"></div>';
  root.appendChild(canvas);
  root.appendChild(calloutEl);
  root.appendChild(overlay);

  const pctEl   = overlay.querySelector('.hs-pct');
  const barWrap = overlay.querySelector('.hs-bar-wrap');
  const bar     = overlay.querySelector('.hs-bar');
  const msgEl   = overlay.querySelector('.hs-msg');
  let _loadStart = null;
  // Ref-counted: the boot sequence kicks off several loads concurrently
  // (main scene/parts, animation, clip assets), each wrapped in its own
  // showLoading()/showReady() pair. Without counting, whichever one finishes
  // first hides the overlay while the others are still fetching — exactly
  // what made the loading screen vanish early while big asset variants kept
  // downloading in the background.
  let _pendingLoads = 0;

  function showLoading() {
    _pendingLoads++;
    pctEl.textContent = '0%';
    pctEl.style.display = '';
    barWrap.style.display = '';
    msgEl.textContent = '';
    msgEl.className = 'hs-msg';
    overlay.style.display = 'flex';
    _loadStart = performance.now();
  }
  function showReady() {
    _pendingLoads = Math.max(0, _pendingLoads - 1);
    if (_pendingLoads === 0) overlay.style.display = 'none';
  }
  function showError(text) {
    pctEl.style.display = 'none';
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
    flipY,
    aaDilation,
    maxPixelRatio: effectiveMaxPixelRatio,
    adaptiveQuality,
    prefetchVariants: prefetchVariants ?? caps.prefetchVariants,
    gpuSort,
    tier,
    onProgress: p => {
      const pct = Math.round(p * 100);
      pctEl.textContent = pct + '%';
      bar.style.width = pct + '%';
      if (_loadStart && p > 0.05) {
        const elapsed = (performance.now() - _loadStart) / 1000;
        const eta = Math.round(elapsed * (1 - p) / p);
        msgEl.textContent = eta > 0 ? '~' + eta + 's' : '';
      }
      if (onProgress) onProgress(p);
    },
  });

  // ── Callout DOM ──────────────────────────────────────────────────────────────
  // Map of id → { card, dot, line }
  const calloutDivs = {};
  const NS = 'http://www.w3.org/2000/svg';

  function buildCallouts(callouts) {
    calloutEl.innerHTML = '';
    for (const key of Object.keys(calloutDivs)) delete calloutDivs[key];
    if (!callouts.length) return;

    // SVG overlay for dots and connector lines
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('class', 'hs-lines');
    calloutEl.appendChild(svg);

    for (const c of callouts) {
      // Line drawn first (behind dot)
      const line = document.createElementNS(NS, 'line');
      line.setAttribute('class', 'hs-line');
      svg.appendChild(line);

      const dot = document.createElementNS(NS, 'circle');
      dot.setAttribute('class', 'hs-dot');
      dot.setAttribute('r', '5');
      svg.appendChild(dot);

      // Find user-authored card: look in root container first, then document-wide
      let card = root.querySelector(`.hs-callout[data-id="${c.id}"]`)
               ?? document.querySelector(`.hs-callout[data-id="${c.id}"]`);
      if (card) {
        calloutEl.appendChild(card); // move into overlay for absolute positioning
      } else {
        card = document.createElement('div');
        card.className = 'hs-callout';
        card.dataset.id = c.id;
        calloutEl.appendChild(card);
      }

      calloutDivs[c.id] = { card, dot, line };
    }
  }

  // Update positions each frame
  viewer.onFrame = () => {
    if (!viewer._animation?.callouts.length) return;
    const projected = viewer.projectCallouts(viewer._animation.callouts);
    for (const { id, visible, x, y } of projected) {
      const entry = calloutDivs[id];
      if (!entry) continue;
      const { card, dot, line } = entry;
      if (visible) {
        const ox = parseFloat(card.dataset.offsetX ?? card.dataset.ox ?? 80);
        const oy = parseFloat(card.dataset.offsetY ?? card.dataset.oy ?? -40);
        const cx = x + ox, cy = y + oy;

        dot.setAttribute('cx', x);   dot.setAttribute('cy', y);
        line.setAttribute('x1', x);  line.setAttribute('y1', y);
        line.setAttribute('x2', cx); line.setAttribute('y2', cy);
        dot.style.display  = '';
        line.style.display = '';

        card.style.left = cx + 'px';
        card.style.top  = cy + 'px';
        card.classList.remove('hs-callout--hidden');
      } else {
        dot.style.display  = 'none';
        line.style.display = 'none';
        card.classList.add('hs-callout--hidden');
      }
    }
  };

  // ── Load scene ───────────────────────────────────────────────────────────────
  // These throw on failure so the boot sequence can catch and show the error.
  // showReady / onLoad are called by boot after ALL parallel loads finish.
  async function load(url) {
    showLoading();
    bar.style.width = '0%';
    try {
      await viewer.load(url);
      showReady();
    } catch (err) {
      const msg = navigator.gpu
        ? err.message
        : 'WebGPU not supported. Use Chrome 113+ or Edge 113+.';
      showError(msg);
      if (onError) onError(err);
      throw err;
    }
  }

  async function loadPartsMap(partsMap) {
    showLoading();
    bar.style.width = '0%';
    try {
      await viewer.loadParts(partsMap);
      showReady();
    } catch (err) {
      const msg = navigator.gpu
        ? err.message
        : 'WebGPU not supported. Use Chrome 113+ or Edge 113+.';
      showError(msg);
      if (onError) onError(err);
      throw err;
    }
  }

  // ── Clips (product customization) ───────────────────────────────────────────
  // Wires a 'click' listener to any page element whose id matches an exported
  // clip's id ("[productName]-[variant]" — see export_holosplat_asset.py /
  // Viewer#playClip). Buttons live in the surrounding page, not inside the
  // player root, so this searches the whole document (same pattern
  // buildCallouts uses below). Clips are fully independent of the main
  // animation, so this just binds whatever's currently in viewer._clips —
  // call again after loadClips() if buttons exist before that resolves.
  function bindClipButtons() {
    const ids = Object.keys(viewer._clips);
    let bound = 0;
    for (const clipId of ids) {
      const btn = document.getElementById(clipId);
      if (!btn || btn.dataset.hsClipBound) continue;
      btn.dataset.hsClipBound = '1';
      btn.addEventListener('click', () => viewer.playClip(clipId));
      bound++;
    }
    if (bound) console.log(`[HoloSplat] ${bound} clip button(s) bound:`, ids);
  }

  // Internal: used by the boot sequence's `clips` option — swallows errors
  // so one bad URL doesn't break Promise.all for the others (same pattern
  // as loadAnim below). Each entry is either a bare url string, or
  // {url, splatsDir, defaults} once a splats path/default variant has been
  // set in the editor — see holosplat/editor.js's saveAssetsAttr.
  async function loadClips(entry) {
    const { url, splatsDir, defaults } = typeof entry === 'string' ? { url: entry } : entry;
    showLoading();
    try {
      // Resolves once each part's default variant is loaded — non-default
      // color variants keep fetching in the background after this returns
      // (see Viewer#loadClips), so they don't hold up the boot sequence.
      await viewer.loadClips(url, { splatsDir, defaults, lod: caps.lod });
      bindClipButtons();
    } catch (err) {
      console.error('[HoloSplat] clips failed to load:', err);
      if (onError) onError(err);
    } finally {
      showReady();
    }
  }

  // ── Load animation ───────────────────────────────────────────────────────────
  async function loadAnim(url) {
    try {
      const anim = await viewer.loadAnimationUrl(url);
      buildCallouts(anim.callouts);
      console.log(
        `[HoloSplat] animation loaded: ${anim.frameCount} frames @ ${anim.fps}fps, ` +
        `${anim.callouts.length} callout(s):`, anim.callouts.map(c => c.id),
        '| markers:', anim.markers
      );
      return anim;
    } catch (err) {
      console.error('[HoloSplat] animation failed to load:', err);
      if (onError) onError(err);
      // Don't rethrow — leave the scene visible, just no animation
    }
  }

  // ── Public API ───────────────────────────────────────────────────────────────
  const api = {
    load,
    loadParts: loadPartsMap,
    loadAnim,
    destroy() {
      viewer.destroy();
      root.innerHTML = '';
      root.classList.remove('hs-player');
    },
    setBackground(bg)        { viewer.setBackground(bg); },
    setSplatScale(s)         { viewer.setSplatScale(s); },
    setAutoRotate(v)         { viewer.setAutoRotate(v); },
    setFlipY(v)              { viewer.setFlipY(v); },
    setShDegree(n)           { return viewer.setShDegree(n); },
    setAaDilation(v)         { viewer.setAaDilation(v); },
    setAnimationPaused(v)    { viewer.setAnimationPaused(v); },
    setCameraFree(v)         { viewer.setCameraFree(v); },
    setMaskFeather(name, v)  { viewer.setMaskFeather(name, v); },
    playClip(clipId)         { viewer.playClip(clipId); },
    playVariant(axis, value) { viewer.playVariant(axis, value); },
    playState(axis, value)   { viewer.playState(axis, value); },
    // Unlike the internal boot-time loader above, this propagates errors
    // (rejects) so a caller like the editor's asset list can show a
    // per-row error state instead of having failures silently swallowed.
    async loadClips(url, opts) { const ids = await viewer.loadClips(url, opts); bindClipButtons(); return ids; },
    unloadClips(ids)         { viewer.unloadClips(ids); },
    getVariants(id)          { return viewer.getVariants(id); },
    setVariant(id, name)     { return viewer.setVariant(id, name); },
    resetCamera()            { viewer.resetCamera(); },
    callout(id)              { return calloutDivs[id]?.card ?? null; },
    get camera()             { return viewer.camera; },
    get animation()          { return viewer._animation; },
    get animationPaused()    { return viewer._animPaused; },
    get clips()              { return viewer._clips; },
  };

  // Register early so the editor script can find the entry even before boot.
  if (!window.__hsPlayers) window.__hsPlayers = [];
  window.__hsPlayers.push({ root, api, viewer });

  // ── Scroll-driven / per-scene playback ──────────────────────────────────────
  // Called after animation loads. Installs animTickOverride + scroll listener.
  function setupScrollPlayback(anim) {
    if (!scenes || !anim) return null;

    // Prevent browser from restoring scroll position on refresh; always start at top.
    if (history.scrollRestoration) history.scrollRestoration = 'manual';
    window.scrollTo(0, 0);

    // Disable scroll anchoring: when content heights use dynamic viewport units
    // (dvh), the browser adjusts scrollY as the mobile address bar animates in/out.
    // That fires spurious scroll events which jitter the animation.
    document.documentElement.style.overflowAnchor = 'none';

    // See the pingpong turnaround easing below (animTickOverride) for why
    // these exist — softens the velocity discontinuity at a pingpong bounce.
    const PINGPONG_EASE_FRAMES = 15;   // ease window, in frames, on each side of the bounce
    const PINGPONG_MIN_SPEED   = 0.12; // speed floor as a fraction of full speed (0..1) — keeps the bounce from fully stalling

    // Sort markers by frame to derive per-scene frame ranges.
    const markerEntries = Object.entries(anim.markers).sort((a, b) => a[1] - b[1]);
    const sceneList = markerEntries.map(([name, fromFrame], i) => {
      const cfg = scenes[name] ?? {};
      return {
        name,
        fromFrame,
        toFrame:    (markerEntries[i + 1]?.[1] ?? anim.frameCount) - 1,
        playback:   cfg.playback   ?? 'scroll',
        pingpong:   cfg.pingpong   ?? false,
        playOnce:   cfg.playOnce   ?? false,
        blendIn:    (cfg.blendIn  ?? 0) / 100,  // fraction of this scene's own container height
        blendOut:   (cfg.blendOut ?? 0) / 100,
        done:       false,         // play-once scenes: true once their first pass completes
        el: cfg.linkedId ? document.getElementById(cfg.linkedId) : null,
      };
    });
    // next: the scene blendOut eases toward — set after the array exists so
    // each entry can reference its successor.
    sceneList.forEach((s, i) => { s.next = sceneList[i + 1] ?? null; });

    // Only install override if there are any non-auto scenes (would change default behaviour).
    const needsOverride = sceneList.some(s => s.playback !== 'auto');
    if (!needsOverride) return null;

    // Swipe-to-scroll is the primary mobile interaction in scroll-driven scenes —
    // let single-finger touch pass through to the page instead of orbiting/panning
    // the camera. Two-finger pinch/pan still works. Cleared during freecamera
    // "explore" acts (see Viewer#setCameraFree) so touch-orbit works there.
    viewer.camera.allowTouchScroll = true;

    // ── Scroll state ────────────────────────────────────────────────────────
    // A scene's zone "drives" its frame range over exactly one zone-height of
    // scroll: progress 0 when the zone's top edge reaches the viewport top,
    // progress 1 once it has scrolled fully past (top edge has moved up by
    // its own height). Returns null if the zone hasn't been reached yet.
    //
    // For the last zone on the page (nothing below it to keep scrolling
    // into), "scrolled fully past" would require scrollY to exceed the
    // document's max scroll by a full viewport height — physically
    // unreachable, so progress would cap below 1 and the animation could
    // never finish by scroll alone. Clamp the zone's effective bottom to
    // the max reachable scrollY so progress can still reach 1 there.
    function scrollFrameFor(s) {
      if (!s.el) return null;
      const rect = s.el.getBoundingClientRect();
      if (rect.top > 0) return null;
      const scrollY     = window.scrollY;
      const zoneTop     = scrollY + rect.top;
      const zoneBottom  = zoneTop + Math.max(s.el.offsetHeight, 1);
      const maxScrollY  = document.documentElement.scrollHeight - window.innerHeight;
      const effBottom   = Math.min(zoneBottom, Math.max(maxScrollY, zoneTop + 1));
      const t = Math.max(0, Math.min(1, (scrollY - zoneTop) / (effBottom - zoneTop)));
      return { t, frame: s.fromFrame + t * (s.toFrame - s.fromFrame) };
    }

    // ── Blend in/out ──────────────────────────────────────────────────────────
    // Eases the seam where one scene hands off to the next by crossfading
    // actual camera/object POSES, not frame numbers — interpolating frame
    // indices would scrub through whatever animation lies between the two
    // frames, which isn't a blend, it's a fast preview of the timeline.
    // blendIn/blendOut are 0 by default (no blend, exact old behaviour).
    //
    // blendIn eases away from a pose captured once at scene-entry, since the
    // scene being left may not be sitting at a fixed, re-queryable frame (an
    // auto/pingpong scene's position depends on how far it got before
    // handoff) — so it must be snapshotted live, in the moment.
    // blendOut, by contrast, blends toward the next scene's start frame,
    // which IS fixed/known in advance, so it's just sampled directly via
    // Animation#getCameraFrame/getObjectFrames' frameOverride — no capture.
    //
    // Both write into viewer._sceneBlend = { otherEye, otherTarget,
    // otherObjects, bf }, which Viewer#_tick crossfades against its own
    // live-computed pose at weight bf (bf=1 → fully live, bf=0 → fully
    // "other"). scrollTarget.frame itself stays a plain, unblended frame
    // number throughout — only the rendered pose is ever blended.
    let blendSceneName   = null;
    let blendFromEye     = null;
    let blendFromTarget  = null;
    let blendFromObjects = null; // { id: { pos, quat } }
    let blendFromVolumes = null; // { name: Float32Array(16) }

    function smoothstep01(t) {
      const x = Math.max(0, Math.min(1, t));
      return x * x * (3 - 2 * x);
    }
    function captureCurrentPose() {
      const cam = anim.getCameraFrame();
      blendFromEye    = cam.eye;
      blendFromTarget = cam.target;
      blendFromObjects = {};
      for (const o of anim.getObjectFrames()) blendFromObjects[o.id] = { pos: o.pos, quat: o.quat };
      blendFromVolumes = {};
      for (const v of anim.getVolumeFrames()) blendFromVolumes[v.name] = v.matrix;
    }

    function updateSceneBlend(s, t) {
      if (blendSceneName !== s.name) {
        blendSceneName = s.name;
        captureCurrentPose();
      }
      if (s.blendIn > 0 && t < s.blendIn) {
        const bf = smoothstep01(t / s.blendIn);
        viewer._sceneBlend = {
          otherEye: blendFromEye, otherTarget: blendFromTarget,
          otherObjects: blendFromObjects, otherVolumes: blendFromVolumes, bf,
        };
        return;
      }
      if (s.blendOut > 0 && s.next && t > 1 - s.blendOut) {
        const rawBf    = smoothstep01((t - (1 - s.blendOut)) / s.blendOut);
        const nextCam  = anim.getCameraFrame(s.next.fromFrame);
        const nextObjs = {};
        for (const o of anim.getObjectFrames(s.next.fromFrame)) nextObjs[o.id] = { pos: o.pos, quat: o.quat };
        const nextVols = {};
        for (const v of anim.getVolumeFrames(s.next.fromFrame)) nextVols[v.name] = v.matrix;
        viewer._sceneBlend = {
          otherEye: nextCam.eye, otherTarget: nextCam.target,
          otherObjects: nextObjs, otherVolumes: nextVols, bf: 1 - rawBf,
        };
        return;
      }
      viewer._sceneBlend = null;
    }

    // scrollTarget holds the scroll scene currently in view and its target frame.
    // Updated by scroll events; kept at last known value when nothing is in view
    // so the frame doesn't snap back on minor oscillations.
    let scrollTarget = null;   // { scene, frame } | null
    // Prevent auto→scroll handoff from triggering at page load before any user scroll.
    let hasScrolled  = false;

    // Frame the auto/pingpong scene was paused at when a scroll scene took over —
    // restored when scrolling back up so it resumes from there.
    let pausedAutoFrame = null;

    // Fallback hand-off frame if scroll is reversed before any auto→scroll
    // handoff has occurred (e.g. page loaded already mid-scroll).
    const firstScrollIdx = sceneList.findIndex(s => s.playback === 'scroll');
    const entryScene     = firstScrollIdx > 0 ? sceneList[firstScrollIdx - 1] : null;
    const entryFrame     = entryScene ? entryScene.toFrame : (sceneList[0]?.fromFrame ?? 0);

    function updateScrollTarget() {
      // Zones are reached in document order as the user scrolls down, so the
      // LAST zone that has been reached is the current one — earlier zones
      // remain non-null (clamped to their toFrame) once scrolled past.
      let found = null;
      for (const s of sceneList) {
        if (s.playback !== 'scroll' || !s.el) continue;
        const r = scrollFrameFor(s);
        if (r !== null) found = { scene: s, t: r.t, frame: r.frame };
      }
      if (found) {
        updateSceneBlend(found.scene, found.t);
        scrollTarget = { scene: found.scene, frame: found.frame };
        return;
      }
      // No scroll zone reached yet. If we were previously tracking one, the user
      // has scrolled back above all of them — hand control straight back to
      // wherever the auto/pingpong scene was paused (or the entry frame if it
      // never ran) so it can resume from there.
      if (scrollTarget) {
        scrollTarget = null;
        blendSceneName = null;
        viewer._sceneBlend = null;
        anim.seekFrame(pausedAutoFrame ?? entryFrame);
      }
    }

    function onScroll() { hasScrolled = true; updateScrollTarget(); }
    window.addEventListener('scroll', onScroll, { passive: true });
    updateScrollTarget();

    // ── Per-frame tick override ─────────────────────────────────────────────
    viewer.animTickOverride = (dt) => {
      const frame = anim.frame;
      let s = null;
      for (const sc of sceneList) {
        // A finished play-once scene sitting at its end frame hands off to the
        // next scene instead of freezing playback forever at the boundary.
        if (sc.playOnce && sc.done && frame >= sc.toFrame - 0.5) continue;
        if (frame <= sc.toFrame + 0.5) { s = sc; break; }
      }
      if (!s) s = sceneList[sceneList.length - 1];
      if (!s) { anim.tick(dt); return; }
      viewer._currentSceneName = s.name;

      if (s.playback === 'auto') {
        // A scroll scene is demanding control past this auto scene — jump
        // anim straight to scrollTarget.frame; viewer._sceneBlend (set by
        // updateSceneBlend above) crossfades the rendered pose from this
        // scene's last frame if the incoming scroll scene has blendIn set —
        // with blendIn at 0 (default) there's no crossfade, same as before
        // blending existed.
        // hasScrolled guard prevents this from firing on page load when scroll=0 but
        // a scroll div is already partially visible in the viewport.
        // playOnce auto scenes (e.g. an intro) are excluded: their playability must
        // not depend on the scrollbar — they always run to completion uninterrupted.
        if (!s.playOnce && hasScrolled && scrollTarget && scrollTarget.frame > s.toFrame + 0.5) {
          pausedAutoFrame = frame;
          anim.seekFrame(scrollTarget.frame);
          return;
        }
        // Play-once scenes hold their final frame after their first pass —
        // no further looping/pingponging until the page reloads.
        if (s.playOnce && s.done) return;
        // Ease speed down near a pingpong turnaround instead of advancing at
        // full speed right up to the instant direction flips. The camera is
        // typically still moving at a real, non-zero speed when it hits the
        // boundary (verified ~0.12 units/sec in this project's scenes), so an
        // instant reversal is a sharp velocity discontinuity — visible as a
        // snap/wobble right at the bounce — even though the underlying
        // keyframed path itself is perfectly smooth. PINGPONG_MIN_SPEED keeps
        // a speed floor so this never fully stalls at the boundary.
        let tickDt = dt;
        if (s.pingpong) {
          const distToBoundary = Math.min(frame - s.fromFrame, s.toFrame - frame);
          const ease = Math.max(PINGPONG_MIN_SPEED, Math.min(1, distToBoundary / PINGPONG_EASE_FRAMES));
          tickDt = dt * ease;
        }
        anim.tick(tickDt);
        const f = anim.frame;
        if (s.pingpong) {
          if (f >= s.toFrame) {
            anim.seekFrame(s.toFrame);
            if (s.playOnce) s.done = true;
            else            anim.direction = -1;
          } else if (f <= s.fromFrame) {
            anim.seekFrame(s.fromFrame);
            anim.direction = 1;
          }
        } else {
          if (f < s.fromFrame) anim.seekFrame(s.fromFrame);
          else if (s.playOnce && f >= s.toFrame) {
            anim.seekFrame(s.toFrame);
            s.done = true;
          }
        }
        return;
      }

      // Scroll scene: the scroll position is the source of truth, regardless of
      // which scene-slot `s` currently sits in (scrollTarget can race ahead of
      // anim.frame across degenerate/zero-length scenes).
      if (scrollTarget) {
        anim.seekFrame(scrollTarget.frame);
      } else if (!s.el) {
        anim.seekFrame(s.fromFrame);
      }
    };

    return () => {
      window.removeEventListener('scroll', onScroll);
      viewer.animTickOverride = null;
      viewer._currentSceneName = null;
      viewer._sceneBlend = null;
    };
  }

  // Pre-populate scene configs from opts.scenes
  if (scenes && typeof scenes === 'object') {
    window.__hsSceneConfigs = window.__hsSceneConfigs || {};
    Object.assign(window.__hsSceneConfigs, scenes);
  }

  // Pre-populate mask configs from opts.masks and apply feather overrides.
  if (masks && typeof masks === 'object') {
    window.__hsMaskConfigs = window.__hsMaskConfigs || {};
    Object.assign(window.__hsMaskConfigs, masks);
    for (const [name, cfg] of Object.entries(masks)) {
      if (cfg && typeof cfg.feather === 'number') viewer.setMaskFeather(name, cfg.feather);
    }
  }

  // Apply SH degree: explicit global sh option takes priority, then max of per-scene values,
  // capped by the device-tier quality preset (e.g. 'low' disables SH entirely).
  const effectiveSh = sh != null ? sh
    : scenes
      ? Object.values(scenes).reduce((m, c) => Math.max(m, c.sh ?? 0), viewer._shDegree)
      : viewer._shDegree;
  const cappedSh = Math.min(effectiveSh, caps.shDegreeCap);
  if (cappedSh !== viewer._shDegree) viewer._shDegree = cappedSh;

  // ── Boot ────────────────────────────────────────────────────────────────────
  viewer.init()
    .then(async () => {
      viewer.start();
      let anim;
      // Clip asset file(s) are fully independent of the main animation/parts
      // load sequence below, so kick them off in parallel regardless of
      // which branch (sequential vs parallel) the rest takes.
      const clipUrls  = clips ? (Array.isArray(clips) ? clips : [clips]) : [];
      const clipLoads = Promise.all(clipUrls.map(loadClips));
      if (partsDir && animSrc) {
        // Sequential: load animation first, then derive + load parts from object IDs
        anim = await loadAnim(animSrc);
        if (anim?.objects.length) {
          const dir = partsDir.replace(/\/?$/, '/');
          let autoPartsMap = Object.fromEntries(
            anim.objects.map(obj => [obj.id, `${dir}${splatNameFromId(obj.id)}${partsExt}`])
          );
          autoPartsMap = await resolvePartsLod(autoPartsMap, caps.lod);
          await loadPartsMap(autoPartsMap);
        }
      } else {
        const loads = [];
        if (parts)    loads.push(resolvePartsLod(parts, caps.lod).then(loadPartsMap));
        else if (src) loads.push(resolveLodUrl(src, caps.lod).then(load));
        if (animSrc)  loads.push(loadAnim(animSrc).then(a => { anim = a; }));
        await Promise.all(loads);
      }
      await clipLoads;
      // All loads done — show ready and call onLoad with animation guaranteed available
      showReady();
      onLoad?.();
      setupScrollPlayback(anim ?? viewer._animation);
    })
    .catch(err => {
      // scene/parts errors already showed their own error UI; only handle
      // unexpected failures here (e.g. viewer.init() failing)
      if (!navigator.gpu) showError('WebGPU not supported. Use Chrome 113+ or Edge 113+.');
    });

  // Inject overlay when ?hs is in the URL (dev only). The full art-direction
  // editor is desktop-only; on touch/narrow viewports load the lightweight
  // stats overlay instead (see holosplat/stats.js).
  if (typeof location !== 'undefined' &&
      new URLSearchParams(location.search).has('hs') &&
      !document.getElementById('__hs-script')) {
    const isMobile = window.matchMedia('(pointer: coarse)').matches
      || window.matchMedia('(max-width: 768px)').matches
      || /Android|iPhone|iPad|iPod/i.test(navigator.userAgent || '');
    const s = document.createElement('script');
    s.id  = '__hs-script';
    s.src = isMobile ? '/holosplat/stats.js' : '/holosplat/editor.js';
    document.head.appendChild(s);
  }

  return api;
}

// ── Data-attribute auto-init ─────────────────────────────────────────────────

function autoInit() {
  document.querySelectorAll('[data-holosplat]').forEach(el => {
    if (el._hsPlayer) return;
    const src      = el.getAttribute('data-holosplat')       || undefined;
    const anim     = el.getAttribute('data-holosplat-anim')  || undefined;
    const partsDir = el.getAttribute('data-holosplat-parts') || undefined;
    el._hsPlayer = player(el, { src, animation: anim, partsDir });
  });
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInit);
  } else {
    autoInit();
  }
}
