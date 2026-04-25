/**
 * HoloSplat Scroll Scene — scroll-driven animation controller.
 *
 * Binds a HoloSplat player's animation playback to the scroll position of the
 * page, using a declarative HTML structure of acts and holds.
 *
 * ── HTML structure ───────────────────────────────────────────────────────────
 *
 *   <div class="hs-scene">
 *
 *     <!-- Sticky canvas: stays in view while scrolling through the track -->
 *     <div class="hs-stage">
 *       <!-- player() targets this element -->
 *     </div>
 *
 *     <!-- Scroll track: normal-flow divs that create scroll height.
 *          Uses margin-top:-100vh to visually overlap the sticky stage. -->
 *     <div class="hs-track">
 *
 *       <!-- Act: maps a Blender marker range to scroll distance.
 *            The larger the height, the slower / more scroll needed to pass
 *            through the frame range.
 *
 *            data-from  — start marker name OR frame number (default: 0)
 *            data-to    — end   marker name OR frame number (default: last frame)
 *            data-loop  — repeat the frame range N times as progress goes 0→1.
 *                         "true" / "" = 1×, "3" = 3×.
 *                         Reverse loops (data-from > data-to) are supported.
 *       -->
 *       <div class="hs-act"
 *            data-from="intro"
 *            data-to="desk_reveal"
 *            style="height:300vh">
 *
 *         <!-- Caption: fades in at data-at progress (0..1) through this act.
 *              data-at defaults to 0 (immediately visible). -->
 *         <div class="hs-caption" data-at="0.3">Introducing the scene</div>
 *
 *       </div>
 *
 *       <!-- Hold: freeze at one frame while user scrolls past (reading time). -->
 *       <div class="hs-hold"
 *            data-frame="desk_reveal"
 *            style="height:120vh">
 *         <div class="hs-caption">Notice the details</div>
 *       </div>
 *
 *       <!-- Reverse: data-from > data-to plays the range backwards. -->
 *       <div class="hs-act"
 *            data-from="desk_reveal"
 *            data-to="intro"
 *            style="height:200vh">
 *       </div>
 *
 *       <!-- Loop 3×: plays intro→zoom three times as user scrolls through. -->
 *       <div class="hs-act"
 *            data-from="intro"
 *            data-to="zoom"
 *            data-loop="3"
 *            style="height:300vh">
 *       </div>
 *
 *     </div>
 *   </div>
 *
 * ── JS usage ─────────────────────────────────────────────────────────────────
 *
 *   // Programmatic:
 *   const p = HoloSplat.player('.hs-stage', { src: '...', animation: '...' });
 *   HoloSplat.scrollScene(document.querySelector('.hs-scene'), p);
 *
 *   // Data-attribute auto-init (no JS needed):
 *   // Place data-holosplat / data-holosplat-anim on the .hs-stage; the scroll
 *   // scene is wired up automatically after the player initialises.
 *
 * ── Marker names ─────────────────────────────────────────────────────────────
 *
 *   data-from / data-to / data-frame accept either:
 *     • A Blender timeline marker name exported in the animation JSON
 *       (e.g. "intro", "desk_reveal"). Marker names are prefixed with
 *       "data-marker-" automatically in Blender by adding them as
 *       timeline markers named exactly as you want to reference them here.
 *     • A bare frame number (e.g. "0", "72").
 *
 *   If a name is not found in the markers dict a warning is logged and 0 is used.
 */

// ── CSS (injected once per page) ─────────────────────────────────────────────

const SCROLL_CSS = `
.hs-scene {
  position: relative;
  width: 100%;
}
/* .hs-stage also carries .hs-player (position:relative) from player.js.
   !important ensures sticky wins regardless of injection order.
   100vw fills the full viewport regardless of parent padding/margin. */
.hs-stage {
  position: sticky !important;
  top: 0;
  left: 0;
  width: 100vw !important;
  height: 100vh !important;
  z-index: 1;
  overflow: hidden;
}
.hs-stage canvas {
  position: absolute !important;
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
  display: block;
}
.hs-track {
  /* Overlap the sticky stage so acts start scrolling at the same time the
     canvas appears — not after it. */
  margin-top: -100vh;
  position: relative;
  /* Clicks pass through to the canvas; restore per-element as needed. */
  pointer-events: none;
  z-index: 2;
}
.hs-act,
.hs-hold {
  position: relative;
}
.hs-caption {
  pointer-events: auto;
}
.hs-caption--hidden {
  opacity: 0;
  pointer-events: none;
}
`;

let sceneCssInjected = false;
function injectScrollCss() {
  if (sceneCssInjected || typeof document === 'undefined') return;
  sceneCssInjected = true;
  const s = document.createElement('style');
  s.textContent = SCROLL_CSS;
  document.head.appendChild(s);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Resolve a marker name or raw frame string to a frame number.
 * Falls back to 0 with a warning if the name is unknown.
 */
function resolveFrame(value, markers, fallback = 0) {
  if (value === undefined || value === null || value === '') return fallback;
  const str = String(value).trim();
  if (Object.prototype.hasOwnProperty.call(markers, str)) return markers[str];
  const n = parseFloat(str);
  if (!isNaN(n)) return n;
  console.warn(`[HoloSplat] scrollScene: unknown marker "${str}" — using ${fallback}`);
  return fallback;
}

/**
 * Parse a single act/hold element into a descriptor object.
 * @param {HTMLElement} el
 * @param {object}      markers  { name: frame }
 * @param {number}      lastFrame  fallback for data-to when omitted
 */
function parseAct(el, markers, lastFrame) {
  const isHold = el.classList.contains('hs-hold');

  if (isHold) {
    return {
      el,
      type: 'hold',
      frame: resolveFrame(el.dataset.frame, markers, 0),
      captions: parseCaptions(el),
    };
  }

  const fromAttr   = el.dataset.from ?? '';
  const isPingpong = fromAttr === 'pingpong-start';

  const from = resolveFrame(fromAttr,        markers, 0);
  const to   = resolveFrame(el.dataset.to,   markers, lastFrame);

  // data-loop: number of times to cycle the frame range as progress 0→1.
  // "true" / "" = 1 (seamless, one full cycle); "3" = 3 repetitions.
  let loop = 0;
  if (el.dataset.loop !== undefined) {
    const v = el.dataset.loop.trim();
    loop = (v === '' || v === 'true') ? 1 : (Math.max(1, parseFloat(v) || 1));
  }

  return {
    el,
    type: isPingpong ? 'pingpong' : 'act',
    from,
    to,
    loop,
    captions: parseCaptions(el),
  };
}

function parseCaptions(el) {
  return [...el.querySelectorAll('.hs-caption')].map(c => ({
    el: c,
    at: parseFloat(c.dataset.at ?? '0'),
  }));
}

/**
 * Compute the target frame for a given 0..1 progress through an act.
 */
function actFrame(act, progress) {
  if (act.type === 'hold') return act.frame;
  // Pingpong acts own their own playback; only commit to end-frame when
  // fully scrolled past so the next act gets a clean starting point.
  if (act.type === 'pingpong') return progress >= 1 ? act.to : act.from;

  const { from, to, loop } = act;
  const range = to - from;
  if (range === 0) return from;

  let p = progress;
  if (loop > 0) {
    // Tile the range `loop` times; modulo keeps it within one cycle.
    p = (progress * loop) % 1;
  }

  return from + p * range;
}

/**
 * Returns the scroll progress (0..1) of an element relative to the top of
 * the viewport.  0 = element top is at viewport top;
 *                1 = element bottom is at viewport top (fully scrolled past).
 */
function getProgress(el) {
  const top = el.getBoundingClientRect().top;
  const h   = el.offsetHeight || 1;
  return Math.max(0, Math.min(1, -top / h));
}

// ── Pingpong transition helpers ───────────────────────────────────────────────

function lerp(a, b, t) { return a + (b - a) * t; }
function smoothstep(a, b, x) {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

// ── scrollScene ───────────────────────────────────────────────────────────────

/**
 * Attach scroll-driven animation control to an .hs-scene element.
 *
 * @param {HTMLElement} sceneEl        The .hs-scene container element.
 * @param {object}      playerInstance A HoloSplat player() return value.
 * @returns {{ rebuild: function, destroy: function }}
 */
export function scrollScene(sceneEl, playerInstance, opts = {}) {
  injectScrollCss();

  const track = sceneEl.querySelector('.hs-track');
  if (!track) {
    console.warn('[HoloSplat] scrollScene: .hs-track not found inside scene');
    return { rebuild() {}, destroy() {} };
  }

  let acts = [];

  // ── Pingpong playback ────────────────────────────────────────────────────────
  // Autonomous time-driven pingpong between two frame bounds.
  // Active while the user is scrolled into a pingpong act; stops automatically
  // when they scroll out, handing control back to the scroll-seek path.

  let ppBlend         = Math.max(0, Math.min(0.49, opts.pingpongBlend ?? 0.15));
  let ppActive        = false;
  let ppDir           = 1;
  let ppFrame         = 0;
  let ppStart         = 0;
  let ppEnd           = 0;
  let ppLastTime      = null;
  let ppRafId         = null;
  let ppBlendProgress = 0;  // updated each scroll tick; drives entry/exit blend

  // fromEnd=true when entering from below (scrolling up): start at ppEnd, go backward.
  function startPingpong(startFrame, endFrame, fromEnd = false) {
    if (ppActive) return;
    ppActive   = true;
    ppStart    = startFrame;
    ppEnd      = endFrame;
    ppDir      = fromEnd ? -1 : 1;
    ppFrame    = fromEnd ? endFrame : startFrame;
    ppLastTime = null;
    ppRafId    = requestAnimationFrame(tickPingpong);
  }

  function stopPingpong() {
    ppActive = false;
    if (ppRafId !== null) { cancelAnimationFrame(ppRafId); ppRafId = null; }
  }

  function tickPingpong(now) {
    if (!ppActive) return;
    ppRafId = requestAnimationFrame(tickPingpong);
    if (ppLastTime === null) { ppLastTime = now; return; }

    const dt  = (now - ppLastTime) / 1000;
    ppLastTime = now;

    const fps = playerInstance.animation?.fps ?? 24;
    ppFrame  += ppDir * fps * dt;
    if (ppFrame >= ppEnd)   { ppFrame = ppEnd;   ppDir = -1; }
    if (ppFrame <= ppStart) { ppFrame = ppStart;  ppDir =  1; }

    // Entry/exit blend: cross-fade between boundary frame and live pingpong frame
    // so the camera never pops when the user scrolls into or out of this section.
    // The blend is symmetric — the low-p zone handles entry from above and exit
    // going up; the high-p zone handles exit going down and entry from below.
    let frame = ppFrame;
    const p   = ppBlendProgress;
    if (p < ppBlend) {
      frame = lerp(ppStart, ppFrame, smoothstep(0, ppBlend, p));
    } else if (p > 1 - ppBlend) {
      frame = lerp(ppFrame, ppEnd, smoothstep(1 - ppBlend, 1, p));
    }

    playerInstance.animation.seekFrame(frame);
  }

  // ── Act building ────────────────────────────────────────────────────────────

  function buildActs() {
    const anim      = playerInstance.animation;
    const markers   = anim?.markers ?? {};
    const lastFrame = anim ? anim.frameCount - 1 : 0;
    acts = [...track.children].map(el => parseAct(el, markers, lastFrame));
    // Hand playback control to scroll — stop the auto-play loop.
    if (anim) playerInstance.setAnimationPaused(true);
  }

  // ── Scroll update ────────────────────────────────────────────────────────────

  function update() {
    if (!playerInstance.animation || !acts.length) return;

    // Walk acts in order; find the one currently in progress.
    // After a fully-scrolled act, keep its frame as the candidate so the last
    // state holds until the next act starts.
    let frame = actFrame(acts[0], 0);
    let activeAct      = acts[0];
    let activeProgress = 0;

    for (const act of acts) {
      const p = getProgress(act.el);
      if (p <= 0) break;              // haven't reached this act yet

      frame          = actFrame(act, p);
      activeAct      = act;
      activeProgress = p;

      if (p < 1) break;              // this is the currently-active act
      // p === 1: fully scrolled past — continue to check next act
    }

    // Pingpong acts own their own rAF loop — scroll only starts/stops it.
    const inPingpong = activeAct.type === 'pingpong'
      && activeProgress > 0 && activeProgress < 1;

    if (inPingpong) {
      ppBlendProgress = activeProgress;
      if (!ppActive) {
        // Entering from below (scrolling up) if we're already past halfway.
        startPingpong(activeAct.from, activeAct.to, activeProgress > 0.5);
      }
    } else {
      stopPingpong();
      playerInstance.animation.seekFrame(frame);
    }

    // ── Debug: log on act change ────────────────────────────────────────────
    const activeId = activeAct?.el?.id ?? '—';
    if (update._activeId !== activeId) {
      update._activeId = activeId;
      const act = activeAct;
      const range = act.type === 'hold'
        ? `hold @ frame ${act.frame}`
        : act.type === 'pingpong'
          ? `pingpong [${act.from} ↔ ${act.to}]`
          : `act [${act.from} → ${act.to}]`;
      console.log(`[HoloSplat] active: ${activeId} (${range}) | current frame: ${frame.toFixed(1)}`);
    }

    // ── Caption visibility ──────────────────────────────────────────────────
    // All acts: hide captions of acts we haven't reached or have left.
    for (const act of acts) {
      if (!act.captions.length) continue;
      const isActive = act === activeAct;
      for (const cap of act.captions) {
        const visible = isActive && activeProgress >= cap.at;
        cap.el.classList.toggle('hs-caption--hidden', !visible);
      }
    }
  }

  // ── Throttle with rAF ────────────────────────────────────────────────────────

  let rafPending = false;
  function onScroll() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => { rafPending = false; update(); });
  }

  // ── Init ─────────────────────────────────────────────────────────────────────

  window.addEventListener('scroll', onScroll, { passive: true });

  // Wait for the animation to load (it's fetched async after player init),
  // then immediately pause auto-play and seek to the correct scroll position.
  // Using rAF so we don't block and the check is cheap.
  (function waitForAnim() {
    if (!playerInstance.animation) { requestAnimationFrame(waitForAnim); return; }
    // Disable orbit/wheel/touch controls so they don't swallow scroll events.
    playerInstance.camera.enabled = false;
    // Let the caller set data-from/to/frame attributes BEFORE we parse them.
    if (opts.onReady) opts.onReady(playerInstance.animation);
    buildActs();
    update();
  })();

  // ── Public API ───────────────────────────────────────────────────────────────
  return {
    /**
     * Re-read act elements from the DOM and re-resolve marker names.
     * Call after programmatic DOM changes to .hs-track.
     */
    rebuild() { stopPingpong(); buildActs(); update(); },

    get pingpongBlend()    { return ppBlend; },
    setPingpongBlend(v)    { ppBlend = Math.max(0, Math.min(0.49, v)); },

    destroy() {
      stopPingpong();
      window.removeEventListener('scroll', onScroll);
      playerInstance.camera.enabled = true;
      playerInstance.setAnimationPaused(false);
    },
  };
}

// ── Data-attribute auto-init ─────────────────────────────────────────────────
//
// Looks for .hs-scene elements that contain a .hs-stage with data-holosplat.
// The player() auto-init in player.js handles creating the player on .hs-stage;
// we wait for DOMContentLoaded (same event) and wire up scroll scenes after.

function autoInitScrollScenes() {
  document.querySelectorAll('.hs-scene').forEach(sceneEl => {
    if (sceneEl._hsScroll) return;

    const stage = sceneEl.querySelector('.hs-stage');
    if (!stage) return;

    // Ensure the stage has a player (created by player.js autoInit).
    if (!stage._hsPlayer) return;

    sceneEl._hsScroll = scrollScene(sceneEl, stage._hsPlayer);
  });
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInitScrollScenes);
  } else {
    autoInitScrollScenes();
  }
}
