/**
 * HoloSplat mobile stats overlay.
 * Injected by player.js when ?hs is in the URL on touch/narrow viewports —
 * the full art-direction editor (editor.js) is desktop-only. Connects to
 * window.__hsPlayers to read live performance numbers via viewer.getStats().
 *
 * Tap the panel to collapse it to a small dot.
 */
(function () {
  if (window.__hsStats) return;
  window.__hsStats = true;

  const CSS = `
    #__hs-stats {
      position:fixed;top:8px;left:8px;z-index:99999;
      background:rgba(20,20,20,.78);border:1px solid rgba(255,255,255,.12);
      border-radius:8px;padding:8px 10px;
      font:11px/1.5 ui-monospace,Menlo,Consolas,monospace;color:#ddd;
      backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);
      white-space:nowrap;user-select:none;-webkit-user-select:none;
    }
    #__hs-stats.collapsed {
      padding:0;width:14px;height:14px;border-radius:50%;
      background:rgba(255,255,255,.25);border:1px solid rgba(255,255,255,.3);
    }
    #__hs-stats.collapsed > * { display:none; }
    #__hs-stats .row { display:flex;justify-content:space-between;gap:14px; }
    #__hs-stats .lbl { color:#888; }
    #__hs-stats .val { color:#fff;font-weight:600; }
    #__hs-stats .val.fps-good { color:#5a9a5a; }
    #__hs-stats .val.fps-ok   { color:#d6b35a; }
    #__hs-stats .val.fps-bad  { color:#d65a5a; }
  `;

  function fmtCount(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return String(n);
  }

  function fpsClass(fps) {
    if (fps >= 50) return 'fps-good';
    if (fps >= 27) return 'fps-ok';
    return 'fps-bad';
  }

  function init() {
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);

    const el = document.createElement('div');
    el.id = '__hs-stats';
    el.innerHTML = `
      <div class="row"><span class="lbl">fps</span><span class="val" data-k="fps">–</span></div>
      <div class="row"><span class="lbl">tier</span><span class="val" data-k="tier">–</span></div>
      <div class="row"><span class="lbl">sort</span><span class="val" data-k="sort">–</span></div>
      <div class="row"><span class="lbl">splats</span><span class="val" data-k="splats">–</span></div>
      <div class="row"><span class="lbl">sh</span><span class="val" data-k="sh">–</span></div>
      <div class="row"><span class="lbl">px ratio</span><span class="val" data-k="px">–</span></div>
      <div class="row"><span class="lbl">scene</span><span class="val" data-k="scene">–</span></div>
    `;
    document.body.appendChild(el);

    el.addEventListener('click', () => el.classList.toggle('collapsed'));

    const fpsEl    = el.querySelector('[data-k="fps"]');
    const tierEl   = el.querySelector('[data-k="tier"]');
    const sortEl   = el.querySelector('[data-k="sort"]');
    const splatsEl = el.querySelector('[data-k="splats"]');
    const shEl     = el.querySelector('[data-k="sh"]');
    const pxEl     = el.querySelector('[data-k="px"]');
    const sceneEl  = el.querySelector('[data-k="scene"]');

    function update() {
      const entry = (window.__hsPlayers || [])[0];
      const stats = entry?.viewer?.getStats?.();
      if (stats) {
        const fps = Math.round(stats.fps);
        fpsEl.textContent = String(fps);
        fpsEl.className = 'val ' + fpsClass(fps);
        tierEl.textContent = stats.tier ?? '–';
        sortEl.textContent = stats.gpuSortFailed ? 'CPU (gpu failed)' : (stats.gpuSort ? 'GPU' : 'CPU');
        splatsEl.textContent = stats.activeSplats === stats.numSplats
          ? fmtCount(stats.numSplats)
          : `${fmtCount(stats.activeSplats)} / ${fmtCount(stats.numSplats)}`;
        shEl.textContent = String(stats.shDegree);
        pxEl.textContent = stats.pixelRatio.toFixed(2);
        sceneEl.textContent = stats.sceneName ?? '–';
      }
      setTimeout(update, 250);
    }
    update();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
