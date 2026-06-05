/**
 * HoloSplat overlay editor.
 * Injected by player.js when ?hs is in the URL. Connects to window.__hsPlayers
 * to access the live player instance without creating a second viewer.
 *
 * Config lives in data-hs-* attrs on the container element:
 *   data-hs-scene       scene URL
 *   data-hs-animation   animation JSON URL
 *   data-hs-flip-y      "true" / "false"
 *
 * Save writes those attrs back to the HTML file via /hs-api/file.
 */
(function () {
  if (window.__hsEditor) return;
  window.__hsEditor = true;

  // ── State ───────────────────────────────────────────────────────────────────
  const S = {
    entry:      null,   // { root, api, viewer }
    markers:    {},
    frameCount: 0,
    panelOpen:  true,
    apiOnline:  false,
    dirty:      false,
    scrubbing:  false,
    filePath:   location.pathname.replace(/^\//, '') || 'index.html',
  };

  // ── CSS ─────────────────────────────────────────────────────────────────────
  const CSS = `
    #__hs-ed { position:fixed;top:0;right:0;height:100%;z-index:99999;display:flex;pointer-events:none; }
    #__hs-ed * { box-sizing:border-box;margin:0;padding:0; }
    #__hs-tab {
      pointer-events:auto;align-self:flex-start;margin-top:48px;
      writing-mode:vertical-rl;transform:rotate(180deg);
      background:#1a1a1a;border:1px solid #333;border-right:none;
      color:#666;font-size:0.62rem;font-weight:700;letter-spacing:0.1em;font-family:system-ui,sans-serif;
      padding:10px 5px;cursor:pointer;border-radius:3px 0 0 3px;user-select:none;
    }
    #__hs-tab:hover { color:#ccc; }
    #__hs-panel {
      pointer-events:auto;width:340px;height:100%;
      background:#1a1a1a;border-left:1px solid #2e2e2e;
      display:flex;flex-direction:column;overflow:hidden;
      font-family:system-ui,sans-serif;font-size:0.78rem;color:#eee;
    }
    #__hs-panel.closed { display:none; }

    /* toolbar */
    #__hs-tb {
      display:flex;align-items:center;gap:6px;flex-shrink:0;
      padding:6px 10px;background:#141414;border-bottom:1px solid #2e2e2e;
    }
    #__hs-tb h1 { font-size:0.88rem;font-weight:700;white-space:nowrap; }
    #__hs-dot { color:#ff9a44;display:none;margin-left:2px; }
    #__hs-st {
      flex:1;font-size:0.68rem;color:#555;
      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
    }
    #__hs-st.err { color:#f87; }
    #__hs-badge {
      font-size:0.62rem;padding:2px 6px;border-radius:8px;flex-shrink:0;
      background:#1e2e1e;border:1px solid #2a4a2a;color:#5a9a5a;white-space:nowrap;
    }
    #__hs-badge.off { background:#2e1e1e;border-color:#4a2a2a;color:#9a5a5a; }
    .__hs-btn {
      background:#242424;border:1px solid #333;color:#aaa;
      padding:3px 8px;border-radius:3px;cursor:pointer;font-size:0.7rem;
      white-space:nowrap;flex-shrink:0;font-family:inherit;
    }
    .__hs-btn:hover { background:#2e2e2e;color:#eee; }
    .__hs-btn.pri { background:#1e3555;border-color:#2a5599;color:#aac; }
    .__hs-btn.pri:hover { background:#254a80; }
    .__hs-btn:disabled { opacity:0.35;cursor:default; }

    /* pane */
    .__hs-pane { border-bottom:1px solid #222;flex-shrink:0; }
    .__hs-pt {
      font-size:0.62rem;font-weight:700;text-transform:uppercase;
      letter-spacing:0.07em;color:#555;padding:5px 10px 3px;
    }

    /* files */
    .__hs-fr { display:flex;align-items:center;gap:4px;padding:3px 8px 5px; }
    .__hs-fl { font-size:0.67rem;color:#555;width:32px;flex-shrink:0; }
    .__hs-fr input {
      flex:1;min-width:0;background:#1e1e1e;border:1px solid #2e2e2e;color:#ccc;
      padding:3px 6px;border-radius:3px;font-size:0.7rem;font-family:inherit;
    }
    .__hs-fr input:focus { border-color:#3a7aff;outline:none; }

    /* info grid */
    .__hs-grid {
      display:grid;grid-template-columns:1fr 1fr;gap:3px;padding:4px 8px 7px;
    }
    .__hs-grid.cols3 { grid-template-columns:1fr 1fr 1fr; }
    .__hs-cell { background:#1e1e1e;border-radius:3px;padding:3px 6px; }
    .__hs-lbl { font-size:0.57rem;color:#444;text-transform:uppercase; }
    .__hs-val { font-size:0.72rem;color:#aaa;font-variant-numeric:tabular-nums; }

    /* markers */
    #__hs-mklist { overflow-y:auto;max-height:140px; }
    .__hs-mkrow {
      display:flex;justify-content:space-between;align-items:center;
      padding:3px 10px;cursor:pointer;
    }
    .__hs-mkrow:hover { background:#1e1e1e; }
    .__hs-mkn { font-size:0.7rem;color:#aaa; }
    .__hs-mkf { font-size:0.65rem;color:#444;font-variant-numeric:tabular-nums; }
    .__hs-empty { padding:7px 10px;color:#3a3a3a;font-size:0.7rem;font-style:italic; }

    /* scrubber */
    #__hs-scrub {
      display:flex;align-items:center;gap:6px;
      padding:6px 10px;background:#141414;border-top:1px solid #222;
      flex-shrink:0;margin-top:auto;
    }
    #__hs-range { flex:1;accent-color:#3a7aff;cursor:pointer; }
    #__hs-flbl {
      font-size:0.67rem;color:#555;white-space:nowrap;
      min-width:96px;text-align:right;font-variant-numeric:tabular-nums;
    }
  `;

  // ── HTML template ────────────────────────────────────────────────────────────
  const HTML = `
<div id="__hs-ed">
  <button id="__hs-tab" title="Toggle HoloSplat editor">HS EDITOR</button>
  <div id="__hs-panel">

    <div id="__hs-tb">
      <h1>HoloSplat<span id="__hs-dot">●</span></h1>
      <span id="__hs-st">connecting…</span>
      <span id="__hs-badge" class="off">API offline</span>
      <button class="__hs-btn pri" id="__hs-save">Save</button>
    </div>

    <div class="__hs-pane">
      <div class="__hs-pt">Files</div>
      <div class="__hs-fr">
        <span class="__hs-fl">Scene</span>
        <input id="__hs-sc" placeholder="/scenes/scene.spz" spellcheck="false">
        <button class="__hs-btn" id="__hs-rsc" title="Reload scene">↺</button>
      </div>
      <div class="__hs-fr">
        <span class="__hs-fl">Anim</span>
        <input id="__hs-an" placeholder="/scenes/anim.json" spellcheck="false">
        <button class="__hs-btn" id="__hs-ran" title="Reload animation">↺</button>
      </div>
    </div>

    <div class="__hs-pane">
      <div class="__hs-pt">Scene</div>
      <div class="__hs-grid">
        <div class="__hs-cell"><div class="__hs-lbl">Splats</div><div class="__hs-val" id="__hs-nsplats">—</div></div>
        <div class="__hs-cell"><div class="__hs-lbl">Status</div><div class="__hs-val" id="__hs-loadst">—</div></div>
        <div class="__hs-cell"><div class="__hs-lbl">FlipY</div><div class="__hs-val" id="__hs-flipy">—</div></div>
        <div class="__hs-cell"><div class="__hs-lbl">Camera</div><div class="__hs-val" id="__hs-cammode">—</div></div>
      </div>
    </div>

    <div class="__hs-pane">
      <div class="__hs-pt">Camera</div>
      <div class="__hs-grid cols3">
        <div class="__hs-cell"><div class="__hs-lbl">theta °</div><div class="__hs-val" id="__hs-cth">—</div></div>
        <div class="__hs-cell"><div class="__hs-lbl">phi °</div><div class="__hs-val" id="__hs-cph">—</div></div>
        <div class="__hs-cell"><div class="__hs-lbl">radius</div><div class="__hs-val" id="__hs-cr">—</div></div>
        <div class="__hs-cell"><div class="__hs-lbl">tx</div><div class="__hs-val" id="__hs-ctx">—</div></div>
        <div class="__hs-cell"><div class="__hs-lbl">ty</div><div class="__hs-val" id="__hs-cty">—</div></div>
        <div class="__hs-cell"><div class="__hs-lbl">tz</div><div class="__hs-val" id="__hs-ctz">—</div></div>
      </div>
    </div>

    <div class="__hs-pane">
      <div class="__hs-pt">Markers <span style="font-weight:400;text-transform:none;letter-spacing:0;color:#333">(click to seek)</span></div>
      <div id="__hs-mklist"><div class="__hs-empty">No animation loaded</div></div>
    </div>

    <div id="__hs-scrub">
      <input type="range" id="__hs-range" min="0" max="0" value="0" step="1">
      <span id="__hs-flbl">no animation</span>
    </div>

  </div>
</div>`;

  // ── DOM helpers ─────────────────────────────────────────────────────────────
  const el  = id => document.getElementById('__hs-' + id);
  const fmt = n  => typeof n === 'number' ? n.toFixed(2) : '—';

  function setStatus(msg, err = false) {
    el('st').textContent = msg;
    el('st').className   = err ? 'err' : '';
  }

  function setDirty(v) {
    S.dirty = v;
    el('dot').style.display = v ? 'inline' : 'none';
  }

  // ── API ─────────────────────────────────────────────────────────────────────
  async function apiCheck() {
    try {
      S.apiOnline = (await fetch('/hs-api/ls')).ok;
    } catch { S.apiOnline = false; }
    el('badge').textContent = S.apiOnline ? 'API online' : 'API offline';
    el('badge').className   = S.apiOnline ? '' : 'off';
    if (!S.apiOnline) {
      el('save').disabled = true;
      el('save').title    = 'Start server.py to enable save';
    }
  }

  // ── Save ─────────────────────────────────────────────────────────────────────
  async function save() {
    if (!S.apiOnline || !S.entry) return;
    const sceneVal = el('sc').value.trim();
    const animVal  = el('an').value.trim();
    setStatus('Saving…');
    try {
      const res  = await fetch(`/hs-api/file?path=${encodeURIComponent(S.filePath)}`);
      if (!res.ok) throw new Error(`Could not read ${S.filePath}`);
      let html = await res.text();

      // Update or insert data-hs-scene / data-hs-animation on the container element.
      // Matches the opening tag of the element with the container's id.
      const id = S.entry.root.id;
      if (id) {
        html = setAttr(html, id, 'data-hs-scene',     sceneVal);
        html = setAttr(html, id, 'data-hs-animation', animVal);
      }

      await fetch(`/hs-api/file?path=${encodeURIComponent(S.filePath)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'text/html; charset=utf-8' },
        body: html,
      });
      setStatus('Saved');
      setDirty(false);
    } catch (e) {
      setStatus(`Save failed: ${e.message}`, true);
    }
  }

  // Replace data-attr value inside a tag with a given id, or add it if absent.
  function setAttr(html, id, attr, value) {
    // Try to replace an existing attr value
    const replaced = html.replace(
      new RegExp(`(id="${id}"[^>]*?)${attr}="[^"]*"`, 's'),
      `$1${attr}="${value}"`
    );
    if (replaced !== html) return replaced;
    // Attr doesn't exist yet — insert after the id attr
    return html.replace(`id="${id}"`, `id="${id}" ${attr}="${value}"`);
  }

  // ── Connect to player ────────────────────────────────────────────────────────
  function connectEntry(entry) {
    S.entry = entry;
    const { root, api, viewer } = entry;

    el('sc').value = root.getAttribute('data-hs-scene')     || '';
    el('an').value = root.getAttribute('data-hs-animation') || '';
    el('flipy').textContent = viewer._flipY ? 'yes' : 'no';

    if (api.animation) {
      connectAnim(api.animation);
    } else {
      // Poll until animation loads
      const poll = setInterval(() => {
        if (api.animation) { clearInterval(poll); connectAnim(api.animation); }
      }, 250);
    }

    setInterval(tick, 250);
    setStatus(`Connected · ${S.filePath}`);
  }

  function connectAnim(anim) {
    S.markers    = anim.markers    || {};
    S.frameCount = anim.frameCount || 0;
    el('range').max   = Math.max(0, S.frameCount - 1);
    el('range').value = 0;
    el('flbl').textContent = `0 / ${S.frameCount}`;
    renderMarkers();
    setStatus(`${S.frameCount} frames · ${Object.keys(S.markers).length} markers`);
  }

  function tick() {
    if (!S.entry) return;
    const { api, viewer } = S.entry;
    const cam = api.camera;

    // Scene info
    el('nsplats').textContent = viewer._numSplats ? viewer._numSplats.toLocaleString() : '0';
    el('loadst').textContent  = viewer._sceneReady ? 'ready' : 'loading…';
    el('cammode').textContent = viewer._cameraFree ? 'free' : 'anim';

    // Camera
    if (cam) {
      el('cth').textContent = fmt(cam.theta  * 180 / Math.PI) + '°';
      el('cph').textContent = fmt(cam.phi    * 180 / Math.PI) + '°';
      el('cr').textContent  = fmt(cam.radius);
      el('ctx').textContent = fmt(cam.target[0]);
      el('cty').textContent = fmt(cam.target[1]);
      el('ctz').textContent = fmt(cam.target[2]);
    }

    // Scrubber follow
    const anim = api.animation;
    if (anim && !S.scrubbing) {
      const f = Math.round(anim.frame || 0);
      el('range').value     = f;
      el('flbl').textContent = `${f} / ${S.frameCount}`;
    }

    // Auto-connect animation if it just loaded
    if (anim && S.frameCount === 0) connectAnim(anim);
  }

  // ── Markers ──────────────────────────────────────────────────────────────────
  function renderMarkers() {
    const rows = Object.entries(S.markers).sort((a, b) => a[1] - b[1]);
    const list = el('mklist');
    list.innerHTML = rows.length
      ? rows.map(([n, f]) =>
          `<div class="__hs-mkrow" data-f="${f}">
             <span class="__hs-mkn">${n}</span>
             <span class="__hs-mkf">${f}</span>
           </div>`).join('')
      : '<div class="__hs-empty">No markers</div>';
    list.querySelectorAll('.__hs-mkrow').forEach(row =>
      row.addEventListener('click', () => seekFrame(+row.dataset.f))
    );
  }

  // ── Scrubber ─────────────────────────────────────────────────────────────────
  function seekFrame(n) {
    const anim = S.entry?.api.animation;
    if (anim) { anim.seekFrame(n); S.entry.api.setAnimationPaused(true); }
    el('range').value     = n;
    el('flbl').textContent = `${Math.round(n)} / ${S.frameCount}`;
  }

  // ── Reload helpers ───────────────────────────────────────────────────────────
  async function reloadScene() {
    const url = el('sc').value.trim();
    if (!url || !S.entry) return;
    setStatus('Loading scene…');
    try {
      await S.entry.api.load(url);
      S.entry.root.setAttribute('data-hs-scene', url);
      setStatus('Scene ready');
      setDirty(true);
    } catch (e) { setStatus(e.message, true); }
  }

  async function reloadAnim() {
    const url = el('an').value.trim();
    if (!url || !S.entry) return;
    setStatus('Loading animation…');
    try {
      const anim = await S.entry.api.loadAnim(url);
      S.entry.root.setAttribute('data-hs-animation', url);
      if (anim) connectAnim(anim);
      setDirty(true);
    } catch (e) { setStatus(e.message, true); }
  }

  // ── Init ─────────────────────────────────────────────────────────────────────
  function init() {
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);

    const wrap = document.createElement('div');
    wrap.innerHTML = HTML;
    document.body.appendChild(wrap.firstElementChild);

    // Toggle panel
    el('tab').addEventListener('click', () => {
      S.panelOpen = !S.panelOpen;
      el('panel').classList.toggle('closed', !S.panelOpen);
    });

    // Save / reload
    el('save').addEventListener('click', save);
    el('rsc').addEventListener('click',  reloadScene);
    el('ran').addEventListener('click',  reloadAnim);

    // Dirty on input changes
    el('sc').addEventListener('input', () => setDirty(true));
    el('an').addEventListener('input', () => setDirty(true));

    // Ctrl+S
    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); save(); }
    });

    // Scrubber
    el('range').addEventListener('mousedown', () => { S.scrubbing = true; });
    el('range').addEventListener('mouseup',   () => { S.scrubbing = false; });
    el('range').addEventListener('input',     e  => seekFrame(+e.target.value));

    apiCheck();

    // Find player — it may already be registered or arrive shortly
    const players = window.__hsPlayers || [];
    if (players.length) {
      connectEntry(players[0]);
    } else {
      setStatus('Waiting for player…');
      const poll = setInterval(() => {
        const ps = window.__hsPlayers || [];
        if (ps.length) { clearInterval(poll); connectEntry(ps[0]); }
      }, 200);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
