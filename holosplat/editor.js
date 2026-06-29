/**
 * HoloSplat overlay editor.
 * Injected by player.js when ?hs is in the URL. Connects to window.__hsPlayers
 * to access the live player instance without creating a second viewer.
 *
 * All config lives in the player() JS call in the source file.
 * Save writes back via /hs-api/js-* endpoints — no data-hs-* DOM attrs.
 */
(function () {
  if (window.__hsEditor) return;
  window.__hsEditor = true;

  // ── State ───────────────────────────────────────────────────────────────────
  const S = {
    entry:          null,   // { root, api, viewer }
    markers:        {},
    frameCount:     0,
    panelOpen:      true,
    apiOnline:      false,
    diskFiles:      null,   // Set<string> of every spz/ply/splat path under scenes/ (relative to project root) — from /hs-api/ls. Populated on connect and re-fetched only on explicit splats-dir change/reload. Never probed per-file.
    animEverPlayed: false,
    scrubbing:      false,
    partsKey:       '',    // sorted part-ID string; re-renders badges whenever loaded set changes
    sceneConfigs:   {},    // markerName → config object
    maskConfigs:    {},    // mask volume name → { feather }
    sceneCards:     {},    // markerName → { playBtn, barEl, from, to, frames } for rAF updates
    sceneLbl:       null,  // overlay label injected into player root
    activeMarker:   null,  // currently active marker name
    showFocalMarker: false, // debug toggle: draw a marker at the focal-point's screen position
    focalMarkerEl:  null,  // overlay marker injected into player root
    focalMarkerCb:  null,  // the "Visualize" toggle's checkbox, set in init()
    globalSh:         0,     // SH degree — global, not per-scene
    globalZIndex:     5,     // player z-index
    globalAaDilation: 0.15,  // anti-aliasing covariance dilation
    // Asset clip files (see export_holosplat_clips.py) — each entry:
    // { url, name, status: 'idle'|'loading'|'ok'|'error', clipIds }.
    // Persisted as just the url list (clips: [...] // hs-clips); name/status/
    // clipIds are derived at load time, not saved.
    assets: [],
  };
  let _apiReady;    // Promise set in init(); awaited before loadPageState so HTML read uses correct apiOnline
  let _saveTimer;     // debounce handle for saveScenesAttr
  let _maskSaveTimer; // debounce handle for saveMasksAttr
  let _aaSaveTimer;   // debounce handle for saveGlobalAaDilation
  let _assetsSaveTimer; // debounce handle for saveAssetsAttr

  // ── CSS ─────────────────────────────────────────────────────────────────────
  const CSS = `
    #__hs-ed {
      position:fixed;top:20px;right:20px;
      bottom:calc(var(--hs-tl-h, 160px) + 40px);
      z-index:99999;display:flex;pointer-events:none;transition:bottom .15s;
    }
    #__hs-ed * { box-sizing:border-box;margin:0; }
    #__hs-tab {
      pointer-events:auto;align-self:flex-start;margin-top:48px;
      writing-mode:vertical-rl;transform:rotate(180deg);
      background:#1a1a1a;border:1px solid #333;border-right:none;
      color:#999;font-size:0.875rem;font-weight:700;letter-spacing:0.1em;font-family:system-ui,sans-serif;
      padding:12px 7px;cursor:pointer;border-radius:3px 0 0 3px;user-select:none;
    }
    #__hs-tab:hover { color:#ccc; }
    #__hs-panel {
      pointer-events:auto;width:420px;height:100%;
      background:#1a1a1a;border:1px solid #2e2e2e;border-radius:20px;
      box-shadow:0 8px 32px rgba(0,0,0,.5);
      display:flex;flex-direction:column;overflow:hidden;
      font-family:system-ui,sans-serif;font-size:1rem;color:#eee;
    }
    #__hs-body {
      flex:1;overflow-y:auto;overflow-x:hidden;
      scrollbar-width:thin;scrollbar-color:#3a7aff #1a1a1a;
    }
    #__hs-body::-webkit-scrollbar { width:8px; }
    #__hs-body::-webkit-scrollbar-track { background:#1a1a1a; }
    #__hs-body::-webkit-scrollbar-thumb { background:#3a7aff;border-radius:4px; }
    #__hs-body::-webkit-scrollbar-thumb:hover { background:#5a9aff; }
    #__hs-body::-webkit-scrollbar-button,
    #__hs-body::-webkit-scrollbar-button:single-button,
    #__hs-body::-webkit-scrollbar-button:vertical:start,
    #__hs-body::-webkit-scrollbar-button:vertical:end,
    #__hs-body::-webkit-scrollbar-button:vertical:start:increment,
    #__hs-body::-webkit-scrollbar-button:vertical:end:increment,
    #__hs-body::-webkit-scrollbar-button:vertical:start:decrement,
    #__hs-body::-webkit-scrollbar-button:vertical:end:decrement {
      display:none;width:0;height:0;
    }
    #__hs-panel.closed { display:none; }
    #__hs-panel.minimized { height:auto; }
    #__hs-panel.minimized #__hs-body { display:none; }

    /* toolbar */
    #__hs-tb {
      display:flex;align-items:center;gap:8px;flex-shrink:0;
      padding:10px 28px;background:#141414;border-bottom:1px solid #2e2e2e;
    }
    #__hs-tb h1 { font-size:1.1rem;font-weight:700;white-space:nowrap; }
    #__hs-st {
      flex:1;font-size:0.875rem;color:#aaa;
      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
    }
    #__hs-st.err { color:#f87; }
    #__hs-badge {
      font-size:0.875rem;padding:3px 9px;border-radius:8px;flex-shrink:0;
      background:#1e2e1e;border:1px solid #2a4a2a;color:#5a9a5a;white-space:nowrap;
    }
    #__hs-badge.off { background:#2e1e1e;border-color:#4a2a2a;color:#9a5a5a; }
    .__hs-btn {
      background:#242424;border:1px solid #333;color:#aaa;
      padding:5px 11px;border-radius:3px;cursor:pointer;font-size:0.875rem;
      white-space:nowrap;flex-shrink:0;font-family:inherit;
    }
    .__hs-btn:hover { background:#2e2e2e;color:#eee; }
    .__hs-btn.pri { background:#1e3555;border-color:#2a5599;color:#aac; }
    .__hs-btn.pri:hover { background:#254a80; }
    .__hs-btn:disabled { opacity:0.35;cursor:default; }
    .__hs-btn.cur    { background:#252520;border-color:#4a4a30;color:#ddd; }
    .__hs-btn.active { background:#1a3020;border-color:#2a6a40;color:#6aaa7a; }
    .__hs-link       { display:inline-flex;align-items:center;text-decoration:none; }

    /* tabs */
    #__hs-tabs {
      display:flex;flex-shrink:0;border-bottom:1px solid #2e2e2e;background:#161616;
    }
    .__hs-tabbtn {
      flex:1;background:none;border:none;color:#888;
      padding:10px 0;font-size:0.8125rem;font-weight:700;text-transform:uppercase;
      letter-spacing:0.07em;cursor:pointer;font-family:inherit;
      border-bottom:2px solid transparent;transition:color .12s,border-color .12s;
    }
    .__hs-tabbtn:hover { color:#ccc; }
    .__hs-tabbtn.active { color:#fff;border-bottom-color:#3a7aff; }
    .__hs-tabpanel { display:none; }
    .__hs-tabpanel.active { display:block; }

    /* pane */
    .__hs-pane { border-bottom:1px solid #222; }
    .__hs-pt {
      font-size:0.875rem;font-weight:700;text-transform:uppercase;
      letter-spacing:0.07em;color:#999;padding:8px 28px 5px;
    }

    /* collapsible pane */
    #__hs-ed .__hs-cpane { border-bottom:1px solid rgba(255,255,255,0.18); margin:0 10px 16px; }
    .__hs-cpane-hd {
      display:flex;align-items:center;gap:8px;padding:8px 28px;
      cursor:pointer;user-select:none;
    }
    #__hs-ed .__hs-cpane-hd { margin-top:5px; }
    .__hs-cpane-hd:hover .__hs-cpane-title { color:#fff; }
    .__hs-cpane-tri { font-size:0.75rem;color:#666;flex-shrink:0; }
    .__hs-cpane-title {
      font-size:0.875rem;font-weight:700;text-transform:uppercase;
      letter-spacing:0.07em;color:#ccc;
    }
    .__hs-cpane-body.closed { display:none; }
    .__hs-sub-pt {
      font-size:0.75rem;font-weight:600;text-transform:uppercase;
      letter-spacing:0.06em;color:#999;padding:6px 28px 2px;
    }
    #__hs-ed .__hs-sub-pt { margin-top:5px; }

    /* files */
    .__hs-fr { display:flex;align-items:center;gap:8px;padding:5px 28px 7px; }
    .__hs-fl { font-size:0.875rem;color:#aaa;width:36px;flex-shrink:0; }
    .__hs-fr input {
      flex:1;min-width:0;background:#1e1e1e;border:1px solid #2e2e2e;color:#ccc;
      padding:5px 9px;border-radius:3px;font-size:0.875rem;font-family:inherit;
    }
    .__hs-fr input:focus { border-color:#3a7aff;outline:none; }
    .__hs-fr input[readonly] { color:#555;cursor:default; }
    .__hs-fr input::selection { background:#3a7aff;color:#fff; }

    .__hs-asset-idx {
      font-size:0.8125rem;color:#aaa;flex-shrink:0;white-space:nowrap;width:auto;
    }
    .__hs-asset-name {
      font-size:0.75rem;color:#5a9a5a;flex-shrink:1;min-width:0;max-width:100px;
      overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
    }
    .__hs-asset-name.err { color:#9a5a5a; }
    .__hs-asset-name.pending { color:#888; }
    .__hs-asset-remove:hover { color:#f87; }

    /* info grid */
    .__hs-grid {
      display:grid;grid-template-columns:1fr 1fr;gap:5px;padding:6px 28px 10px;
    }
    .__hs-grid.cols3 { grid-template-columns:1fr 1fr 1fr; }
    .__hs-cell { background:#1e1e1e;border-radius:3px;padding:5px 9px; }
    .__hs-lbl { font-size:0.875rem;color:#999;text-transform:uppercase; }
    .__hs-val { font-size:0.9375rem;color:#aaa;font-variant-numeric:tabular-nums; }
    .__hs-val.warn { color:#f87; }
    .__hs-fp-hd { font-size:0.75rem;color:#888;padding:0 28px 2px;letter-spacing:.04em;text-transform:uppercase; }
    .__hs-fp-hd.none,#__hs-fp-grid.none,#__hs-fp-toggle-row.none { display:none; }
    #__hs-fp-toggle-row { padding:0 28px; }
    .__hs-focal-marker {
      position:absolute;width:8px;height:8px;border-radius:50%;
      background:#fff;border:1.5px solid #e33;
      transform:translate(-50%,-50%);pointer-events:none;z-index:99998;
    }

    /* markers */
    .__hs-empty { padding:9px 28px;color:#666;font-size:0.9375rem;font-style:italic; }

    /* parts list (rendered into the splat-status modal — see openPartsModal) */
    .__hs-pr { display:flex;align-items:center;gap:8px;padding:4px 28px; }
    .__hs-prn {
      font-size:0.9375rem;color:#aaa;flex:1;min-width:0;
      overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
    }
    .__hs-pb {
      font-size:0.9rem;width:22px;height:22px;border-radius:50%;
      display:inline-flex;align-items:center;justify-content:center;
      flex-shrink:0;background:#1e1e1e;border:1px solid #333;color:#555;
    }
    .__hs-pb.ok  { background:#1a2a1a;border-color:#2a4a2a;color:#5a9a5a; }
    .__hs-pb.err { background:#2a1a1a;border-color:#4a2a2a;color:#9a5a5a; }
    .__hs-pft {
      font-size:0.75rem;color:#888;background:#1e1e1e;border:1px solid #333;
      border-radius:3px;padding:1px 6px;flex-shrink:0;text-transform:uppercase;
      letter-spacing:.04em;
    }
    .__hs-plod {
      font-size:0.7rem;color:#7aa6d6;background:#1e2a3e;border:1px solid #2a4a6a;
      border-radius:8px;padding:1px 7px;flex-shrink:0;white-space:nowrap;
    }
    .__hs-plod.err { color:#9a5a5a;background:#2a1a1a;border-color:#4a2a2a; }

    /* splat-status summary row (shown under the Website row and each asset row) */
    .__hs-splat-row {
      display:flex;align-items:center;gap:8px;padding:5px 28px 7px;
      font-size:0.8125rem;color:#999;
    }
    .__hs-splat-row input {
      flex:1;min-width:0;background:#1e1e1e;border:1px solid #2e2e2e;color:#ccc;
      padding:5px 9px;border-radius:3px;font-size:0.875rem;font-family:inherit;
    }
    .__hs-splat-row input:focus { border-color:#3a7aff;outline:none; }
    .__hs-splat-summary { display:flex;align-items:center;gap:8px;flex-shrink:0; }
    .__hs-splat-count { color:#999; }
    .__hs-splat-status { font-weight:600; }
    .__hs-splat-status.ok  { color:#5a9a5a; }
    .__hs-splat-status.err { color:#e08a8a; }
    .__hs-splat-status.pending { color:#777; }

    /* modal */
    #__hs-modal-overlay {
      position:fixed;inset:0;background:rgba(0,0,0,0.55);z-index:200000;
      display:flex;align-items:center;justify-content:center;
    }
    #__hs-modal-box {
      background:#181818;border:1px solid #333;border-radius:6px;
      width:420px;max-height:70vh;display:flex;flex-direction:column;
      box-shadow:0 8px 32px rgba(0,0,0,0.5);
    }
    #__hs-modal-hd {
      display:flex;align-items:center;justify-content:space-between;
      padding:12px 16px;border-bottom:1px solid #2a2a2a;flex-shrink:0;
    }
    #__hs-modal-title { font-size:0.9375rem;font-weight:700;color:#eee; }
    #__hs-modal-close {
      background:none;border:none;color:#888;font-size:1.1rem;
      cursor:pointer;padding:0 4px;font-family:inherit;
    }
    #__hs-modal-close:hover { color:#eee; }
    #__hs-modal-body { overflow-y:auto;padding:6px 0; }

    /* mask feather-edit modal body (see renderMasksEditBody) */
    .__hs-mask-edit-body { padding:0 16px 8px; }
    .__hs-mask-master-row .__hs-attr-lbl { color:#eee;font-weight:600; }
    .__hs-mask-divider { height:1px;background:#2a2a2a;margin:4px 0 6px; }

    #__hs-global-cfg { padding:6px 28px; }
    #__hs-global-cfg .__hs-attr-row { padding:3px 0; }
    #__hs-global-cfg .__hs-attr-lbl { font-size:0.8125rem;color:#888; }

    /* scene cards */
    #__hs-ed .__hs-scard { border-bottom:1px solid rgba(255,255,255,0.12); margin:0 10px; }
    .__hs-scard-hd {
      position:relative;overflow:hidden;isolation:isolate;
      display:flex;align-items:center;gap:8px;
      padding:8px 12px 8px 8px;cursor:pointer;user-select:none;
    }
    .__hs-scard-hd:hover { background:#1d1d1d; }
    .__hs-scard--active > .__hs-scard-hd {
      box-shadow:inset 2px 0 0 rgba(58,122,255,0.65);
    }
    .__hs-scard-bar {
      position:absolute;left:0;top:0;bottom:0;width:0%;
      background:rgba(40,80,200,0.28);pointer-events:none;z-index:-1;
      transition:width 0.1s linear;
    }
    .__hs-scard-play {
      flex-shrink:0;width:20px;height:20px;padding:0;
      display:flex;align-items:center;justify-content:center;
      background:none;border:none;cursor:pointer;
      font-size:0.6rem;color:#4a8aff;
    }
    .__hs-scard-play:hover { color:#A9C6F5; }
    .__hs-scard-play.playing { color:#fff;font-size:0.85rem; }
    .__hs-scard-play.paused  { color:#fff;font-size:0.85rem; }
    .__hs-scard-nm { font-size:0.9375rem;color:#ccc;flex:1;cursor:pointer; }
    .__hs-scard-nm:hover { color:#fff; }
    .__hs-scard--active .__hs-scard-nm { cursor:ew-resize; }
    .__hs-scard-fr { font-size:0.8125rem;color:#888;font-variant-numeric:tabular-nums;flex-shrink:0; }
    .__hs-scard-bd { padding:4px 28px 14px 36px;display:none; }
    .__hs-scard-bd.open { display:block; }

    /* scene name label on the player canvas */
    .__hs-scene-lbl {
      position:absolute;bottom:14px;left:50%;transform:translateX(-50%);
      background:rgba(0,0,0,0.52);color:rgba(255,255,255,0.72);
      font-size:0.8125rem;padding:3px 11px;border-radius:3px;
      pointer-events:none;font-family:system-ui,sans-serif;
      z-index:10;white-space:nowrap;letter-spacing:0.04em;
    }

    .__hs-stri { color:#666;font-size:0.7rem;width:12px;flex-shrink:0;transition:transform .15s; }

    .__hs-sinf-hd { display:flex;align-items:center;gap:6px;padding:5px 0 3px;cursor:pointer;user-select:none; }
    .__hs-sinf-nm { font-size:0.8rem;color:#888;text-transform:uppercase;letter-spacing:.06em;flex:1; }
    .__hs-sinf-bd { padding-left:14px;padding-bottom:6px;display:none; }
    .__hs-sinf-bd.open { display:block; }
    .__hs-srow { display:grid;grid-template-columns:80px 1fr;padding:2px 0; }
    .__hs-srow span:first-child { font-size:0.8125rem;color:#888; }
    .__hs-srow span:last-child  { font-size:0.8125rem;color:#777; }

    #__hs-ed .__hs-sdiv { border-top:1px solid rgba(255,255,255,0.15);margin:10px 10px; }

    .__hs-sel-row { display:flex;align-items:center;gap:10px;padding:5px 0; }
    .__hs-sel-lbl { font-size:0.875rem;color:#aaa;flex-shrink:0;min-width:90px; }
    .__hs-el-sel {
      background:#1e1e1e;border:1px solid #2e2e2e;color:#ccc;
      padding:4px 6px;border-radius:3px;font-size:0.875rem;font-family:inherit;flex:1;
    }
    .__hs-el-sel:focus { border-color:#3a7aff;outline:none; }

    .__hs-drop-btn {
      background:#1e1e1e;border:1px solid #2e2e2e;color:#ccc;
      padding:4px 6px;border-radius:3px;font-size:0.875rem;font-family:inherit;
      cursor:pointer;text-align:left;flex:1;min-width:0;
    }
    .__hs-drop-btn:hover { border-color:#444; }
    .__hs-drop-menu {
      position:fixed;background:#1e1e1e;border:1px solid #3a3a3a;
      border-radius:3px;box-shadow:0 4px 16px rgba(0,0,0,.6);
      z-index:999999;padding:2px 0;
    }
    .__hs-drop-item {
      padding:5px 12px;font-size:0.875rem;color:#ccc;cursor:pointer;
      font-family:system-ui,sans-serif;white-space:nowrap;
    }
    .__hs-drop-item:hover { background:#2a2a2a;color:#fff; }
    .__hs-drop-item.active { color:#3a7aff; }

    .__hs-chkrow { display:flex;align-items:center;gap:8px;padding:5px 0;cursor:pointer; }
    .__hs-chkrow label { font-size:0.9375rem;color:#888;cursor:pointer;flex:1; }

    .__hs-ablock { padding:2px 0; }
    .__hs-ablk-hd { display:flex;align-items:center;gap:8px;padding:5px 0;cursor:pointer;user-select:none; }
    .__hs-ablk-nm { font-size:0.9375rem;color:#888;flex:1; }
    .__hs-ablk-bd { padding-left:20px;display:none;padding-bottom:4px; }
    .__hs-ablk-bd.open { display:block; }

    .__hs-sub-chk { display:flex;align-items:center;gap:7px;padding:3px 0;cursor:pointer; }
    .__hs-sub-chk label { font-size:0.875rem;color:#777;cursor:pointer; }
    .__hs-radio-grp { display:flex;gap:20px;padding:4px 0 2px; }
    .__hs-radio-grp label { display:flex;align-items:center;gap:5px;font-size:0.875rem;color:#777;cursor:pointer; }
    .__hs-deg-row { display:flex;align-items:center;gap:8px;padding:3px 0; }
    .__hs-deg-chk { display:flex;align-items:center;gap:6px;min-width:90px;cursor:pointer; }
    .__hs-deg-chk label { font-size:0.875rem;color:#aaa;cursor:pointer; }
    .__hs-ninp {
      width:60px;background:#1e1e1e;border:1px solid #2e2e2e;color:#ccc;
      padding:3px 7px;border-radius:3px;font-size:0.875rem;font-family:inherit;text-align:right;
      cursor:ew-resize;
      -moz-appearance:textfield;
    }
    .__hs-ninp:focus { border-color:#3a7aff;outline:none;cursor:text; }
    .__hs-ninp.dragging { cursor:grabbing; }
    .__hs-ninp::-webkit-inner-spin-button,
    .__hs-ninp::-webkit-outer-spin-button { -webkit-appearance:none;margin:0; }
    .__hs-deg-unit { font-size:0.8125rem;color:#aaa; }
    .__hs-ninp:disabled,.__hs-deg-chk label.__hs-dim { color:#333; }
    .__hs-ninp:disabled { border-color:#222;color:#333; }

    .__hs-toggle { position:relative;display:inline-block;width:32px;height:18px;flex-shrink:0;cursor:pointer; }
    .__hs-toggle input { opacity:0;width:0;height:0;position:absolute; }
    .__hs-toggle-track { position:absolute;inset:0;background:#2a2a2a;border:1px solid #3a3a3a;border-radius:9px;transition:background .2s,border-color .2s; }
    .__hs-toggle input:checked + .__hs-toggle-track { background:#1e3a6e;border-color:#3a6aff; }
    .__hs-toggle-thumb { position:absolute;top:3px;left:2px;width:12px;height:12px;background:#555;border-radius:50%;transition:transform .2s,background .2s;pointer-events:none; }
    .__hs-toggle input:checked ~ .__hs-toggle-thumb { transform:translateX(14px);background:#3a7aff; }

    .__hs-attr-row { display:flex;align-items:center;padding:5px 0;min-height:28px; }
    .__hs-attr-lbl { font-size:0.875rem;color:#aaa;flex:1; }
    .__hs-num-wrap { display:flex;align-items:center;gap:6px; }
    .__hs-scard-range { font-size:0.8125rem;color:#888;font-variant-numeric:tabular-nums;flex-shrink:0; }
    .__hs-scard-dot {
      width:5px;height:5px;border-radius:50%;flex-shrink:0;
      background:transparent;transition:background .15s;
    }
    .__hs-scard--configured .__hs-scard-dot { background:#4a8aff; }
    .__hs-scard-linked {
      font-size:0.6rem;font-family:monospace;letter-spacing:0;
      background:rgba(58,122,255,0.12);color:#5a9aff;
      border:1px solid rgba(58,122,255,0.25);border-radius:2px;
      padding:1px 3px;flex-shrink:0;line-height:1.4;display:none;
    }
    .__hs-scard-linked.has-id { display:inline-block; }

    /* timeline (floating bottom bar) */
    #__hs-tl {
      position:fixed;bottom:20px;left:20px;right:20px;z-index:99997;
      height:160px;background:#1c1e22;border:1px solid #282828;border-radius:20px;
      box-shadow:0 8px 32px rgba(0,0,0,.5);
      font-family:system-ui,sans-serif;box-sizing:border-box;padding:0 20px;
      pointer-events:auto;user-select:none;overflow:hidden;transition:height .15s;
    }
    #__hs-tl.collapsed { height:56px;display:flex;align-items:center; }
    #__hs-tl.collapsed #__hs-tl-btns,
    #__hs-tl.collapsed #__hs-tl-labels,
    #__hs-tl.collapsed #__hs-tl-footer,
    #__hs-tl.collapsed #__hs-tl-meta { display:none; }
    #__hs-tl.collapsed #__hs-tl-track { flex:1; }
    #__hs-tl-min {
      position:fixed;left:20px;bottom:calc(20px + var(--hs-tl-h, 160px));z-index:99998;
      pointer-events:auto;
      background:#1c1e22;border:1px solid #282828;border-bottom:none;border-radius:8px 8px 0 0;
      color:#666;cursor:pointer;line-height:1;
      font-size:1.25rem;padding:4px 11px;font-family:inherit;
      transition:color .12s,bottom .15s;
    }
    #__hs-tl-min:hover { color:#ccc; }
    #__hs-tl-btns {
      display:flex;justify-content:center;align-items:center;gap:2px;padding:10px 0 6px;
    }
    .__hs-tl-btn {
      background:none;border:none;color:#4a8aff;cursor:pointer;
      font-size:1rem;padding:3px 10px;border-radius:3px;font-family:inherit;line-height:1;
      transition:color .12s;
    }
    .__hs-tl-btn:hover { color:#A9C6F5; }
    .__hs-tl-btn.playing { color:#fff; }
    .__hs-tl-btn.paused  { color:#fff; }
    #__hs-tl-track { position:relative; }
    #__hs-tl-labels { position:relative;height:36px;overflow:visible;margin-bottom:4px; }
    .__hs-tl-seg-lbl {
      position:absolute;top:0;
      font-size:0.6875rem;color:#888;white-space:nowrap;
      padding:2px 5px;border-radius:2px;line-height:1.4;cursor:pointer;
      transform:translateX(-1px);transition:color .1s,background .1s;
    }
    .__hs-tl-seg-lbl:hover { background:rgba(58,122,255,0.18);color:#ddd; }
    .__hs-tl-seg-lbl.configured { color:#4a8aff; }
    .__hs-tl-seg-lbl.configured:hover { background:rgba(58,122,255,0.22);color:#80aaff; }
    .__hs-tl-seg-lbl.active { background:#3a7aff;color:#fff; }
    #__hs-tl-bar {
      position:relative;height:7px;background:#252525;border-radius:2px;cursor:pointer;overflow:hidden;
    }
    #__hs-tl-fill {
      position:absolute;left:0;top:0;bottom:0;width:0;
      background:#3a7aff;pointer-events:none;
    }
    .__hs-tl-mark {
      position:absolute;top:0;bottom:0;width:1px;background:#333;pointer-events:none;z-index:1;
    }
    .__hs-tl-scenedot {
      position:absolute;bottom:0;left:0;width:7px;height:7px;border-radius:50%;
      background:#475569;transform:translateX(-50%);pointer-events:none;z-index:2;
    }
    .__hs-tl-scenedot.configured { background:#4a8aff; }
    #__hs-tl-footer {
      display:flex;justify-content:center;gap:12px;padding:6px 0 6px;
    }
    #__hs-flbl { font-size:0.8125rem;color:#aaa;font-variant-numeric:tabular-nums; }
    #__hs-scene-nm { font-size:0.8125rem;color:#3a7aff;opacity:0.8;letter-spacing:0.03em; }
    #__hs-tl-meta {
      position:absolute;right:20px;top:10px;
      font-size:0.75rem;color:#888;text-align:right;
      display:flex;flex-direction:column;gap:3px;pointer-events:none;
    }
    .__hs-tl-dot { color:#3a7aff;font-size:0.6875rem; }

    /* blend-zone visualizer — overlaid on the live page's linked scene
       container while dragging/editing a blend in/out value; not a toggle,
       purely transient feedback. Lives outside #__hs-ed (appended to
       document.body), positioned via the linked element's own rect. */
    #__hs-blend-overlay { position:fixed;pointer-events:none;z-index:99998;display:none; }
    .__hs-blend-zone { position:absolute;left:0;right:0;background:rgba(80,170,255,0.28); }
    .__hs-blend-zone-top { top:0; }
    .__hs-blend-zone-bot { bottom:0; }
  `;

  // ── HTML template ────────────────────────────────────────────────────────────
  const HTML = `
<div id="__hs-ed">
  <button id="__hs-tab" title="Toggle HoloSplat editor">HS EDITOR</button>
  <div id="__hs-panel">

    <div id="__hs-tb">
      <h1>HoloSplat</h1>
      <span id="__hs-st">connecting…</span>
      <span id="__hs-badge" class="off">API offline</span>
      <button class="__hs-btn" id="__hs-min" title="Minimize panel">−</button>
    </div>

    <div id="__hs-tabs">
      <button class="__hs-tabbtn active" data-tab="scenes">Scenes</button>
      <button class="__hs-tabbtn" data-tab="setup">Setup</button>
      <button class="__hs-tabbtn" data-tab="tools">Tools</button>
    </div>

    <div id="__hs-body">

    <div class="__hs-tabpanel active" data-tab="scenes">
      <div class="__hs-cpane">
        <div class="__hs-cpane-body">
          <div id="__hs-scenes-list"><div class="__hs-empty">No animation loaded</div></div>
        </div>
      </div>
    </div>

    <div class="__hs-tabpanel" data-tab="setup">
      <div class="__hs-cpane">
        <div class="__hs-cpane-hd">
          <span class="__hs-cpane-tri">▼</span>
          <span class="__hs-cpane-title">Render</span>
        </div>
        <div class="__hs-cpane-body">
          <div class="__hs-grid">
            <div class="__hs-cell"><div class="__hs-lbl">Sort</div><div class="__hs-val" id="__hs-sortmode" title="GPU compute-shader radix sort vs CPU fallback (see src/renderer.js _gpuSortFailed)">—</div></div>
            <div class="__hs-cell"><div class="__hs-lbl">Pixel ratio</div><div class="__hs-val" id="__hs-pxratio" title="Adaptive quality's current render scale vs its max (drops under sustained low fps, eases back up — see Viewer#_updateAdaptiveQuality)">—</div></div>
          </div>
          <div id="__hs-global-cfg"></div>
        </div>
      </div>

      <div class="__hs-cpane">
        <div class="__hs-cpane-hd">
          <span class="__hs-cpane-tri">▼</span>
          <span class="__hs-cpane-title">Files</span>
        </div>
        <div class="__hs-cpane-body">
          <div class="__hs-sub-pt">Main Animation</div>
          <div class="__hs-fr">
            <input id="__hs-an" placeholder="/scenes/anim.json" spellcheck="false" title="Main website animation JSON (camera + scene timeline)">
            <button class="__hs-btn" id="__hs-ran" title="Reload animation">↺</button>
          </div>
          <div class="__hs-splat-row">
            <input id="__hs-pd" placeholder="/scenes/headphones/" spellcheck="false" title="Directory for per-object splat files — one .spz per animation object">
            <button class="__hs-btn" id="__hs-sync" title="Load parts from animation and update HTML">↺</button>
            <span class="__hs-splat-summary" id="__hs-website-splat-summary"></span>
          </div>
          <div class="__hs-splat-row">
            <span class="__hs-fl" style="width:auto">Masks</span>
            <span class="__hs-splat-summary" id="__hs-website-masks-summary"></span>
          </div>
          <div class="__hs-sub-pt">Assets</div>
          <div id="__hs-assets-list"></div>
          <div class="__hs-fr">
            <button class="__hs-btn" id="__hs-asset-add" title="Add an asset clip file">+ Add asset</button>
          </div>
        </div>
      </div>

      <div class="__hs-cpane">
        <div class="__hs-cpane-hd">
          <span class="__hs-cpane-tri">▼</span>
          <span class="__hs-cpane-title">3D Scene</span>
        </div>
        <div class="__hs-cpane-body">
          <div class="__hs-grid">
            <div class="__hs-cell"><div class="__hs-lbl">Splats</div><div class="__hs-val" id="__hs-nsplats">—</div></div>
            <div class="__hs-cell"><div class="__hs-lbl">Status</div><div class="__hs-val" id="__hs-loadst">—</div></div>
            <div class="__hs-cell"><div class="__hs-lbl">FlipY</div><div class="__hs-val" id="__hs-flipy">—</div></div>
            <div class="__hs-cell"><div class="__hs-lbl">Camera</div><div class="__hs-val" id="__hs-cammode">—</div></div>
          </div>
          <div class="__hs-sub-pt">Camera</div>
          <div class="__hs-grid cols3">
            <div class="__hs-cell"><div class="__hs-lbl">theta °</div><div class="__hs-val" id="__hs-cth">—</div></div>
            <div class="__hs-cell"><div class="__hs-lbl">phi °</div><div class="__hs-val" id="__hs-cph">—</div></div>
            <div class="__hs-cell"><div class="__hs-lbl">radius</div><div class="__hs-val" id="__hs-cr">—</div></div>
            <div class="__hs-cell"><div class="__hs-lbl">tx</div><div class="__hs-val" id="__hs-ctx">—</div></div>
            <div class="__hs-cell"><div class="__hs-lbl">ty</div><div class="__hs-val" id="__hs-cty">—</div></div>
            <div class="__hs-cell"><div class="__hs-lbl">tz</div><div class="__hs-val" id="__hs-ctz">—</div></div>
          </div>
          <div class="__hs-fp-hd" id="__hs-fp-hd"><span>Focal Point</span></div>
          <div class="__hs-grid cols3" id="__hs-fp-grid">
            <div class="__hs-cell"><div class="__hs-lbl">x</div><div class="__hs-val" id="__hs-fpx">—</div></div>
            <div class="__hs-cell"><div class="__hs-lbl">y</div><div class="__hs-val" id="__hs-fpy">—</div></div>
            <div class="__hs-cell"><div class="__hs-lbl">z</div><div class="__hs-val" id="__hs-fpz">—</div></div>
          </div>
          <div id="__hs-fp-toggle-row"></div>
          <div class="__hs-sub-pt">Masks</div>
          <div id="__hs-masks-list"><div class="__hs-empty">No mask volumes</div></div>
        </div>
      </div>
    </div>

    <div class="__hs-tabpanel" data-tab="tools">
      <div class="__hs-cpane">
        <div class="__hs-cpane-body">
          <div class="__hs-fr" style="padding-top:10px;padding-bottom:6px">
            <button class="__hs-btn" id="__hs-init" title="Inject hs-player into this page">Init page</button>
            <a class="__hs-btn __hs-link" href="/examples/compress.html" target="_blank" title="Compress .splat/.ply files to .spz">Compress</a>
            <a class="__hs-btn __hs-link" href="/examples/prune.html" target="_blank" title="Prune low-impact splats / generate LOD tiers">Prune / LOD</a>
            <a class="__hs-btn __hs-link" href="/examples/pack.html" target="_blank" title="Pack color/material variants of the same model into one .spzv">Pack Variants</a>
          </div>
        </div>
      </div>
    </div>

    </div><!-- /#__hs-body -->

  </div>
</div>`;

  // ── DOM helpers ─────────────────────────────────────────────────────────────
  const el  = id => document.getElementById('__hs-' + id);
  const fmt = n  => typeof n === 'number' ? n.toFixed(2) : '—';

  function setStatus(msg, err = false) {
    el('st').textContent = msg;
    el('st').className   = err ? 'err' : '';
  }

  // ── API ─────────────────────────────────────────────────────────────────────
  async function apiCheck() {
    try {
      const res = await fetch('/hs-api/ls');
      S.apiOnline = res.ok;
      if (res.ok) S.diskFiles = new Set((await res.json()).spz.map(p => '/' + p));
    } catch { S.apiOnline = false; }
    el('badge').textContent = S.apiOnline ? 'API online' : 'API offline';
    el('badge').className   = S.apiOnline ? '' : 'off';
  }

  // Re-fetches the on-disk splat file listing. Call only on an explicit
  // splats-dir change/reload — never per-file, never on every render (that's
  // what caused the editor to flood the server with 404 HEAD probes before).
  async function refreshDiskFiles() {
    if (!S.apiOnline) return;
    try {
      const res = await fetch('/hs-api/ls');
      if (res.ok) S.diskFiles = new Set((await res.json()).spz.map(p => '/' + p));
    } catch { /* keep stale listing rather than wiping it on a transient error */ }
  }

  // ── Connect to player ────────────────────────────────────────────────────────
  function connectEntry(entry) {
    S.entry = entry;
    const { root, api, viewer } = entry;

    // Pre-populate S.sceneConfigs from the player's in-memory configs.
    // window.__hsSceneConfigs is set by player.js from the saved `scenes:` JS variable,
    // so it already reflects the last persisted state without needing an API call.
    {
      const live = window.__hsSceneConfigs || {};
      for (const [name, raw] of Object.entries(live)) {
        if (!S.sceneConfigs[name]) S.sceneConfigs[name] = mergeWithDefault(raw);
      }
    }

    // Pre-populate S.maskConfigs the same way, from window.__hsMaskConfigs
    // (set by player.js from the saved `masks:` JS variable).
    {
      const live = window.__hsMaskConfigs || {};
      for (const [name, raw] of Object.entries(live)) {
        if (!S.maskConfigs[name]) S.maskConfigs[name] = { ...raw };
      }
    }

    S.partsKey = ''; // reset so tick re-renders badges immediately after connect
    el('pd').value = '';
    el('an').value = '';
    el('flipy').textContent = viewer._flipY ? 'yes' : 'no';

    // Inject scene-name label into the player canvas area
    const lbl = document.createElement('div');
    lbl.className = '__hs-scene-lbl';
    lbl.style.display = 'none';
    root.appendChild(lbl);
    S.sceneLbl = lbl;

    // Focal-point debug marker — white circle, red border, toggled via the
    // "Focal pt" checkbox in Setup ▸ 3D Scene. Lets you visually confirm
    // whether orbiting is actually pivoting around the focal point.
    const fpMarker = document.createElement('div');
    fpMarker.className = '__hs-focal-marker';
    fpMarker.style.display = 'none';
    root.appendChild(fpMarker);
    S.focalMarkerEl = fpMarker;

    // Read HTML file + animation JSON → populate fields and render scenes immediately.
    // Await _apiReady so S.apiOnline is known before loadPageState's HTML read.
    (_apiReady || Promise.resolve()).then(() => loadPageState().then(() => {
      renderScenes(); renderMasks();
      renderAssetsList(); loadAllAssets();
    }));

    if (api.animation) {
      connectAnim(api.animation);
    } else {
      // Poll until animation loads
      const poll = setInterval(() => {
        if (api.animation) { clearInterval(poll); connectAnim(api.animation); }
      }, 250);
    }

    setInterval(tick, 250);
    setStatus('Connected');
  }

  function connectAnim(anim) {
    S.markers    = anim.markers    || {};
    S.frameCount = anim.frameCount || 0;
    el('range').max    = Math.max(0, S.frameCount - 1);
    el('range').value  = 0;
    el('flbl').textContent = `0 / ${S.frameCount}`;
    renderTimeline();
    renderParts();
    // Read saved attrs from HTML file before building scene cards so the editor
    // always reflects the persisted state, not just the in-memory DOM.
    (_apiReady || Promise.resolve()).then(() => loadPageState().then(() => { renderScenes(); renderMasks(); }));
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

    // Sort mode — _gpuSortFailed is a one-way latch (see src/renderer.js):
    // once a GPU validation error fires, this viewer instance is stuck on
    // the CPU sorter (slower, softer image) until the page is reloaded.
    const sortEl = el('sortmode');
    if (sortEl) {
      const gpuFailed = !!viewer._renderer?._gpuSortFailed;
      const gpuOn     = !!viewer._gpuSort && !gpuFailed;
      sortEl.textContent = gpuOn ? 'GPU' : (viewer._gpuSort ? 'CPU (fallback)' : 'CPU');
      sortEl.classList.toggle('warn', viewer._gpuSort && gpuFailed);
    }

    // Adaptive-quality render scale — see Viewer#_updateAdaptiveQuality.
    const pxEl = el('pxratio');
    if (pxEl && viewer._maxPixelRatio) {
      const pct = Math.round(100 * viewer._effectivePixelRatio / viewer._maxPixelRatio);
      pxEl.textContent = `${viewer._effectivePixelRatio.toFixed(2)} (${pct}%)`;
      pxEl.classList.toggle('warn', pct < 90);
    }

    // Camera
    if (cam) {
      el('cth').textContent = fmt(cam.theta  * 180 / Math.PI) + '°';
      el('cph').textContent = fmt(cam.phi    * 180 / Math.PI) + '°';
      el('cr').textContent  = fmt(cam.radius);
      el('ctx').textContent = fmt(cam.target[0]);
      el('cty').textContent = fmt(cam.target[1]);
      el('ctz').textContent = fmt(cam.target[2]);
    }

    // Focal point (per-frame when animated)
    const fp = api.animation?.getFocalPoint();
    const hasFp = fp != null;
    el('fp-hd').classList.toggle('none', !hasFp);
    el('fp-grid').classList.toggle('none', !hasFp);
    el('fp-toggle-row').classList.toggle('none', !hasFp);
    if (hasFp) {
      el('fpx').textContent = fmt(fp[0]);
      el('fpy').textContent = fmt(fp[1]);
      el('fpz').textContent = fmt(fp[2]);
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

    // Re-render parts badges whenever the loaded-parts set changes.
    const partsKey = Object.keys(viewer._partIndex || {}).sort().join('\n');
    if (partsKey !== S.partsKey) { S.partsKey = partsKey; renderParts(); }

    // Active scene: update player label + scene list highlight
    const active = getActiveMarker();
    if (active !== S.activeMarker) {
      S.activeMarker = active;
      if (S.sceneLbl) {
        S.sceneLbl.textContent   = active || '';
        S.sceneLbl.style.display = active ? '' : 'none';
      }
      const nm = el('scene-nm');
      if (nm) nm.textContent = active || '';
      const list = el('scenes-list');
      if (list) {
        for (const card of list.querySelectorAll('.__hs-scard')) {
          card.classList.toggle('__hs-scard--active', card.dataset.scKey === active);
        }
      }
    }

    // Sync playback button highlight
    updatePlayState();
  }

  // ── Timeline ─────────────────────────────────────────────────────────────────
  function renderTimeline() {
    const labels = el('tl-labels');
    const track  = el('tl-track');
    if (!labels || !track) return;
    labels.innerHTML = '';
    for (const d of track.querySelectorAll('.__hs-tl-mark, .__hs-tl-scenedot')) d.remove();
    if (!S.frameCount) return;
    const sorted = Object.entries(S.markers).sort((a, b) => a[1] - b[1]);
    if (!sorted.length) return;
    for (let i = 0; i < sorted.length; i++) {
      const [name, startF] = sorted[i];
      const leftPct = startF / S.frameCount * 100;
      const lbl = document.createElement('span');
      lbl.className = '__hs-tl-seg-lbl';
      lbl.dataset.scKey = name;
      lbl.textContent = name;
      lbl.style.left = leftPct + '%';
      if (hasActiveConfig(S.sceneConfigs[name] || {})) lbl.classList.add('configured');
      lbl.addEventListener('click', () => {
        const anim = S.entry?.api.animation;
        if (!anim || !S.entry) return;
        anim.seekFrame(startF);
        S.entry.api.setAnimationPaused(false);
        S.animEverPlayed = true;
        el('range').value = startF;
        el('flbl').textContent = `${startF} / ${S.frameCount}`;
        updatePlayState();
      });
      labels.appendChild(lbl);
      if (i > 0) {
        const div = document.createElement('div');
        div.className = '__hs-tl-mark';
        div.style.left = leftPct + '%';
        track.appendChild(div);
      }
      // Per-scene dot on the slider — the only marker visible when collapsed.
      const dot = document.createElement('div');
      dot.className = '__hs-tl-scenedot';
      dot.style.left = leftPct + '%';
      if (hasActiveConfig(S.sceneConfigs[name] || {})) dot.classList.add('configured');
      track.appendChild(dot);
    }
    requestAnimationFrame(() => adjustLabelOverlaps(labels));
  }

  // After labels are in the DOM, check for horizontal overlaps and stack
  // colliding labels upward so they stay readable.
  function adjustLabelOverlaps(container) {
    if (!container) return;
    const lbls = [...container.querySelectorAll('.__hs-tl-seg-lbl')];
    if (lbls.length < 2) { container.style.height = ''; return; }
    // Container is hidden (e.g. timeline collapsed) — all rects collapse to
    // 0x0, which would make every label "overlap" and stack into N separate
    // rows. Skip until it's actually laid out (re-run on expand instead).
    if (container.getBoundingClientRect().width === 0) return;
    lbls.sort((a, b) => a.getBoundingClientRect().left - b.getBoundingClientRect().left);
    const rowRights = [-Infinity]; // rightmost pixel reached in each row (row 0 = baseline)
    for (const lbl of lbls) {
      const rect = lbl.getBoundingClientRect();
      let row = 0;
      while (row < rowRights.length && rowRights[row] > rect.left - 2) row++;
      if (row >= rowRights.length) rowRights.push(-Infinity);
      rowRights[row] = rect.right;
      lbl.style.top = row > 0 ? `${row * 20}px` : '0';
    }
    // Grow the labels area to fit stacked rows so they don't overlap the bar below.
    container.style.height = rowRights.length > 1 ? `${rowRights.length * 20 + 16}px` : '';
  }

  let _fpsCount = 0, _fpsLast = 0;

  // Runs at rAF rate to keep scene card progress bars, play buttons, and timeline smooth
  function rafBars() {
    requestAnimationFrame(rafBars);
    if (!S.entry) return;
    const frame  = S.entry.api.animation?.frame ?? 0;
    const paused = S.entry.viewer._animPaused;
    const active = S.activeMarker;

    // Focal-point debug marker — reuses the same screen-space projection as
    // Blender-exported callouts (Viewer#projectCallouts) so it tracks the
    // current view exactly, including behind-camera hiding.
    if (S.showFocalMarker && S.focalMarkerEl) {
      const fp = S.entry.api.animation?.getFocalPoint();
      const proj = fp ? S.entry.viewer.projectCallouts([{ id: '__focal', pos: fp }])[0] : null;
      if (proj?.visible) {
        S.focalMarkerEl.style.display = '';
        S.focalMarkerEl.style.left = proj.x + 'px';
        S.focalMarkerEl.style.top  = proj.y + 'px';
      } else {
        S.focalMarkerEl.style.display = 'none';
      }
    }

    // Scene card bars
    for (const [key, cd] of Object.entries(S.sceneCards)) {
      const isActive = key === active;
      const pct = isActive
        ? Math.max(0, Math.min(100, (frame - cd.from) / cd.frames * 100))
        : 0;
      if (cd._pct !== pct) { cd._pct = pct; cd.barEl.style.width = pct + '%'; }
      const isPlay   = isActive && !paused;
      const isPaused = isActive && paused && S.animEverPlayed;
      const showStop = isPlay || isPaused;
      if (cd._playing !== showStop) {
        cd._playing = showStop;
        cd.playBtn.textContent = showStop ? '⏸' : '▶';
        cd.playBtn.classList.toggle('playing', isPlay);
        cd.playBtn.classList.toggle('paused',  isPaused);
      }
    }

    // Timeline progress fill
    const tlFill = el('tl-fill');
    if (tlFill && S.frameCount > 1) {
      const pct = Math.max(0, Math.min(100, frame / (S.frameCount - 1) * 100));
      if (tlFill.__pct !== pct) { tlFill.__pct = pct; tlFill.style.width = pct + '%'; }
    }

    // Timeline active scene label
    const tlLabels = el('tl-labels');
    if (tlLabels && tlLabels.__active !== active) {
      tlLabels.__active = active;
      for (const lbl of tlLabels.querySelectorAll('.__hs-tl-seg-lbl'))
        lbl.classList.toggle('active', lbl.dataset.scKey === active);
    }

    // FPS counter (updates once per second)
    _fpsCount++;
    const now = performance.now();
    if (!_fpsLast) _fpsLast = now;
    if (now - _fpsLast >= 1000) {
      const fps = Math.round(_fpsCount * 1000 / (now - _fpsLast));
      const fpsEl = el('tl-fps');
      if (fpsEl) fpsEl.textContent = fps;
      _fpsCount = 0;
      _fpsLast  = now;
    }
  }

  function updatePlayState() {
    if (!S.entry) return;
    const paused  = S.entry.viewer._animPaused;
    const reverse = S.entry.api.animation?.direction === -1;

    const fwdBtn = el('playpause');
    const bwdBtn = el('rev');
    if (fwdBtn) {
      const fwdPlaying = !paused && !reverse;
      const fwdPaused  = paused && !reverse && S.animEverPlayed;
      fwdBtn.textContent = (fwdPlaying || fwdPaused) ? '⏸' : '▶';
      fwdBtn.title       = fwdPlaying ? 'Pause' : 'Play';
      fwdBtn.classList.toggle('playing', fwdPlaying);
      fwdBtn.classList.toggle('paused',  fwdPaused);
    }
    if (bwdBtn) {
      const bwdPlaying = !paused && reverse;
      const bwdPaused  = paused && reverse && S.animEverPlayed;
      bwdBtn.textContent = (bwdPlaying || bwdPaused) ? '⏸' : '◀';
      bwdBtn.title       = bwdPlaying ? 'Pause' : 'Play backward';
      bwdBtn.classList.toggle('playing', bwdPlaying);
      bwdBtn.classList.toggle('paused',  bwdPaused);
    }
  }

  function getActiveMarker() {
    const anim = S.entry?.api.animation;
    if (!anim) return null;
    const frame = anim.frame;
    let active = null, maxF = -1;
    for (const [name, f] of Object.entries(S.markers)) {
      if (f <= frame && f > maxF) { maxF = f; active = name; }
    }
    return active;
  }

  // ── Scrubber ─────────────────────────────────────────────────────────────────
  function seekFrame(n) {
    const anim = S.entry?.api.animation;
    if (anim) { anim.seekFrame(n); S.entry.api.setAnimationPaused(true); }
    el('range').value     = n;
    el('flbl').textContent = `${Math.round(n)} / ${S.frameCount}`;
    scrollToMatchFrame(n);
  }

  // Scroll the page so the scroll-driven position matches the seeked frame.
  // Inverts the scrollFrameFor formula from player.js:
  //   t = (vh - rect.top) / (elH + vh)  ↔  scrollY = absTop - vh + t*(elH+vh)
  function scrollToMatchFrame(n) {
    if (!S.markers || !S.sceneConfigs) return;
    const markerEntries = Object.entries(S.markers).sort((a, b) => a[1] - b[1]);
    for (let i = 0; i < markerEntries.length; i++) {
      const [name, fromFrame] = markerEntries[i];
      const toFrame = (markerEntries[i + 1]?.[1] ?? S.frameCount) - 1;
      if (n < fromFrame - 0.5 || n > toFrame + 0.5) continue;
      const cfg = S.sceneConfigs[name] || {};
      // Auto/pingpong scenes (e.g. an autoplay intro) are independent of scroll
      // position by design — don't drag the page around while scrubbing them.
      if (!cfg.linkedId || cfg.playback === 'auto') continue;
      const target = document.getElementById(cfg.linkedId);
      if (!target) continue;
      const t = (toFrame > fromFrame)
        ? Math.max(0, Math.min(1, (n - fromFrame) / (toFrame - fromFrame)))
        : 0;
      const vh        = window.innerHeight;
      const absTop    = target.getBoundingClientRect().top + window.scrollY;
      const targetY   = absTop - vh + t * (target.offsetHeight + vh);
      window.scrollTo(0, Math.max(0, targetY));
      return;
    }
  }

  // ── Parts sync ───────────────────────────────────────────────────────────────
  // Mirrors animation.js splatNameFromId.
  // Strips "hs-part." / "ctrl." prefixes and trailing Blender duplicate suffixes.
  // Preserves hierarchical names: "headphones.cup.l" → "headphones.cup.l".
  function splatNameFromId(id) {
    let s = id.replace(/^hs-part\./, '').replace(/^ctrl\./, '');
    return s.replace(/(\.\d+)+$/, '');
  }

  // Resolves the on-disk splat file(s) for a part. If a packed "<base>.spzv"
  // exists (built with the Pack Variants tool — see examples/pack.html), it
  // is preferred: geometry is loaded once and every declared variant becomes
  // a runtime-swappable palette (Viewer#setVariant), so only the active
  // variant costs anything to render. Otherwise, parts with declared
  // color/material variants are returned as { url, variants }: only the
  // active variant's file is loaded now (each variant was independently
  // trained, so they may have different geometry/splat counts — loading all
  // of them up front would be wasteful); Viewer#setVariant fetches the
  // others lazily. Parts without variants try "<dir><splat-name>", then fall
  // back to whatever path(s) are already configured (so a renamed/suffixed
  // file — e.g. "headphones.fork.left.blue" — still resolves even when the
  // bare splat-name doesn't exist on disk).
  async function resolvePartPaths(dir, obj, lastParts) {
    const splatName = splatNameFromId(obj.id);
    const base = `${dir}${splatName}`;

    if (obj.variants?.length) {
      if (S.diskFiles?.has(`${base}.spzv`)) return [`${base}.spzv`];

      // Keep whichever variant is currently active (if it still exists),
      // otherwise fall back to the first declared variant.
      const existing = lastParts[obj.id];
      const prevUrl = existing && typeof existing === 'object' && !Array.isArray(existing) ? existing.url : null;
      let activeName = obj.variants[0];
      if (prevUrl) {
        const m = prevUrl.match(/\.([^./]+)\.(?:spz|ply|splat)$/i);
        if (m && obj.variants.includes(m[1])) activeName = m[1];
      }
      const order = [activeName, ...obj.variants.filter(v => v !== activeName)];
      for (const v of order) {
        const found = await findExistingSplat(`${base}.${v}`);
        if (found) return { url: found, variants: obj.variants };
      }
    }

    const found = await findExistingSplat(base);
    if (found) return [splitSplatExt(found).base];

    const existing = lastParts[obj.id] ?? lastParts[splatName] ?? null;
    const existingPaths = Array.isArray(existing) ? existing : existing ? [existing] : [];
    const resolved = [];
    for (const p of existingPaths) {
      const f = await findExistingSplat(p);
      if (f) resolved.push(splitSplatExt(f).base);
    }
    return resolved.length ? resolved : [base];
  }

  async function syncParts() {
    const anim = S.entry?.api.animation;
    if (!anim?.objects?.length) {
      setStatus('No objects in animation — load an animation first', true);
      return;
    }
    const dir = el('pd').value.trim().replace(/\/?$/, '/');
    if (!dir || dir === '/') {
      setStatus('Set a parts directory first', true);
      return;
    }
    const lastParts = S.entry?.viewer._lastParts || {};
    await refreshDiskFiles(); // explicit reload — refresh the disk listing before resolving against it
    setStatus(`Resolving ${anim.objects.length} part path(s)…`);
    const partsMap = {};
    let fileCount = 0;
    for (const obj of anim.objects) {
      const paths = await resolvePartPaths(dir, obj, lastParts);
      if (Array.isArray(paths)) {
        partsMap[obj.id] = paths.length > 1 ? paths : paths[0];
        fileCount += paths.length;
      } else {
        partsMap[obj.id] = paths; // { url, variants }
        fileCount += 1;
      }
    }
    const n = Object.keys(partsMap).length;
    setStatus(`Loading ${fileCount} splat file(s) for ${n} part(s)…`);
    try {
      await S.entry.api.loadParts(partsMap);
    } catch (e) {
      setStatus(`Sync failed: ${e.message}`, true);
      return;
    }
    renderParts();
    if (!S.apiOnline) {
      setStatus(`Loaded ${fileCount} file(s) for ${n} part(s) (API offline — not saved)`);
      return;
    }
    try {
      // This writes a static, fully-resolved `parts:` map (with variant
      // info) — the right choice for parts with declared color/material
      // variants, since player.js's dynamic partsDir-based derivation
      // (saveDirUrl, below) doesn't know about variants. partsDir takes
      // priority over this map at runtime if both are set, so don't save a
      // partsDir for pages that need this sync instead — see saveDirUrl.
      await saveParts(partsMap);
      setStatus(`Synced ${n} parts — saved to file`);
    } catch (e) {
      setStatus(`Parts loaded but save failed: ${e.message}`, true);
    }
  }

  async function saveParts(partsMap) {
    const res = await fetch('/hs-api/js-parts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ page: window.location.pathname, parts: partsMap }),
    });
    if (!res.ok) throw new Error(`/hs-api/js-parts → ${res.status}`);
  }

  // Splat files are detected by extension fallback (.spz → .ply → .splat),
  // mirroring loadUrl() in src/viewer.js.
  const SPLAT_EXTS = ['spz', 'ply', 'splat'];

  function splitSplatExt(path) {
    const m = path.match(/\.(spz|spzv|ply|splat)$/i);
    if (m) {
      const ext = m[1].toLowerCase();
      return { base: path.slice(0, -m[0].length), exts: [ext, ...SPLAT_EXTS.filter(e => e !== ext)] };
    }
    return { base: path, exts: SPLAT_EXTS };
  }

  // Looks up a candidate path against the cached on-disk file listing
  // (S.diskFiles, from /hs-api/ls) — never probes the network per-file.
  // If the listing isn't available (API offline, or not fetched yet),
  // assumes nothing exists rather than guessing via HTTP requests.
  function findExistingSplat(path) {
    if (!S.diskFiles) return null;
    const { base, exts } = splitSplatExt(path);
    for (const ext of exts) {
      if (S.diskFiles.has(`${base}.${ext}`)) return `${base}.${ext}`;
    }
    return null;
  }

  // LOD tiers live in a "<name>.lods/" folder next to the source file, as
  // "<name>.lodN.spz" — see src/device-tier.js resolveLodUrl (same lookup,
  // shared with the runtime loader) and examples/prune.html (which
  // generates them). Always .spz regardless of the source file's format.
  function lodCandidatePath(base, i) {
    const slash = base.lastIndexOf('/');
    const dir   = base.slice(0, slash + 1);
    const name  = base.slice(slash + 1);
    return `${dir}${name}.lods/${name}.lod${i}.spz`;
  }

  // ── Modal (generic) ──────────────────────────────────────────────────────────

  function closeModal() {
    document.getElementById('__hs-modal-overlay')?.remove();
  }

  /** Opens a simple modal dialog. `buildBody(bodyEl)` populates the content. */
  function openModal(title, buildBody) {
    closeModal();
    const overlay = document.body.appendChild(document.createElement('div'));
    overlay.id = '__hs-modal-overlay';
    overlay.addEventListener('click', e => { if (e.target === overlay) closeModal(); });

    const box = overlay.appendChild(document.createElement('div'));
    box.id = '__hs-modal-box';

    const hd = box.appendChild(document.createElement('div'));
    hd.id = '__hs-modal-hd';
    const titleEl = hd.appendChild(document.createElement('span'));
    titleEl.id = '__hs-modal-title';
    titleEl.textContent = title;
    const closeBtn = hd.appendChild(document.createElement('button'));
    closeBtn.id = '__hs-modal-close';
    closeBtn.textContent = '✕';
    closeBtn.addEventListener('click', closeModal);

    const bodyEl = box.appendChild(document.createElement('div'));
    bodyEl.id = '__hs-modal-body';
    buildBody(bodyEl);
  }

  // ── Splat-status rows (shared between the Website row and each asset row) ───
  //
  // A "row" is { label, loaded, path }: `loaded` means the part is already
  // resolved in the live viewer (this._partIndex); `path` is the splat file
  // path to additionally probe on disk via findExistingSplat — null for
  // asset-clip rows, which don't have a separate file of their own to check
  // (a clip just animates a part the main scene already loaded), only
  // whether that part is currently loaded at all.

  /** Renders the detailed per-row list (type/LOD badges where a path exists) into `bodyEl`, live-updating as rows resolve. */
  function renderPartRowsDetail(bodyEl, rows) {
    if (!rows.length) { bodyEl.innerHTML = '<div class="__hs-empty">No parts</div>'; return; }
    bodyEl.innerHTML = rows.map(r => {
      const typeBadge = r.path ? `<span class="__hs-pft" data-role="type"></span>` : '';
      const lodBadge  = r.path ? `<span class="__hs-plod" data-role="lod" style="display:none"></span>` : '';
      return `<div class="__hs-pr">
        <span class="__hs-prn" title="${r.label}">${r.label}</span>
        ${typeBadge}
        ${lodBadge}
        <span class="__hs-pb ${r.loaded ? 'ok' : ''}" data-role="status"
        >${r.loaded ? '✓' : '…'}</span>
      </div>`;
    }).join('');

    const rowEls = bodyEl.querySelectorAll('.__hs-pr');
    rows.forEach((r, i) => {
      const rowEl = rowEls[i];
      const badge = rowEl.querySelector('[data-role="status"]');
      if (!r.path) {
        badge.className = `__hs-pb ${r.loaded ? 'ok' : 'err'}`;
        badge.textContent = r.loaded ? '✓' : '✗';
        badge.title = r.loaded ? 'Loaded in viewer' : 'Not loaded in the current scene';
        return;
      }
      const typeBadge = rowEl.querySelector('[data-role="type"]');
      {
        const found = findExistingSplat(r.path);
        typeBadge.textContent = found ? found.slice(found.lastIndexOf('.') + 1).toLowerCase() : '?';
        typeBadge.title = found ? `Found: ${found}` : `Missing — tried ${r.path}.{spz,ply,splat}`;
        if (!r.loaded) {
          badge.className = `__hs-pb ${found ? 'ok' : 'err'}`;
          badge.textContent = found ? '✓' : '✗';
          badge.title = found ? `Found: ${found}` : `Missing — tried ${r.path}.{spz,ply,splat}`;
        } else {
          badge.title = 'Loaded in viewer';
        }
      }
      const lodBadge = rowEl.querySelector('[data-role="lod"]');
      if (lodBadge) {
        const { base: lodBase } = splitSplatExt(r.path);
        const found = [0, 1, 2, 3].map(i => {
          const candidate = lodCandidatePath(lodBase, i);
          return S.diskFiles?.has(candidate) ? candidate : null;
        });
        const files = found.filter(Boolean);
        lodBadge.classList.toggle('err', files.length === 0);
        lodBadge.textContent = files.length ? `${files.length} LOD${files.length > 1 ? 's' : ''}` : '0 LODs';
        lodBadge.title = files.length
          ? found.map((f, i) => f ? `lod${i}: ${f}` : null).filter(Boolean).join('\n')
          : `No ${lodBase.slice(lodBase.lastIndexOf('/') + 1)}.lods/ folder found next to ${lodBase}`;
        lodBadge.style.display = '';
      }
    });
  }

  function openPartsModal(title, rows) {
    openModal(title, bodyEl => renderPartRowsDetail(bodyEl, rows));
  }

  /** Renders the compact "<n> splats  ✓ / <m> not loaded  [show]" line into `container`.
   *  Status reflects only what's already resolved in the live viewer (r.loaded) — no
   *  disk probing here, since that ran on every render and flooded the server with
   *  404s for variants that are expected not to be loaded yet. Click "show" to
   *  actually check disk via the detail modal (renderPartRowsDetail). */
  function renderSplatSummary(container, rows, modalTitle) {
    if (!container) return;
    container.innerHTML = '';
    container.style.display = '';

    const countEl = container.appendChild(document.createElement('span'));
    countEl.className = '__hs-splat-count';
    countEl.textContent = `${rows.length} splat${rows.length === 1 ? '' : 's'}`;

    const showBtn = container.appendChild(document.createElement('button'));
    showBtn.className = '__hs-btn';
    showBtn.textContent = 'show';
    showBtn.addEventListener('click', () => openPartsModal(modalTitle, rows));

    if (!rows.length) return;

    const missing = rows.filter(r => !r.loaded).length;
    const statusEl = container.appendChild(document.createElement('span'));
    if (missing === 0) {
      statusEl.className = '__hs-splat-status ok';
      statusEl.textContent = '✓';
    } else {
      statusEl.className = '__hs-splat-status';
      statusEl.textContent = `${missing} not loaded`;
    }
  }

  // ── Mask-status rows (shown under the Main Animation row and each asset row) ─
  //
  // `maskList` is [{ name, softEdge }] — softEdge is the Blender-exported
  // default feather, overridden per-name by S.maskConfigs (see applyMaskFeather).

  /** Renders the compact "<n> masks  [Edit]" line into `container`. */
  function renderMaskSummary(container, maskList, modalTitle) {
    if (!container) return;
    container.innerHTML = '';

    const countEl = container.appendChild(document.createElement('span'));
    countEl.className = '__hs-splat-count';
    countEl.textContent = `${maskList.length} mask${maskList.length === 1 ? '' : 's'}`;

    const editBtn = container.appendChild(document.createElement('button'));
    editBtn.className = '__hs-btn';
    editBtn.textContent = 'Edit';
    editBtn.addEventListener('click', () => openModal(modalTitle, bodyEl => renderMasksEditBody(bodyEl, maskList)));
  }

  /** Renders per-mask feather controls plus a master control that applies to all of them at once. */
  function renderMasksEditBody(bodyEl, maskList) {
    bodyEl.innerHTML = '';
    if (!maskList.length) {
      bodyEl.innerHTML = '<div class="__hs-empty">No mask volumes</div>';
      return;
    }

    const wrap = bodyEl.appendChild(document.createElement('div'));
    wrap.className = '__hs-mask-edit-body';

    const rowCtrls = [];
    const masterRow = wrap.appendChild(mkRow('All masks', mkNum(maskList[0].softEdge ?? 0.05, 0, 10, 0.01, null, v => {
      for (const m of maskList) applyMaskFeather(m.name, v);
      for (const c of rowCtrls) c.inp.value = v;
    }).el));
    masterRow.classList.add('__hs-mask-master-row');

    wrap.appendChild(document.createElement('div')).className = '__hs-mask-divider';

    for (const m of maskList) {
      const defaultFeather = m.softEdge ?? 0.05;
      const current = S.maskConfigs[m.name]?.feather ?? defaultFeather;
      const ctrl = mkNum(current, 0, 10, 0.01, null, v => applyMaskFeather(m.name, v));
      rowCtrls.push(ctrl);
      wrap.appendChild(mkRow(m.name, ctrl.el));
    }
  }

  function applyMaskFeather(name, value) {
    S.maskConfigs[name] = { ...(S.maskConfigs[name] || {}), feather: value };
    S.entry?.api.setMaskFeather(name, value);
    saveMaskConfig(name, S.maskConfigs[name]);
  }

  // Strips a trailing ".<variant>" suffix from path if it matches one of the
  // part's known variants, so per-variant paths can be rebuilt from it.
  function stripVariantSuffix(path, variants) {
    for (const v of variants) {
      if (path.endsWith(`.${v}`)) return path.slice(0, -(v.length + 1));
    }
    return path;
  }

  // Builds one row per splat file referenced by the animation JSON — the
  // animated parts themselves, plus every declared color/material variant,
  // each as its own independent entry. See "Splat-status rows" above for
  // what `loaded`/`path` mean and how rows get checked.
  function buildAnimPartRows() {
    const anim = S.entry?.api.animation;
    if (!anim?.objects?.length) return [];

    const partIndex = S.entry?.viewer._partIndex || {};
    const lastParts = S.entry?.viewer._lastParts || {};

    const rows = [];
    for (const obj of anim.objects) {
      const name     = splatNameFromId(obj.id);
      const rawPath  = lastParts[obj.id] ?? lastParts[name] ?? null;
      const loaded   = partIndex[obj.id] !== undefined || partIndex[name] !== undefined;
      const variants = obj.variants || [];

      if (rawPath && typeof rawPath === 'object' && !Array.isArray(rawPath) && rawPath.url) {
        // Lazy variant part (resolvePartPaths returned { url, variants }):
        // only the active variant is loaded — the others are fetched on
        // demand by Viewer#setVariant, since each has its own geometry.
        const m = rawPath.url.match(/\.([^./]+)\.(?:spz|ply|splat)$/i);
        const active = m ? m[1] : '?';
        rows.push({ label: `${name} (${variants.length} variants, "${active}" active)`, loaded, path: rawPath.url });
        continue;
      }

      const paths = Array.isArray(rawPath) ? rawPath : rawPath ? [rawPath] : [];
      if (variants.length && paths[0]?.endsWith('.spzv')) {
        // Packed variants (examples/pack.html): one file holds the shared
        // geometry plus every variant's color/alpha palette — show it as a
        // single row, swappable at runtime via Viewer#setVariant.
        rows.push({ label: `${name} (${variants.length} variants packed)`, loaded, path: paths[0] });
      } else if (variants.length) {
        // Every variant is its own real file — no separate bare-name row.
        // All variants present in `paths` are loaded simultaneously
        // (sharing this part's transform) until per-color masks select
        // between them.
        const base = paths.length ? stripVariantSuffix(paths[0], variants) : null;
        for (const v of variants) {
          const vPath = base ? `${base}.${v}` : null;
          rows.push({ label: `${name}.${v}`, loaded: loaded && paths.includes(vPath), path: vPath });
        }
      } else {
        rows.push({ label: name, loaded, path: paths[0] ?? null });
      }
    }
    return rows;
  }

  // Builds one row per splat file this asset declares (see
  // export_holosplat_asset.py's "parts" field) — every variant of every
  // part, not just the one matching the asset's currently-selected default,
  // so the modal reflects every file on disk this asset could load, the
  // same way the Website row lists every variant of every animated object.
  // The JSON itself has no idea where the actual splat files live — that's
  // the designer's job, via the asset's own "splats path" field — so each
  // row's `path` is resolved against that, exactly like the Website row's,
  // and checked the same way (findExistingSplat, on demand in the detail
  // modal — see openPartsModal). `loaded` always reflects the live viewer
  // (regardless of whether a splats path is set), by matching this row's own
  // filename against the slots actually on the GPU right now.
  function buildAssetPartRows(asset) {
    const dir       = asset.splatsDir?.trim().replace(/\/?$/, '/');
    const partIndex = S.entry?.viewer._partIndex || {};
    const fileNames = S.entry?.viewer._fileNames || [];
    const rows = [];
    for (const [id, part] of Object.entries(asset.parts ?? {})) {
      const variants = part.variants ?? [];
      const slots = partIndex[id] ?? partIndex[part.splatName];
      if (!variants.length) {
        rows.push({ label: part.splatName, loaded: slots !== undefined, path: dir ? dir + part.splatName : null });
        continue;
      }
      // Default variant loads first; the rest fetch lazily in the background
      // (see Viewer#loadClips) — check each row's own filename against the
      // slots actually on the GPU right now, rather than assuming only the
      // default is ever loaded, so this stays accurate once the background
      // fetch finishes too.
      for (const v of variants) {
        const label  = `${part.splatName}.${v}`;
        const loaded = slots?.some(slot => fileNames[slot] === label) ?? false;
        rows.push({ label, loaded, path: dir ? dir + label : null });
      }
    }
    return rows;
  }

  // Re-checks the Website row's splat status (called on every load, not just
  // after "sync", so newly-referenced or renamed splat files that are
  // missing show up here).
  function renderParts() {
    renderSplatSummary(el('website-splat-summary'), buildAnimPartRows(), 'Website splats');
  }

  // ── Reload helpers ───────────────────────────────────────────────────────────
  async function reloadAnim() {
    const url = el('an').value.trim();
    if (!url || !S.entry) return;
    setStatus('Loading animation…');
    try {
      const anim = await S.entry.api.loadAnim(url);
      if (anim) connectAnim(anim);
      if (S.apiOnline) await saveAnimUrl(url);
      setStatus('Animation loaded');
    } catch (e) { setStatus(e.message, true); }
  }

  async function saveAnimUrl(url) {
    try {
      await fetch('/hs-api/js-anim', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ page: window.location.pathname, url }),
      });
    } catch { }
  }

  // Persists the splats directory as an explicit `partsDir: '...'` line in
  // the HTML — player.js then derives every part's file path at runtime as
  // `${partsDir}${splatNameFromId(obj.id)}${ext}` directly from the
  // animation JSON's objects list (src/player.js:602-611), no `parts:` map
  // needed. Only covers parts with no declared variants — partsDir takes
  // priority over an explicit `parts:` map at runtime, so leave this field
  // empty for scenes using "sync" (parts with color/material variants).
  async function saveDirUrl(dir) {
    try {
      await fetch('/hs-api/js-partsDir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ page: window.location.pathname, partsDir: dir }),
      });
    } catch { }
  }

  // ── Scenes ───────────────────────────────────────────────────────────────────

  function computeScenes() {
    const sorted = Object.entries(S.markers).sort((a, b) => a[1] - b[1]);
    return sorted.map(([name, from], i) => {
      const to = (sorted[i + 1]?.[1] ?? S.frameCount) - 1;
      return { name, from, to, frames: to - from + 1 };
    });
  }

  // All page elements with IDs that aren't part of the editor UI.
  function scanPageEls() {
    return Array.from(document.querySelectorAll('[id]'))
      .filter(e => !e.id.startsWith('__hs-') && e.id)
      .map(e => ({ id: e.id, tag: e.tagName.toLowerCase() }));
  }

  // ── Blend-zone visualizer ─────────────────────────────────────────────────────
  // Transient overlay on the live page's linked scene container, showing the
  // blend in/out percentages as light-blue bands at its top/bottom. Not a
  // toggle — shown only while a blend value is actively being dragged/typed
  // (see mkNum's onActive), hidden the instant that stops.
  function showBlendOverlay(linkedEl, blendInPct, blendOutPct) {
    if (!linkedEl) return;
    let ov = document.getElementById('__hs-blend-overlay');
    if (!ov) {
      ov = document.createElement('div');
      ov.id = '__hs-blend-overlay';
      ov.appendChild(document.createElement('div')).className = '__hs-blend-zone __hs-blend-zone-top';
      ov.appendChild(document.createElement('div')).className = '__hs-blend-zone __hs-blend-zone-bot';
      document.body.appendChild(ov);
    }
    const rect = linkedEl.getBoundingClientRect();
    ov.style.left   = rect.left   + 'px';
    ov.style.top    = rect.top    + 'px';
    ov.style.width  = rect.width  + 'px';
    ov.style.height = rect.height + 'px';
    ov.style.display = 'block';
    ov.children[0].style.height = Math.max(0, Math.min(100, blendInPct))  + '%';
    ov.children[1].style.height = Math.max(0, Math.min(100, blendOutPct)) + '%';
  }
  function hideBlendOverlay() {
    const ov = document.getElementById('__hs-blend-overlay');
    if (ov) ov.style.display = 'none';
  }

  // ── Shared attribute-row controls (used by scene + mask cards) ───────────────

  function mkToggle(checked, onChange) {
    const lbl = document.createElement('label');
    lbl.className = '__hs-toggle';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = checked;
    const track = document.createElement('span'); track.className = '__hs-toggle-track';
    const thumb = document.createElement('span'); thumb.className = '__hs-toggle-thumb';
    lbl.appendChild(cb); lbl.appendChild(track); lbl.appendChild(thumb);
    cb.addEventListener('change', () => onChange(cb.checked));
    return { el: lbl, cb };
  }

  function mkRow(label, ctrl) {
    const r = document.createElement('div'); r.className = '__hs-attr-row';
    const l = document.createElement('span'); l.className = '__hs-attr-lbl'; l.textContent = label;
    r.appendChild(l); r.appendChild(ctrl);
    return r;
  }

  // onActive(bool), if given, fires while the value is being actively
  // adjusted — dragging or focused for typing — and is meant for transient
  // UI feedback (e.g. the blend-zone visualizer), not persisted state.
  function mkNum(val, min, max, step, unit, onChange, onActive) {
    const wrap = document.createElement('div'); wrap.className = '__hs-num-wrap';
    const inp = document.createElement('input');
    inp.type = 'number'; inp.className = '__hs-ninp';
    inp.min = min; inp.max = max; inp.step = step; inp.value = val;
    inp.addEventListener('change', () => {
      const v = parseFloat(inp.value);
      const clamped = isNaN(v) ? min : Math.max(min, Math.min(max, v));
      inp.value = clamped;
      onChange(clamped);
    });
    if (onActive) {
      inp.addEventListener('focus', () => onActive(true));
      inp.addEventListener('blur',  () => onActive(false));
    }

    // Drag-slider: horizontal drag changes value; short click = focus for typing
    let drag = null;
    inp.addEventListener('pointerdown', e => {
      if (e.button !== 0 || inp === document.activeElement) return;
      drag = { x: e.clientX, val: parseFloat(inp.value) || 0, moved: false };
      inp.setPointerCapture(e.pointerId);
      inp.classList.add('dragging');
      onActive?.(true);
      e.preventDefault();
    });
    inp.addEventListener('pointermove', e => {
      if (!drag) return;
      const dx = e.clientX - drag.x;
      if (Math.abs(dx) > 3) drag.moved = true;
      if (drag.moved) {
        const newVal = Math.max(min, Math.min(max, +(drag.val + dx * step).toFixed(10)));
        inp.value = newVal;
        onChange(newVal);
      }
    });
    inp.addEventListener('pointerup', e => {
      if (!drag) return;
      const moved = drag.moved;
      drag = null;
      inp.classList.remove('dragging');
      inp.releasePointerCapture(e.pointerId);
      if (!moved) { inp.focus(); inp.select(); }       // 'focus' listener takes over onActive
      else onActive?.(false);                          // drag gesture ended
    });

    wrap.appendChild(inp);
    if (unit) {
      const u = document.createElement('span'); u.className = '__hs-deg-unit'; u.textContent = unit;
      wrap.appendChild(u);
    }
    return { el: wrap, inp };
  }

  // NOTE: orbit/follow-mouse config (damping, resetSpeed, resetEase, h/v
  // limits) was removed here as part of a deliberate cleanup — it will be
  // re-added slowly later.
  //
  // blendIn/blendOut replace the old scrollTop/scrollBot fields (removed
  // because they were editable here but never read by player.js's runtime).
  // These ARE read — see setupScrollPlayback's updateSceneBlend in player.js.
  function defaultConfig() {
    return {
      linkedId: '',
      playback: 'scroll',
      waitForTimeline: false,
      pingpong: false,
      playOnce: false,
      blendIn:  0,   // % of this scene's own container height, eased in from the previous scene's frame
      blendOut: 0,   // % of this scene's own container height, eased out toward the next scene's frame
      pan:   { enabled: false, damping: 0, button: 'right', limited: false, radius: 500 },
      zoom:  { enabled: false, limited: false, range: 500 },
      gyro:  false,
    };
  }

  // Merge sparse raw config with defaults, producing a complete config object.
  function mergeWithDefault(raw) {
    const def = defaultConfig();
    const r = raw || {};
    return { ...def, ...r,
      pan:   { ...def.pan,   ...(r.pan   || {}) },
      zoom:  { ...def.zoom,  ...(r.zoom  || {}) },
    };
  }

  // Strip keys that equal their default values so saved JSON is minimal.
  // Disabled blocks are omitted entirely; enabled blocks keep only non-default fields.
  // Missing key = false/default when read back via mergeWithDefault.
  function sparsify(cfg) {
    const def = defaultConfig();
    const out = {};
    if (cfg.linkedId        !== def.linkedId)        out.linkedId        = cfg.linkedId;
    if (cfg.playback        !== def.playback)        out.playback        = cfg.playback;
    if (cfg.waitForTimeline !== def.waitForTimeline) out.waitForTimeline = cfg.waitForTimeline;
    if (cfg.pingpong        !== def.pingpong)        out.pingpong        = cfg.pingpong;
    if (cfg.playOnce        !== def.playOnce)        out.playOnce        = cfg.playOnce;
    if (cfg.blendIn         !== def.blendIn)         out.blendIn         = cfg.blendIn;
    if (cfg.blendOut        !== def.blendOut)        out.blendOut        = cfg.blendOut;
    if (cfg.gyro            !== def.gyro)            out.gyro            = cfg.gyro;
    for (const key of ['pan', 'zoom']) {
      const blk  = cfg[key] || {};
      const dblk = def[key] || {};
      if (!blk.enabled) continue; // disabled → omit entire block
      const sparse = { enabled: true };
      for (const [k, v] of Object.entries(blk)) {
        if (k !== 'enabled' && v !== dblk[k]) sparse[k] = v;
      }
      out[key] = sparse;
    }
    return out;
  }

  function hasActiveConfig(cfg) {
    return !!(cfg.pan?.enabled || cfg.zoom?.enabled || cfg.pingpong || cfg.playOnce || cfg.gyro
      || cfg.blendIn > 0 || cfg.blendOut > 0);
  }

  // Fetch the HTML source file and read config from JS sentinels.
  // Ensures the editor reflects any saves that happened in a previous session.
  async function loadPageState() {
    // ── HTML source read (requires API) ────────────────────────────────────────
    if (S.apiOnline) {
      try {
        const rel = window.location.pathname.replace(/^\//, '').split('?')[0] || 'index.html';
        const res = await fetch(`/hs-api/file?path=${encodeURIComponent(rel)}`);
        if (res.ok) {
          const html = await res.text();
          const doc  = new DOMParser().parseFromString(html, 'text/html');

          function mergeScenes(parsed) {
            for (const [name, raw] of Object.entries(parsed)) {
              S.sceneConfigs[name] = mergeWithDefault(raw);
            }
          }

          // Read all config from JS source sentinels
          for (const scriptEl of doc.querySelectorAll('script')) {
            const src = scriptEl.textContent;
            const mScenes = src.match(/^[ \t]*scenes\s*:\s*(\{.+\}),?\s*\/\/\s*hs-scenes\s*$/m);
            if (mScenes) { try { mergeScenes(JSON.parse(mScenes[1])); } catch { } }
            const mMasks = src.match(/^[ \t]*masks\s*:\s*(\{.+\}),?\s*\/\/\s*hs-masks\s*$/m);
            if (mMasks) { try {
              for (const [name, raw] of Object.entries(JSON.parse(mMasks[1]))) S.maskConfigs[name] = { ...raw };
            } catch { } }
            const mClips = src.match(/^[ \t]*clips\s*:\s*(\[.*\]),?\s*\/\/\s*hs-clips\s*$/m);
            if (mClips && !S.assets.length) { try {
              // Each entry is either a bare url string, or {url, splatsDir,
              // defaults} once a splats path and/or per-axis default
              // variant has been set — see saveAssetsAttr.
              S.assets = JSON.parse(mClips[1]).map(entry => {
                const obj = typeof entry === 'string' ? { url: entry } : entry;
                return {
                  url: obj.url ?? '', splatsDir: obj.splatsDir ?? '', defaults: obj.defaults ?? {},
                  status: 'idle', clipIds: [], axes: {}, states: {}, parts: {}, masks: [],
                };
              });
            } catch { } }
            const mSh = src.match(/^[ \t]*sh\s*:\s*(\d+)[^\n]*\/\/\s*hs-sh\s*$/m);
            if (mSh) S.globalSh = +mSh[1];
            const mZi = src.match(/^[ \t]*zIndex\s*:\s*(-?\d+)[^\n]*\/\/\s*hs-zi\s*$/m);
            if (mZi) { S.globalZIndex = +mZi[1]; if (S.entry?.root) S.entry.root.style.zIndex = mZi[1]; }
            const mAa = src.match(/^[ \t]*aaDilation\s*:\s*([0-9.]+)[^\n]*\/\/\s*hs-aa\s*$/m);
            if (mAa) { S.globalAaDilation = +mAa[1]; S.entry?.api?.setAaDilation?.(S.globalAaDilation); }
            const mAnim = src.match(/^\s*animation\s*:\s*(['"])(.*?)\1/m);
            if (mAnim && !el('an').value) el('an').value = mAnim[2];
            const mPartsDir = src.match(/^\s*partsDir\s*:\s*(['"])(.*?)\1/m);
            if (mPartsDir && !el('pd').value) { el('pd').value = mPartsDir[2]; S.lastSavedDir = mPartsDir[2]; }
            // Derive partsDir from saved compact parts line (// hs-parts sentinel)
            if (!el('pd').value) {
              const mHsParts = src.match(/^[ \t]*parts\s*:\s*(\{.+\}),?\s*\/\/\s*hs-parts\s*$/m);
              if (mHsParts) {
                try {
                  const map = JSON.parse(mHsParts[1]);
                  const firstVal = Object.values(map)[0] || '';
                  const lastSlash = firstVal.lastIndexOf('/');
                  if (lastSlash >= 0) el('pd').value = firstVal.slice(0, lastSlash + 1);
                } catch { }
              }
            }
            // Derive partsDir from existing multi-line parts block (first path value)
            if (!el('pd').value) {
              const mBlock = src.match(/parts\s*:\s*\{([\s\S]+?)\}/);
              if (mBlock) {
                const firstPath = mBlock[1].match(/['"]([^'"]+\.(?:spz|ply|splat))['"]/)?.[1];
                if (firstPath) {
                  const lastSlash = firstPath.lastIndexOf('/');
                  if (lastSlash >= 0) el('pd').value = firstPath.slice(0, lastSlash + 1);
                }
              }
            }
          }
          window.__hsSceneConfigs = window.__hsSceneConfigs || {};
          Object.assign(window.__hsSceneConfigs, S.sceneConfigs);
          window.__hsMaskConfigs = window.__hsMaskConfigs || {};
          Object.assign(window.__hsMaskConfigs, S.maskConfigs);
        }
      } catch { }
    }
    renderGlobalControls();

    // ── Animation markers (direct fetch, no API needed) ────────────────────────
    if (S.frameCount === 0) {
      const animUrl = el('an').value.trim();
      if (animUrl) {
        try {
          const r    = await fetch(animUrl);
          if (!r.ok) throw new Error(r.status);
          const data = await r.json();
          const raw  = data.markers;
          S.markers    = Array.isArray(raw)
            ? Object.fromEntries(raw.map(m => [m.name, m.frame]))
            : (raw || {});
          S.frameCount = data.frameCount || 0;
          el('range').max        = Math.max(0, S.frameCount - 1);
          el('flbl').textContent = `0 / ${S.frameCount}`;
          setStatus(`${S.frameCount} frames · ${Object.keys(S.markers).length} markers`);
        } catch { /* animation file unreachable — wait for live player */ }
      }
    }
  }

  function debouncedSaveScenesAttr() {
    clearTimeout(_saveTimer);
    _saveTimer = setTimeout(saveScenesAttr, 300);
  }

  async function saveScenesAttr() {
    // Mirror full configs to global so viewer _syncCameraMode picks them up
    window.__hsSceneConfigs = window.__hsSceneConfigs || {};
    Object.assign(window.__hsSceneConfigs, S.sceneConfigs);

    if (!S.apiOnline) return;

    // Build sparse version for storage — omit keys matching defaults
    const sparse = {};
    for (const [name, cfg] of Object.entries(S.sceneConfigs)) sparse[name] = sparsify(cfg);

    try {
      await fetch('/hs-api/js-scenes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ page: window.location.pathname, scenes: sparse }),
      });
    } catch { }
  }

  function saveConfig(name, config) {
    S.sceneConfigs[name] = config;
    (window.__hsSceneConfigs = window.__hsSceneConfigs || {})[name] = config;
    debouncedSaveScenesAttr();
  }

  function debouncedSaveMasksAttr() {
    clearTimeout(_maskSaveTimer);
    _maskSaveTimer = setTimeout(saveMasksAttr, 300);
  }

  async function saveMasksAttr() {
    // Mirror full configs to global so the player picks up live feather overrides
    window.__hsMaskConfigs = window.__hsMaskConfigs || {};
    Object.assign(window.__hsMaskConfigs, S.maskConfigs);

    if (!S.apiOnline) return;

    // Build sparse version for storage — omit feather values matching the
    // volume's exported default so saved JSON is minimal. Covers both the
    // main animation's volumes and every asset's clip/transition masks.
    const vols = [
      ...(S.entry?.viewer?._animation?.volumes ?? []),
      ...S.assets.flatMap(a => a.masks ?? []),
    ];
    const sparse = {};
    for (const vol of vols) {
      const cfg = S.maskConfigs[vol.name];
      if (cfg && cfg.feather != null && cfg.feather !== (vol.softEdge ?? 0.05)) {
        sparse[vol.name] = { feather: cfg.feather };
      }
    }

    try {
      await fetch('/hs-api/js-masks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ page: window.location.pathname, masks: sparse }),
      });
    } catch { }
  }

  function saveMaskConfig(name, config) {
    S.maskConfigs[name] = config;
    (window.__hsMaskConfigs = window.__hsMaskConfigs || {})[name] = config;
    debouncedSaveMasksAttr();
  }

  const SH_LABELS = ['0 — dc only', '1', '2', '3 — full'];

  function renderGlobalControls() {
    const wrap = el('global-cfg');
    if (!wrap) return;
    wrap.innerHTML = '';

    // SH degree — custom dropdown (avoids overflow:hidden clipping on panel)
    const row1 = wrap.appendChild(document.createElement('div'));
    row1.className = '__hs-attr-row';
    const lbl1 = row1.appendChild(document.createElement('span'));
    lbl1.className = '__hs-attr-lbl'; lbl1.textContent = 'SH degree';
    const dropBtn = document.createElement('button');
    dropBtn.className = '__hs-drop-btn';
    dropBtn.id = '__hs-sh-sel';
    dropBtn.textContent = SH_LABELS[S.globalSh] ?? S.globalSh;
    row1.appendChild(dropBtn);

    let shMenu = null;
    const closeShMenu = () => { if (shMenu) { shMenu.remove(); shMenu = null; } };
    dropBtn.addEventListener('click', e => {
      e.stopPropagation();
      if (shMenu) { closeShMenu(); return; }
      const rect = dropBtn.getBoundingClientRect();
      const menu = document.createElement('div');
      menu.className = '__hs-drop-menu';
      menu.style.cssText = `left:${rect.left}px;top:${rect.bottom + 2}px;min-width:${rect.width}px;`;
      shMenu = menu;
      for (let i = 0; i < SH_LABELS.length; i++) {
        const item = document.createElement('div');
        item.className = '__hs-drop-item' + (i === S.globalSh ? ' active' : '');
        item.textContent = SH_LABELS[i];
        item.addEventListener('click', () => {
          S.globalSh = i;
          dropBtn.textContent = SH_LABELS[i];
          S.entry?.api?.setShDegree?.(i);
          saveGlobalSh();
          closeShMenu();
        });
        menu.appendChild(item);
      }
      document.body.appendChild(menu);
      setTimeout(() => document.addEventListener('click', closeShMenu, { once: true }), 0);
    });

    // z-index
    const row2 = wrap.appendChild(document.createElement('div'));
    row2.className = '__hs-attr-row';
    const lbl2 = row2.appendChild(document.createElement('span'));
    lbl2.className = '__hs-attr-lbl'; lbl2.textContent = 'z-index';
    const ziInp = document.createElement('input');
    ziInp.type = 'number'; ziInp.className = '__hs-ninp'; ziInp.style.width = '70px';
    ziInp.value = S.globalZIndex;
    ziInp.addEventListener('change', () => {
      const v = parseInt(ziInp.value) || 0;
      ziInp.value = v;
      S.globalZIndex = v;
      if (S.entry?.root) S.entry.root.style.zIndex = String(v);
      saveGlobalZIndex();
    });
    row2.appendChild(ziInp);

    // AA dilation
    wrap.appendChild(mkRow('AA dilation', mkNum(S.globalAaDilation, 0, 0.5, 0.01, null, v => {
      S.globalAaDilation = v;
      S.entry?.viewer?.setAaDilation?.(v);
      clearTimeout(_aaSaveTimer);
      _aaSaveTimer = setTimeout(() => saveGlobalAaDilation(), 400);
    }).el));
  }

  async function saveGlobalSh() {
    if (!S.apiOnline) return;
    try {
      await fetch('/hs-api/js-sh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ page: window.location.pathname, sh: S.globalSh }),
      });
    } catch { }
  }

  async function saveGlobalZIndex() {
    if (!S.apiOnline) return;
    try {
      await fetch('/hs-api/js-zIndex', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ page: window.location.pathname, zIndex: S.globalZIndex }),
      });
    } catch { }
  }

  async function saveGlobalAaDilation() {
    if (!S.apiOnline) return;
    try {
      await fetch('/hs-api/js-aaDilation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ page: window.location.pathname, aaDilation: S.globalAaDilation }),
      });
    } catch { }
  }

  // ── Asset clip files (product customization — see export_holosplat_clips.py) ─

  function assetNameLabel(asset) {
    if (asset.status === 'loading') return 'loading…';
    if (asset.status === 'error')   return 'error';
    if (!asset.clipIds?.length)     return '—';
    return asset.clipIds.join(', ');
  }

  function renderAssetsList() {
    const list = el('assets-list');
    if (!list) return;
    list.innerHTML = '';
    S.assets.forEach((asset, i) => {
      const row = list.appendChild(document.createElement('div'));
      row.className = '__hs-fr __hs-asset-row';

      const idx = row.appendChild(document.createElement('span'));
      idx.className = '__hs-asset-idx';
      idx.textContent = `Asset ${i}`;

      const nameEl = row.appendChild(document.createElement('span'));
      nameEl.className = '__hs-asset-name' + (asset.status === 'error' ? ' err' : asset.status === 'loading' ? ' pending' : '');
      nameEl.textContent = assetNameLabel(asset);
      nameEl.title = asset.clipIds?.join('\n') ?? '';

      const inp = row.appendChild(document.createElement('input'));
      inp.placeholder = '/scenes/headphones.json';
      inp.spellcheck  = false;
      inp.value = asset.url ?? '';
      inp.addEventListener('blur', () => {
        const v = inp.value.trim();
        if (v === asset.url) return;
        asset.url = v;
        debouncedSaveAssetsAttr();
        if (v) loadAsset(i);
      });
      inp.addEventListener('keydown', e => { if (e.key === 'Enter') inp.blur(); });

      const reloadBtn = row.appendChild(document.createElement('button'));
      reloadBtn.className = '__hs-btn';
      reloadBtn.title = 'Reload';
      reloadBtn.textContent = '↺';
      reloadBtn.addEventListener('click', () => { refreshDiskFiles(); loadAsset(i); });

      const removeBtn = row.appendChild(document.createElement('button'));
      removeBtn.className = '__hs-btn __hs-asset-remove';
      removeBtn.title = 'Remove';
      removeBtn.textContent = '✕';
      removeBtn.addEventListener('click', () => removeAsset(i));

      // Row 2: where this asset's referenced splat files actually live —
      // the clip JSON itself has no idea, so the designer points it here.
      const splatRow = list.appendChild(document.createElement('div'));
      splatRow.className = '__hs-splat-row';

      const dirInp = splatRow.appendChild(document.createElement('input'));
      dirInp.placeholder = '/scenes/headphones/';
      dirInp.spellcheck  = false;
      dirInp.value = asset.splatsDir ?? '';
      dirInp.title = "Directory where this asset's referenced splat files live";
      dirInp.addEventListener('blur', () => {
        const v = dirInp.value.trim();
        if (v === asset.splatsDir) return;
        asset.splatsDir = v;
        debouncedSaveAssetsAttr();
        refreshDiskFiles();
        loadAsset(i);
      });
      dirInp.addEventListener('keydown', e => { if (e.key === 'Enter') dirInp.blur(); });

      const summaryEl = splatRow.appendChild(document.createElement('span'));
      summaryEl.className = '__hs-splat-summary';
      renderSplatSummary(summaryEl, buildAssetPartRows(asset), `Asset ${i} splats`);

      // Row 2b: masks declared by this asset's clips/transitions, with a
      // feather-edit modal (see renderMaskSummary/openMasksModal).
      const maskRow = list.appendChild(document.createElement('div'));
      maskRow.className = '__hs-splat-row';
      const maskLbl = maskRow.appendChild(document.createElement('span'));
      maskLbl.className = '__hs-fl'; maskLbl.style.width = 'auto'; maskLbl.textContent = 'Masks';
      const maskSummaryEl = maskRow.appendChild(document.createElement('span'));
      maskSummaryEl.className = '__hs-splat-summary';
      renderMaskSummary(maskSummaryEl, asset.masks ?? [], `Asset ${i} masks`);

      // Row 3: one dropdown per variant axis discovered in the asset JSON
      // (see export_holosplat_asset.py), picking that axis's default value.
      // How switching values actually animates anything isn't implemented
      // yet — this just records which value loads by default.
      const axisNames = Object.keys(asset.axes ?? {});
      if (axisNames.length) {
        const axisRow = list.appendChild(document.createElement('div'));
        axisRow.className = '__hs-fr';
        for (const axis of axisNames) {
          const lbl = axisRow.appendChild(document.createElement('span'));
          lbl.className = '__hs-fl'; lbl.textContent = axis;
          const sel = axisRow.appendChild(document.createElement('select'));
          sel.className = '__hs-el-sel'; sel.style.width = 'auto';
          for (const value of asset.axes[axis]) {
            const o = document.createElement('option');
            o.value = value; o.textContent = value;
            if (asset.defaults?.[axis] === value) o.selected = true;
            sel.appendChild(o);
          }
          sel.addEventListener('change', () => {
            asset.defaults = { ...(asset.defaults ?? {}), [axis]: sel.value };
            debouncedSaveAssetsAttr();
            loadAsset(i);
          });
        }
      }

      // Row 4: one dropdown per state axis discovered in the asset JSON
      // (see export_holosplat_asset.py's "state: <axis>=<value>" markers),
      // picking that axis's default state — same defaults map as the axis
      // row above (Viewer applies whichever one matches via loadClips()).
      const stateAxisNames = Object.keys(asset.states ?? {});
      if (stateAxisNames.length) {
        const stateRow = list.appendChild(document.createElement('div'));
        stateRow.className = '__hs-fr';
        for (const axis of stateAxisNames) {
          const lbl = stateRow.appendChild(document.createElement('span'));
          lbl.className = '__hs-fl'; lbl.textContent = `state: ${axis}`;
          const sel = stateRow.appendChild(document.createElement('select'));
          sel.className = '__hs-el-sel'; sel.style.width = 'auto';
          for (const value of asset.states[axis]) {
            const o = document.createElement('option');
            o.value = value; o.textContent = value;
            if (asset.defaults?.[axis] === value) o.selected = true;
            sel.appendChild(o);
          }
          sel.addEventListener('change', () => {
            asset.defaults = { ...(asset.defaults ?? {}), [axis]: sel.value };
            debouncedSaveAssetsAttr();
            loadAsset(i);
          });
        }
      }
    });
  }

  async function loadAsset(i, { skipIfLoaded = false } = {}) {
    const asset = S.assets[i];
    if (!asset?.url || !S.entry) return;
    // Unload whatever this slot previously contributed before reloading, so
    // editing/reloading a URL doesn't leave stale clips registered under it.
    // Skipped when skipIfLoaded resolves to a no-op reload below (nothing to
    // unload yet in that case anyway, since clipIds is still empty on connect).
    if (asset.clipIds?.length) S.entry.api.unloadClips(asset.clipIds);
    asset.status   = 'loading';
    asset.clipIds  = [];
    // Deliberately leave asset.axes/parts alone here — clearing them would
    // collapse the axis dropdowns and splat-summary row for the duration of
    // the fetch, then snap them back once it resolves. Keep showing the
    // previous data until the new data is ready to replace it.
    renderAssetsList();
    try {
      // Fetch the JSON directly first — loadClips alone only returns clip
      // ids, but we need axes/parts/defaults resolved before asking the
      // player to load it, so the asset's geometry loads on the first try.
      const res = await fetch(asset.url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      asset.axes  = data.axes  ?? {};
      asset.parts = data.parts ?? {};
      // State axes (see export_holosplat_asset.py's "state: <axis>=<value>"
      // markers) — value list per axis, used by the default-state dropdown
      // above. Falls back to each axis's own exported default below.
      asset.states = Object.fromEntries(
        Object.entries(data.states ?? {}).map(([axis, s]) => [axis, Object.keys(s.markers ?? {})])
      );

      // Mask volumes this asset's clips/transitions declare — deduped by
      // name, since the same volume can recur across multiple clips.
      const maskMap = {};
      for (const c of data.clips ?? []) for (const m of c.masks ?? []) maskMap[m.name] = m.softEdge ?? 0.05;
      for (const t of Object.values(data.transitions ?? {})) for (const m of t.masks ?? []) maskMap[m.name] = m.softEdge ?? 0.05;
      for (const s of Object.values(data.states ?? {})) for (const m of s.masks ?? []) maskMap[m.name] = m.softEdge ?? 0.05;
      asset.masks = Object.entries(maskMap).map(([name, softEdge]) => ({ name, softEdge }));

      // Drop defaults for axes that no longer exist; fill in a first value
      // for any new axis so every axis always has a default once loaded —
      // for state axes, prefer the axis's own exported default value over
      // an arbitrary first one.
      const defaults = {};
      for (const [axis, values] of Object.entries(asset.axes)) {
        defaults[axis] = (asset.defaults?.[axis] && values.includes(asset.defaults[axis]))
          ? asset.defaults[axis] : values[0];
      }
      for (const [axis, values] of Object.entries(asset.states)) {
        if (asset.defaults?.[axis] && values.includes(asset.defaults[axis])) {
          defaults[axis] = asset.defaults[axis];
        } else {
          defaults[axis] = data.states[axis]?.default ?? values[0];
        }
      }
      asset.defaults = defaults;

      // The player's own boot sequence already loads every `clips:` entry
      // (see player.js), so the editor's automatic on-connect pass
      // (loadAllAssets, skipIfLoaded:true) would otherwise reload the exact
      // same asset a second time — duplicating every variant fetch (extra
      // 404s) and, worse, briefly replacing the already-fully-loaded
      // multi-variant scene with the single-default-variant one mid-scroll,
      // which is what caused the flicker/stutter. Detect that case by
      // checking whether everything this asset declares (parts, axis
      // transitions, and/or clips — headphones-rig.json, for instance, is
      // transitions+parts only, with no "clips" of its own) is already
      // registered in the live viewer, and if so just adopt its ids instead
      // of reloading.
      const partIds        = Object.keys(data.parts ?? {});
      const transitionAxes = Object.keys(data.transitions ?? {});
      const stateAxes       = Object.keys(data.states ?? {});
      const clipIds        = (data.clips ?? []).map(c => c.id);
      const alreadyLoaded = skipIfLoaded
        && (partIds.length > 0 || transitionAxes.length > 0 || stateAxes.length > 0 || clipIds.length > 0)
        && partIds.every(id => S.entry.viewer._partIndex[id]?.length)
        && transitionAxes.every(axis => S.entry.viewer._transitions[axis])
        && stateAxes.every(axis => S.entry.viewer._states[axis])
        && clipIds.every(id => S.entry.viewer._clips[id]);
      asset.clipIds = alreadyLoaded
        ? clipIds
        : await S.entry.api.loadClips(asset.url, { splatsDir: asset.splatsDir, defaults: asset.defaults });
      asset.status  = 'ok';

      // Re-apply any saved feather overrides — loadClips() just reset these
      // masks to their default softEdge, so the persisted value needs pushing
      // back in (mirrors buildMaskCard's same re-apply for animation volumes).
      // Not needed when we skipped the reload above — the boot sequence's
      // own opts.masks handling (see player.js) already applied these.
      if (!alreadyLoaded) {
        for (const m of asset.masks) {
          if (!S.maskConfigs[m.name]) S.maskConfigs[m.name] = {};
          const cfg = S.maskConfigs[m.name];
          if (cfg.feather != null && cfg.feather !== (m.softEdge ?? 0.05)) {
            S.entry.api.setMaskFeather(m.name, cfg.feather);
          }
        }
      }

      debouncedSaveAssetsAttr();
    } catch (err) {
      asset.status  = 'error';
      asset.clipIds = [];
      asset.masks   = [];
      console.error(`[HoloSplat] asset ${i} ("${asset.url}") failed to load:`, err);
    }
    renderAssetsList();
  }

  function addAsset() {
    S.assets.push({ url: '', splatsDir: '', defaults: {}, status: 'idle', clipIds: [], axes: {}, states: {}, parts: {}, masks: [] });
    renderAssetsList();
    debouncedSaveAssetsAttr();
  }

  function removeAsset(i) {
    const asset = S.assets[i];
    if (asset?.clipIds?.length) S.entry?.api.unloadClips(asset.clipIds);
    S.assets.splice(i, 1);
    renderAssetsList();
    debouncedSaveAssetsAttr();
  }

  function debouncedSaveAssetsAttr() {
    clearTimeout(_assetsSaveTimer);
    _assetsSaveTimer = setTimeout(saveAssetsAttr, 400);
  }

  async function saveAssetsAttr() {
    if (!S.apiOnline) return;
    try {
      await fetch('/hs-api/js-clips', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          page:  window.location.pathname,
          clips: S.assets
            .filter(a => a.url)
            .map(a => {
              if (!a.splatsDir && !Object.keys(a.defaults ?? {}).length) return a.url;
              const obj = { url: a.url };
              if (a.splatsDir) obj.splatsDir = a.splatsDir;
              if (Object.keys(a.defaults ?? {}).length) obj.defaults = a.defaults;
              return obj;
            }),
        }),
      });
    } catch { }
  }

  // Load (or reload) every configured asset's clips — called once the
  // player/animation is connected, mirroring how parts/masks apply on connect.
  function loadAllAssets() {
    S.assets.forEach((asset, i) => { if (asset.url) loadAsset(i, { skipIfLoaded: true }); });
  }

  function switchTab(tab) {
    for (const btn of document.querySelectorAll('.__hs-tabbtn'))
      btn.classList.toggle('active', btn.dataset.tab === tab);
    for (const panel of document.querySelectorAll('.__hs-tabpanel'))
      panel.classList.toggle('active', panel.dataset.tab === tab);
  }

  function saveUiState() {
    const state = {
      _scenes: {}, _masks: {}, sh: S.globalSh,
      panelMin: el('panel').classList.contains('minimized'),
      tlMin: el('tl')?.classList.contains('collapsed') ?? false,
      tab: document.querySelector('.__hs-tabbtn.active')?.dataset.tab || 'scenes',
      focalMarker: S.showFocalMarker,
    };
    for (const hd of document.querySelectorAll('.__hs-cpane-hd')) {
      const title = hd.querySelector('.__hs-cpane-title')?.textContent?.trim();
      const body  = hd.nextElementSibling;
      if (title) state[title] = body.classList.contains('closed');
    }
    for (const card of document.querySelectorAll('.__hs-scard[data-sc-key]')) {
      const bd = card.querySelector('.__hs-scard-bd');
      if (bd) state._scenes[card.dataset.scKey] = bd.classList.contains('open');
    }
    for (const card of document.querySelectorAll('.__hs-scard[data-mask-key]')) {
      const bd = card.querySelector('.__hs-scard-bd');
      if (bd) state._masks[card.dataset.maskKey] = bd.classList.contains('open');
    }
    try { localStorage.setItem('__hs-ui', JSON.stringify(state)); } catch {}
  }

  function loadUiState() {
    try {
      const saved = JSON.parse(localStorage.getItem('__hs-ui') || '{}');
      for (const hd of document.querySelectorAll('.__hs-cpane-hd')) {
        const title = hd.querySelector('.__hs-cpane-title')?.textContent?.trim();
        const body  = hd.nextElementSibling;
        const tri   = hd.querySelector('.__hs-cpane-tri');
        if (title && saved[title] === true) {
          body.classList.add('closed');
          tri.textContent = '▶';
        }
      }
      const scenes = saved._scenes || {};
      for (const card of document.querySelectorAll('.__hs-scard[data-sc-key]')) {
        const tri = card.querySelector('.__hs-stri');
        const bd  = card.querySelector('.__hs-scard-bd');
        if (bd && scenes[card.dataset.scKey]) {
          bd.classList.add('open');
          if (tri) tri.textContent = '▼';
        }
      }
      const masks = saved._masks || {};
      for (const card of document.querySelectorAll('.__hs-scard[data-mask-key]')) {
        const tri = card.querySelector('.__hs-stri');
        const bd  = card.querySelector('.__hs-scard-bd');
        if (bd && masks[card.dataset.maskKey]) {
          bd.classList.add('open');
          if (tri) tri.textContent = '▼';
        }
      }
      if (saved.sh != null) {
        S.globalSh = saved.sh;
        const shBtn = document.getElementById('__hs-sh-sel');
        if (shBtn) shBtn.textContent = SH_LABELS[S.globalSh] ?? S.globalSh;
        S.entry?.api?.setShDegree?.(S.globalSh)?.catch(() => {});
      }
      if (saved.focalMarker) {
        S.showFocalMarker = true;
        if (S.focalMarkerCb) S.focalMarkerCb.checked = true;
      }
      if (saved.panelMin) {
        el('panel').classList.add('minimized');
        el('min').textContent = '+';
        el('min').title = 'Restore panel';
      }
      if (saved.tab) switchTab(saved.tab);
      S._tlMinSaved = saved.tlMin === true;
    } catch {}
  }

  function renderScenes() {
    const list = el('scenes-list');
    if (!S.markers || !Object.keys(S.markers).length) {
      list.innerHTML = '<div class="__hs-empty">No animation loaded</div>';
      return;
    }
    const scenes = computeScenes();

    S.sceneCards = {};
    list.innerHTML = '';
    for (const scene of scenes) {
      const config = S.sceneConfigs[scene.name] || defaultConfig();
      S.sceneConfigs[scene.name] = config;
      list.appendChild(buildSceneCard(scene, config));
    }
    renderTimeline();
    loadUiState();
  }

  function buildSceneCard(scene, cfg) {
    const card = document.createElement('div');
    card.className = '__hs-scard';
    card.dataset.scKey = scene.name;
    if (hasActiveConfig(cfg)) card.classList.add('__hs-scard--configured');

    // ── Helpers ──────────────────────────────────────────────────────────────
    const update = (updater) => {
      const c = S.sceneConfigs[scene.name] || defaultConfig();
      updater(c);
      saveConfig(scene.name, c);
      card.classList.toggle('__hs-scard--configured', hasActiveConfig(c));
      renderTimeline();
    };

    // Block: collapsible section with enable toggle in header.
    // buildBody(bbd, enable) — enable() programmatically turns the block on (for child controls).
    function mkBlock(container, name, enabled, buildBody) {
      const blk = container.appendChild(document.createElement('div'));
      blk.className = '__hs-ablock';
      const bhd = blk.appendChild(document.createElement('div'));
      bhd.className = '__hs-ablk-hd';
      const btri = bhd.appendChild(document.createElement('span'));
      btri.className = '__hs-stri'; btri.textContent = enabled ? '▼' : '▶';
      const bnm = bhd.appendChild(document.createElement('span'));
      bnm.className = '__hs-ablk-nm'; bnm.textContent = name;
      const { el: togEl, cb: togCb } = mkToggle(enabled, v => {
        update(c => {
          if (name === 'pan') c.pan.enabled = v;
          else if (name === 'zoom') c.zoom.enabled = v;
          else if (name === 'phone gyroscope') c.gyro = v;
        });
        if (bbd) { bbd.classList.toggle('open', v); btri.textContent = v ? '▼' : '▶'; }
      });
      bhd.appendChild(togEl);
      // Callable by child controls: turns the block on if it isn't already
      const enable = () => {
        if (togCb.checked) return;
        togCb.checked = true;
        update(c => {
          if (name === 'pan') c.pan.enabled = true;
          else if (name === 'zoom') c.zoom.enabled = true;
        });
        if (bbd) { bbd.classList.add('open'); btri.textContent = '▼'; }
      };
      let bbd = null;
      if (buildBody) {
        bbd = blk.appendChild(document.createElement('div'));
        bbd.className = `__hs-ablk-bd${enabled ? ' open' : ''}`;
        bhd.addEventListener('click', e => {
          if (togEl.contains(e.target)) return;
          const open = bbd.classList.toggle('open');
          btri.textContent = open ? '▼' : '▶';
        });
        buildBody(bbd, enable);
      }
    }

    // ── Header ───────────────────────────────────────────────────────────────
    const hd = card.appendChild(document.createElement('div'));
    hd.className = '__hs-scard-hd';

    // Progress fill bar (absolutely-positioned, behind flex content)
    const barEl = document.createElement('div');
    barEl.className = '__hs-scard-bar';
    hd.appendChild(barEl);

    // Play / pause button (leftmost)
    const playBtn = document.createElement('button');
    playBtn.className = '__hs-scard-play';
    playBtn.textContent = '▶';
    playBtn.title = 'Play this scene';
    playBtn.addEventListener('click', e => {
      e.stopPropagation();
      if (S.activeMarker === scene.name) {
        // Toggle pause/play for active scene
        S.entry?.api.setAnimationPaused(!S.entry.viewer._animPaused);
      } else {
        // Jump to scene start and play
        const anim = S.entry?.api.animation;
        if (anim) { anim.seekFrame(scene.from); S.entry.api.setAnimationPaused(false); }
        el('range').value = scene.from;
        el('flbl').textContent = `${scene.from} / ${S.frameCount}`;
      }
    });
    hd.appendChild(playBtn);

    // Expand triangle
    const tri = hd.appendChild(document.createElement('span'));
    tri.className = '__hs-stri'; tri.textContent = '▶';

    // Blue dot — visible when this scene has any feature active
    const dot = hd.appendChild(document.createElement('span'));
    dot.className = '__hs-scard-dot';

    // Linked-div badge — visible when a linkedId is configured
    const linkedBadge = hd.appendChild(document.createElement('span'));
    linkedBadge.className = '__hs-scard-linked' + (cfg.linkedId ? ' has-id' : '');
    linkedBadge.textContent = 'div';
    linkedBadge.title = cfg.linkedId ? `#${cfg.linkedId}` : '';

    // Scene name — scrubable when this scene is active
    const nm = hd.appendChild(document.createElement('span'));
    nm.className = '__hs-scard-nm'; nm.textContent = scene.name;
    nm.addEventListener('mousedown', e => {
      if (S.activeMarker !== scene.name) return;
      e.stopPropagation();
      const wasPaused = S.entry?.viewer._animPaused ?? true;
      S.entry?.api.setAnimationPaused(true);
      const rect = nm.getBoundingClientRect();
      const doSeek = clientX => {
        const t = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        const f = Math.round(scene.from + t * scene.frames);
        S.entry?.api.animation?.seekFrame(f);
        el('range').value = f;
        el('flbl').textContent = `${f} / ${S.frameCount}`;
      };
      doSeek(e.clientX);
      const onMove = ev => doSeek(ev.clientX);
      const onUp = () => {
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
        nm.style.userSelect = '';
        if (!wasPaused) S.entry?.api.setAnimationPaused(false);
      };
      nm.style.userSelect = 'none';
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    });

    const frSpan = hd.appendChild(document.createElement('span'));
    frSpan.className = '__hs-scard-fr'; frSpan.textContent = `${scene.frames}f`;
    const rngSpan = hd.appendChild(document.createElement('span'));
    rngSpan.className = '__hs-scard-range'; rngSpan.textContent = `${scene.from}–${scene.to}`;

    // Register for rAF updates
    S.sceneCards[scene.name] = { playBtn, barEl, from: scene.from, frames: scene.frames };

    const bd = card.appendChild(document.createElement('div'));
    bd.className = '__hs-scard-bd';
    hd.addEventListener('click', e => {
      if (e.target === playBtn || playBtn.contains(e.target)) return;
      const open = bd.classList.toggle('open');
      tri.textContent = open ? '▼' : '▶';
      saveUiState();
    });

    // ── HTML element picker ───────────────────────────────────────────────────
    {
      const selRow = bd.appendChild(document.createElement('div'));
      selRow.className = '__hs-sel-row';
      const lbl = selRow.appendChild(document.createElement('span'));
      lbl.className = '__hs-sel-lbl'; lbl.textContent = 'html element';
      const sel = selRow.appendChild(document.createElement('select'));
      sel.className = '__hs-el-sel';
      sel.innerHTML = `<option value="">— none —</option>`;
      const pageEls = scanPageEls();
      const linkedId = cfg.linkedId || '';
      for (const e of pageEls)
        sel.innerHTML += `<option value="${e.id}"${e.id === linkedId ? ' selected' : ''}>#${e.id} &lt;${e.tag}&gt;</option>`;
      sel.addEventListener('change', () => {
        update(c => { c.linkedId = sel.value; });
        linkedBadge.classList.toggle('has-id', !!sel.value);
        linkedBadge.title = sel.value ? `#${sel.value}` : '';
      });
    }

    bd.appendChild(document.createElement('div')).className = '__hs-sdiv';

    // ── Scene playback ────────────────────────────────────────────────────────
    {
      const sec = bd.appendChild(document.createElement('div'));
      sec.style.cssText = 'padding:4px 0 6px;';
      const hdr = sec.appendChild(document.createElement('div'));
      hdr.style.cssText = 'font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#999;padding:2px 0 6px;';
      hdr.textContent = 'scene playback';

      const rbRow = sec.appendChild(document.createElement('div'));
      rbRow.className = '__hs-attr-row';
      for (const [value, label] of [['scroll', 'scroll'], ['auto', 'auto']]) {
        const rbLbl = rbRow.appendChild(document.createElement('label'));
        rbLbl.style.cssText = 'display:inline-flex;align-items:center;gap:4px;margin-right:12px;cursor:pointer;font-size:0.85rem;';
        const rb = document.createElement('input');
        rb.type = 'radio'; rb.name = `pb-${scene.name}`; rb.value = value;
        rb.checked = (cfg.playback || 'scroll') === value;
        rb.addEventListener('change', () => { if (rb.checked) update(c => c.playback = value); });
        rbLbl.appendChild(rb);
        rbLbl.appendChild(document.createTextNode(label));
      }

      // Wait for timeline toggle
      const wRow = sec.appendChild(document.createElement('div'));
      wRow.className = '__hs-attr-row';
      wRow.style.marginTop = '4px';
      const wLbl = wRow.appendChild(document.createElement('span'));
      wLbl.className = '__hs-attr-lbl'; wLbl.textContent = 'wait for timeline';
      wRow.appendChild(mkToggle(cfg.waitForTimeline ?? false, v => update(c => c.waitForTimeline = v)).el);
    }

    bd.appendChild(document.createElement('div')).className = '__hs-sdiv';

    // ── Ping-pong ─────────────────────────────────────────────────────────────
    bd.appendChild(mkRow('ping-pong', mkToggle(cfg.pingpong, v => update(c => c.pingpong = v)).el));

    // ── Play once ────────────────────────────────────────────────────────────
    bd.appendChild(mkRow('play once', mkToggle(cfg.playOnce, v => update(c => c.playOnce = v)).el));

    bd.appendChild(document.createElement('div')).className = '__hs-sdiv';

    // ── Blend in/out ──────────────────────────────────────────────────────────
    // % of this scene's own linked container height, crossfading the actual
    // rendered camera/object pose with the adjacent scene's — see
    // updateSceneBlend in player.js. The blue zone overlay is transient
    // feedback (shown while actively dragging/typing a value), not a
    // persisted toggle.
    {
      const showZone = () => showBlendOverlay(
        cfg.linkedId ? document.getElementById(cfg.linkedId) : null,
        cfg.blendIn, cfg.blendOut
      );
      bd.appendChild(mkRow('blend in',
        mkNum(cfg.blendIn, 0, 100, 1, '%', v => { update(c => c.blendIn = v); showZone(); },
          active => active ? showZone() : hideBlendOverlay()).el));
      bd.appendChild(mkRow('blend out',
        mkNum(cfg.blendOut, 0, 100, 1, '%', v => { update(c => c.blendOut = v); showZone(); },
          active => active ? showZone() : hideBlendOverlay()).el));
    }

    bd.appendChild(document.createElement('div')).className = '__hs-sdiv';

    // NOTE: an "Orbit" block (follow pointer, damping, reset speed/ease,
    // h/v limits) used to live here — removed as part of a deliberate
    // cleanup; will be re-added slowly later.

    // ── Pan ───────────────────────────────────────────────────────────────────
    mkBlock(bd, 'pan', cfg.pan.enabled, (bbd, enable) => {
      bbd.appendChild(mkRow('damping',
        mkNum(cfg.pan.damping ?? 0, 0, 100, 1, '%', v => update(c => c.pan.damping = v)).el));

      {
        const sel = document.createElement('select');
        sel.className = '__hs-el-sel'; sel.style.width = 'auto';
        for (const opt of ['right', 'left']) {
          const o = document.createElement('option'); o.value = opt; o.textContent = opt + ' click';
          if ((cfg.pan.button ?? 'right') === opt) o.selected = true;
          sel.appendChild(o);
        }
        sel.addEventListener('change', () => update(c => c.pan.button = sel.value));
        bbd.appendChild(mkRow('drag button', sel));
      }

      const limitedDep = [];
      bbd.appendChild(mkRow('limited',
        mkToggle(cfg.pan.limited, v => {
          if (v) enable();
          update(c => c.pan.limited = v);
          limitedDep.forEach(r => r.style.display = v ? '' : 'none');
        }).el));

      const radRow = mkRow('radius',
        mkNum(cfg.pan.radius, 1, 99999, 1, null, v => update(c => c.pan.radius = v)).el);
      radRow.style.display = cfg.pan.limited ? '' : 'none';
      limitedDep.push(radRow); bbd.appendChild(radRow);
    });

    bd.appendChild(document.createElement('div')).className = '__hs-sdiv';

    // ── Zoom ──────────────────────────────────────────────────────────────────
    mkBlock(bd, 'zoom', cfg.zoom.enabled, (bbd, enable) => {
      const limitedDep = [];
      bbd.appendChild(mkRow('limited',
        mkToggle(cfg.zoom.limited, v => {
          if (v) enable();
          update(c => c.zoom.limited = v);
          limitedDep.forEach(r => r.style.display = v ? '' : 'none');
        }).el));

      const rangeRow = mkRow('range',
        mkNum(cfg.zoom.range, 1, 99999, 1, null, v => update(c => c.zoom.range = v)).el);
      rangeRow.style.display = cfg.zoom.limited ? '' : 'none';
      limitedDep.push(rangeRow); bbd.appendChild(rangeRow);
    });

    bd.appendChild(document.createElement('div')).className = '__hs-sdiv';

    // ── Phone gyroscope ───────────────────────────────────────────────────────
    mkBlock(bd, 'phone gyroscope', cfg.gyro, null);

    return card;
  }

  // ── Masks ────────────────────────────────────────────────────────────────────

  function renderMasks() {
    const list = el('masks-list');
    const vols = S.entry?.viewer?._animation?.volumes ?? [];
    renderMaskSummary(el('website-masks-summary'), vols.map(v => ({ name: v.name, softEdge: v.softEdge })), 'Main Animation masks');
    if (!vols.length) {
      list.innerHTML = '<div class="__hs-empty">No mask volumes</div>';
      return;
    }
    list.innerHTML = '';
    for (const vol of vols) {
      if (!S.maskConfigs[vol.name]) S.maskConfigs[vol.name] = {};
      list.appendChild(buildMaskCard(vol));
    }
    loadUiState();
  }

  function buildMaskCard(vol) {
    const card = document.createElement('div');
    card.className = '__hs-scard';
    card.dataset.maskKey = vol.name;

    const defaultFeather = vol.softEdge ?? 0.05;
    const cfg = S.maskConfigs[vol.name] || {};
    if (cfg.feather != null && cfg.feather !== defaultFeather) card.classList.add('__hs-scard--configured');

    // ── Header ───────────────────────────────────────────────────────────────
    const hd = card.appendChild(document.createElement('div'));
    hd.className = '__hs-scard-hd';

    const tri = hd.appendChild(document.createElement('span'));
    tri.className = '__hs-stri'; tri.textContent = '▶';

    const dot = hd.appendChild(document.createElement('span'));
    dot.className = '__hs-scard-dot';

    const nm = hd.appendChild(document.createElement('span'));
    nm.className = '__hs-scard-nm'; nm.style.cursor = 'default';
    nm.textContent = vol.name;

    const bd = card.appendChild(document.createElement('div'));
    bd.className = '__hs-scard-bd';
    hd.addEventListener('click', () => {
      const open = bd.classList.toggle('open');
      tri.textContent = open ? '▼' : '▶';
      saveUiState();
    });

    // ── Feather (soft-edge falloff, in scene units) ───────────────────────────
    const current = cfg.feather ?? defaultFeather;
    bd.appendChild(mkRow('feather', mkNum(current, 0, 10, 0.01, null, v => {
      applyMaskFeather(vol.name, v);
      card.classList.toggle('__hs-scard--configured', v !== defaultFeather);
    }).el));

    // Apply the saved/loaded feather to the live player immediately so the
    // viewport reflects the persisted value as soon as the card is built.
    if (current !== defaultFeather) S.entry?.api.setMaskFeather(vol.name, current);

    return card;
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

    // Minimize panel to just its toolbar
    el('min').addEventListener('click', () => {
      const minimized = el('panel').classList.toggle('minimized');
      el('min').textContent = minimized ? '+' : '−';
      el('min').title = minimized ? 'Restore panel' : 'Minimize panel';
      saveUiState();
    });

    // Tabs
    for (const btn of document.querySelectorAll('.__hs-tabbtn')) {
      btn.addEventListener('click', () => {
        switchTab(btn.dataset.tab);
        saveUiState();
      });
    }

    // Collapsible pane headers
    for (const hd of document.querySelectorAll('.__hs-cpane-hd')) {
      hd.addEventListener('click', () => {
        const body = hd.nextElementSibling;
        const tri  = hd.querySelector('.__hs-cpane-tri');
        const closed = body.classList.toggle('closed');
        tri.textContent = closed ? '▶' : '▼';
        saveUiState();
      });
    }

    {
      const { el: togEl, cb } = mkToggle(S.showFocalMarker, v => {
        S.showFocalMarker = v;
        if (!v && S.focalMarkerEl) S.focalMarkerEl.style.display = 'none';
        saveUiState();
      });
      S.focalMarkerCb = cb;
      el('fp-toggle-row').appendChild(mkRow('Visualize', togEl));
    }

    loadUiState();

    // Init page
    el('init').addEventListener('click', () => {
      if (document.querySelector('.hs-player')) {
        setStatus('Page already has a player', true); return;
      }
      const tag = document.createElement('hs-player');
      tag.id = 'hs-main';
      document.body.appendChild(tag);
      setStatus('Injected <hs-player id="hs-main"> — reload page to connect');
    });

    // Reload / sync
    el('ran').addEventListener('click',  reloadAnim);
    el('an').addEventListener('keydown', e => { if (e.key === 'Enter') reloadAnim(); });
    el('sync').addEventListener('click', syncParts);
    el('asset-add').addEventListener('click', addAsset);

    el('pd').addEventListener('blur', () => {
      renderParts();
      const dir = el('pd').value.trim();
      if (S.apiOnline && dir !== S.lastSavedDir) {
        S.lastSavedDir = dir;
        refreshDiskFiles(); // splats dir changed — disk listing may now be stale
        saveDirUrl(dir).then(() => setStatus(dir ? `Saved splats dir: ${dir}` : 'Cleared splats dir'));
      }
    });
    el('pd').addEventListener('keydown', e => { if (e.key === 'Enter') el('pd').blur(); });

    // ── Timeline ──────────────────────────────────────────────────────────────
    const tl = document.createElement('div');
    tl.id = '__hs-tl';
    tl.innerHTML = `
      <div id="__hs-tl-btns">
        <button class="__hs-tl-btn" id="__hs-tl-tostart" title="Jump to start">|◀</button>
        <button class="__hs-tl-btn" id="__hs-rev"        title="Play backward">◀</button>
        <button class="__hs-tl-btn" id="__hs-playpause"  title="Play">▶</button>
        <button class="__hs-tl-btn" id="__hs-tl-toend"   title="Jump to end">▶|</button>
      </div>
      <div id="__hs-tl-track">
        <div id="__hs-tl-labels"></div>
        <div id="__hs-tl-bar"><div id="__hs-tl-fill"></div></div>
      </div>
      <div id="__hs-tl-footer">
        <span id="__hs-flbl">no animation</span>
        <span id="__hs-scene-nm"></span>
      </div>
      <div id="__hs-tl-meta">
        <div>frame rate&nbsp;&nbsp;: <span id="__hs-tl-fps">—</span>&nbsp;<span class="__hs-tl-dot">●</span></div>
        <div>frame per em : <span id="__hs-tl-fpe">—</span>&nbsp;<span class="__hs-tl-dot">●</span></div>
      </div>
      <input type="range" id="__hs-range" style="display:none" min="0" max="0" value="0" step="1">
    `;
    document.body.appendChild(tl);

    // Minimize toggle lives outside the timeline panel, like the #__hs-tab/#__hs-panel pair.
    const tlMin = document.createElement('button');
    tlMin.id = '__hs-tl-min';
    tlMin.title = 'Minimize timeline';
    tlMin.textContent = '▼';
    document.body.appendChild(tlMin);

    // Minimize timeline to just its playback controls
    function setTlCollapsed(collapsed) {
      tl.classList.toggle('collapsed', collapsed);
      tlMin.textContent = collapsed ? '▲' : '▼';
      tlMin.title = collapsed ? 'Restore timeline' : 'Minimize timeline';
      document.documentElement.style.setProperty('--hs-tl-h', collapsed ? '56px' : '160px');
      // Labels were skipped while hidden (0-width) — recompute now they're visible.
      if (!collapsed) requestAnimationFrame(() => adjustLabelOverlaps(el('tl-labels')));
    }
    tlMin.addEventListener('click', () => {
      setTlCollapsed(!tl.classList.contains('collapsed'));
      saveUiState();
    });
    if (S._tlMinSaved) setTlCollapsed(true);

    // Timeline bar: click + drag to seek
    const tlBar = el('tl-bar');
    let tlScrubbing = false;
    function tlSeek(clientX) {
      const rect = tlBar.getBoundingClientRect();
      const t = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      seekFrame(Math.round(t * Math.max(0, S.frameCount - 1)));
    }
    tlBar.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      tlScrubbing = true; S.scrubbing = true;
      tlSeek(e.clientX); e.preventDefault();
    });
    document.addEventListener('mousemove', e => { if (tlScrubbing) tlSeek(e.clientX); });
    document.addEventListener('mouseup',   () => { if (tlScrubbing) { tlScrubbing = false; S.scrubbing = false; } });

    // Playback controls
    el('playpause').addEventListener('click', () => {
      if (!S.entry) return;
      S.animEverPlayed = true;
      const paused  = S.entry.viewer._animPaused;
      const reverse = S.entry.api.animation?.direction === -1;
      if (!paused && !reverse) {
        S.entry.api.setAnimationPaused(true);
      } else {
        if (S.entry.api.animation) S.entry.api.animation.direction = 1;
        S.entry.api.setAnimationPaused(false);
      }
      updatePlayState();
    });
    el('rev').addEventListener('click', () => {
      if (!S.entry) return;
      S.animEverPlayed = true;
      const paused  = S.entry.viewer._animPaused;
      const reverse = S.entry.api.animation?.direction === -1;
      if (!paused && reverse) {
        S.entry.api.setAnimationPaused(true);
      } else {
        if (S.entry.api.animation) S.entry.api.animation.direction = -1;
        S.entry.api.setAnimationPaused(false);
      }
      updatePlayState();
    });
    el('tl-tostart').addEventListener('click', () => {
      if (!S.entry) return;
      seekFrame(0); S.entry.api.setAnimationPaused(true); updatePlayState();
    });
    el('tl-toend').addEventListener('click', () => {
      if (!S.entry || !S.frameCount) return;
      seekFrame(S.frameCount - 1); S.entry.api.setAnimationPaused(true); updatePlayState();
    });

    _apiReady = apiCheck();
    _apiReady.then(() => {
      if (S.apiOnline) loadPageState().then(() => { renderScenes(); renderMasks(); });
    });

    // Start rAF loop for smooth scene card progress bars
    requestAnimationFrame(rafBars);

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
