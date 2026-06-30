# HoloSplat

WebGPU Gaussian Splat viewer library with scroll-driven animation, an art-direction editor, and a Node.js middleware for the editor API.

## Build

```bash
node build.js          # builds dist/holosplat.esm.js and dist/holosplat.iife.js
node build.js --watch  # watch mode
```

Source lives in `src/`. Entry point is `src/index.js`. Bundle target is browser (esbuild).  
`src/server.js` is Node.js only — it is **not** bundled, exported as `holosplat/server`.

## Key source files

| File | Purpose |
|------|---------|
| `src/viewer.js` | Core `Viewer` class — WebGPU init, load, render loop |
| `src/player.js` | `player()` — embeddable wrapper with auto-init and callouts |
| `src/camera.js` | `OrbitCamera` — spherical orbit, lookAt/perspective matrices |
| `src/scroll-scene.js` | `scrollScene()` — maps `.hs-act`/`.hs-hold` DOM to animation frames |
| `src/animation.js` | `Animation` — parses Blender JSON, seekFrame, tick |
| `src/server.js` | `createHsApiHandler()` — Express/Vite/Next.js middleware for editor API |
| `holosplat/editor.js` | Desktop `?hs` overlay — art-direction editor (never deployed) |
| `holosplat/stats.js` | Mobile/touch `?hs` overlay — lightweight fps/splat-count/SH/scene readout |
| `bin/holosplat.cjs` | `npx holosplat init` CLI |
| `server.py` | Python dev server (adds `/hs-api` routes for the editor) |

## View matrix convention

Column-major `Float32Array(16)`, WebGPU convention. When multiplying a world-space point:
```
vx = view[0]*wx + view[4]*wy + view[8]*wz  + view[12]
vy = view[1]*wx + view[5]*wy + view[9]*wz  + view[13]
vz = view[2]*wx + view[6]*wy + view[10]*wz + view[14]
```

## Gaussian data layout

Flat `Float32Array`, 16 floats per splat:
```
[0..2]  x, y, z          position
[3]     (unused)
[4..6]  r, g, b           colour  (0..1, linear)
[7]     alpha             opacity (0..1)
[8..10] sx, sy, sz        scale   (world-space, not log)
[11]    (unused)
[12..15] qx, qy, qz, qw  rotation quaternion (normalised)
```

## Scene/scroll config (managed by the editor)

There is no separate config file — all config lives in the `player({...})` call in the page's own source (see `examples/headphones.html`). The editor patches this call in place via the `/hs-api/js-*` routes (regex-anchored on the `player(` call's closing `});`).

`scenes` is keyed by Blender timeline marker name and maps each marker to a scroll zone:

```js
player('#scene', {
  animation: '/scenes/anim.json',     // hs-anim
  scenes: {                           // hs-scenes
    'hero-section': { linkedId: 'hero-section', playback: 'auto', pingpong: true, blendOut: 46 },
    'feature-01':   { linkedId: 'feature-01' },
  },
  masks: { 'headphones': { feather: 0.18 } },  // hs-masks — mask volume name → feather
  sh: 3,                                        // hs-sh — global SH degree
});
```

`linkedId` is the id of the DOM element (the scroll container) that drives that scene. `setupScrollPlayback` itself just does `document.getElementById(cfg.linkedId)` — any id works at runtime — but **give every linkable scroll-zone element `class="hs-zone"` too**:

```html
<div id="hero-section" class="hs-zone" style="height:300vh">...</div>
```

The editor's "html element" picker (Scenes tab → linkedId dropdown) only lists `[id].hs-zone` elements — without that class, every id on the page (nav links, buttons, form fields) would show up as a candidate. Keep ids free-form/human-readable (matching Blender marker names, e.g. `hero-section`); `hs-zone` is purely the opt-in marker, it has no required naming pattern itself. See `src/player.js`'s `setupScrollPlayback` for the full set of recognised per-scene keys (`playback`, `pingpong`, `playOnce`, `blendIn`, `blendOut`).

## Scene format preference

Always prefer `.spz` over `.ply` or `.splat` when recommending or generating scene file paths, configs, or examples. SPZ is the primary format — smaller files, faster loads. Use `.ply` only when the source toolchain cannot export SPZ.

## Editor

The editor (`holosplat/editor.js`) is not a standalone page — it's an overlay injected into whichever page you open with `?hs` in the URL, regardless of whether that page has a `player()` call yet:
- Never deployed — only loads when `?hs` is present
- Reads/writes the page's own `player({...})` call via the `/hs-api/js-*` routes
- Three tabs: **Scenes** (per-marker scene cards), **Setup** (render/file/3D-scene settings), **Tools** (utility links + **Init page**)
- If no player is found on the page, the editor disables Scenes/Setup and parks on Tools — **Init page** writes a blank `player()` scaffold into the page's HTML and reloads to connect

### Resolving a page's URL to its source file

Every `/hs-api/*` route that reads or writes a page's `player({...})` call takes a `page` field — the browser's `location.pathname` (e.g. `/`, `/colors`). That's resolved to a real file by `_resolve_page()` (server.py) / `pagePath()` (src/server.js):

1. Check **`hs-pages.json`** at the project root — `{ "<urlPath>": "<file path>" }` — first.
2. Fall back to treating the URL path itself as a project-relative file path (`/` → `index.html`).

Plain static-HTML projects (server.py-served sites) never need `hs-pages.json` — the URL already is the filename. Framework projects (Next.js, etc.) do, since a URL has no file of the same name. When **Init page** can't find a literal file for the current URL (404), the editor shows a copy-paste prompt (`askClaudePrompt()` in editor.js) for an AI coding assistant to create the player component and register it in `hs-pages.json` — the editor doesn't attempt to generate or insert framework-specific component code itself.

Never include `holosplat/` in production builds or deploy it.
