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
| `holosplat/index.html` | Art-direction editor UI (served at `/holosplat/`, never deployed) |
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

## hs-config.json (managed by the editor)

```json
{
  "version": 1,
  "scene": "scenes/scene.spz",
  "animation": "scenes/anim.json",
  "acts": [
    { "id": "intro",   "type": "act",       "from": "intro",          "to": "pingpong-start", "height": 200 },
    { "id": "loop",    "type": "pingpong",   "from": "pingpong-start", "to": "pingpong-end",   "height": 150 },
    { "id": "explore", "type": "freecamera", "from": "freecamera-start","to":"freecamera-end", "height": 120 },
    { "id": "outro",   "type": "hold",       "frame": "final-marker",                          "height": 100 }
  ]
}
```

`from`/`to`/`frame` reference Blender timeline marker names exported in the animation JSON.

## Scene format preference

Always prefer `.spz` over `.ply` or `.splat` when recommending or generating scene file paths, configs, or examples. SPZ is the primary format — smaller files, faster loads. Use `.ply` only when the source toolchain cannot export SPZ.

## Editor

The `/holosplat` editor is a standalone HTML page that:
- Is served locally but **excluded from all deployments** (`.vercelignore`)
- Reads/writes `hs-config.json` via the `/hs-api` routes
- Shows the Blender marker list, lets you assign markers to scroll acts, drag to set heights

Never include `holosplat/` in production builds or deploy it.
