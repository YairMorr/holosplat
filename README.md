# HoloSplat

WebGPU Gaussian Splat viewer with scroll-driven animation, an art-direction editor, and a Node.js server middleware.

---

## Table of Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Development server](#development-server)
- [Embedding a scene](#embedding-a-scene)
  - [Script tag — no build tools](#script-tag--no-build-tools)
  - [Data-attribute auto-init](#data-attribute-auto-init)
  - [ESM / bundler](#esm--bundler)
- [Scroll-driven animation](#scroll-driven-animation)
  - [HTML structure](#html-structure)
  - [Act types](#act-types)
  - [Captions](#captions)
- [Callouts](#callouts)
  - [Web page side](#web-page-side)
  - [Styling callouts](#styling-callouts)
- [Blender workflow](#blender-workflow)
  - [Export script](#export-script)
  - [CONFIG options](#config-options)
  - [Timeline markers](#timeline-markers)
  - [Adding callout anchors](#adding-callout-anchors)
  - [Coordinate systems and GS object](#coordinate-systems-and-gs-object)
- [Art-direction editor](#art-direction-editor)
  - [Starting the editor](#starting-the-editor)
  - [Editor workflow](#editor-workflow)
- [hs-config.json reference](#hs-configjson-reference)
- [Node.js server middleware](#nodejs-server-middleware)
- [Animated multi-part scenes](#animated-multi-part-scenes)
- [Building the library](#building-the-library)
- [JavaScript API reference](#javascript-api-reference)

---

## How it works

A HoloSplat scene has three parts:

1. **A splat file** (`.spz`, `.ply`, or `.splat`) — the 3D Gaussian Splat capture.
2. **An animation JSON** — exported from Blender; contains a per-frame camera path, FOV, timeline markers, and optional callout positions.
3. **`hs-config.json`** — maps Blender timeline markers to scroll acts, and sets the scroll height for each act. Written by the editor.

At runtime, `scrollScene()` maps the visitor's scroll position to a frame number. The player seeks the animation to that frame, sets the camera, and renders the splat.

```
scroll position
      │
      ▼
 hs-config.json   ← built in the editor
 (acts + heights)
      │
      ▼
  frame number
      │
      ▼
 animation.json   ← exported from Blender
 (eye + forward per frame)
      │
      ▼
  camera matrices
      │
      ▼
 WebGPU renderer  ← splat file (.spz / .ply / .splat)
```

---

## Quick start

```bash
# Install (or add to an existing project)
npm install holosplat

# Scaffold editor + dev server into the project root
npx holosplat init

# Start the Python dev server
python server.py          # → http://localhost:8080

# In a second terminal, watch-build the library if you're editing src/
node build.js --watch
```

After `init`, open `http://localhost:8080/holosplat/` for the art-direction editor.

---

## Development server

`server.py` is a zero-dependency Python 3 HTTP server that also serves the `/hs-api` routes the editor needs.

```bash
python server.py          # default port 8080
python server.py 3000     # custom port
```

It serves:

| Path | What |
|------|------|
| `/` | Project root (static files) |
| `/holosplat/` | Art-direction editor UI |
| `/scenes/` | Scene and animation files |
| `/hs-api/ls` | Lists loadable files for the editor |
| `/hs-api/file?path=…` | Read / write files (GET / PUT) |

The `scenes/` folder is created automatically if it doesn't exist. Put your `.spz`, `.ply`, `.splat`, and `.json` files there.

**WebGPU requires a secure context** — `http://localhost` is fine. `file://` URLs are not.

---

## Embedding a scene

### Script tag — no build tools

```html
<script src="holosplat.iife.js"></script>

<div id="viewer" style="width:100%; height:500px"></div>

<script>
  HoloSplat.player('#viewer', {
    src: 'https://cdn.example.com/scene.spz',
  });
</script>
```

### Data-attribute auto-init

Place the `<script>` tag anywhere on the page. Any `data-holosplat` element initialises automatically when the DOM is ready — no JavaScript required.

```html
<script src="holosplat.iife.js"></script>

<div
  data-holosplat="https://cdn.example.com/scene.spz"
  data-holosplat-anim="https://cdn.example.com/anim.json"
  style="width:100%; height:500px">
</div>
```

### ESM / bundler

```js
import { player, scrollScene } from 'holosplat';

const p = player('#viewer', {
  src: '/scenes/scene.spz',
  animation: '/scenes/anim.json',
});
```

### `player()` options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `src` | string | — | URL of `.spz`, `.ply`, or `.splat` file |
| `animation` | string | — | URL of animation JSON |
| `background` | string \| number[] | `'transparent'` | `'#rrggbb'`, `'transparent'`, or `[r,g,b,a]` (0–1) |
| `fov` | number | 60 | Vertical field of view, degrees |
| `near` | number | 0.1 | Near clip plane |
| `far` | number | 2000 | Far clip plane |
| `splatScale` | number | 1 | Global splat size multiplier |
| `autoRotate` | boolean | false | Slow continuous orbit when idle |
| `flipY` | boolean | false | 180° X-axis flip (for COLMAP/OpenCV captured scenes) |
| `onLoad` | function | — | Called when scene is ready |
| `onProgress` | function(0..1) | — | Called during fetch |
| `onError` | function(Error) | — | Called on any error |

### `player()` API

```js
const p = player('#viewer', { src: '…' });

p.load(url)              // swap scene
p.loadAnim(url)          // swap animation
p.destroy()              // stop + remove all DOM
p.setBackground(bg)
p.setSplatScale(s)
p.setAutoRotate(bool)
p.setFlipY(bool)
p.setAnimationPaused(bool)
p.setCameraFree(bool)    // let user orbit while animation is attached
p.resetCamera()          // re-fit camera to scene bounds, reset angle
p.focusCamera()          // re-fit camera to scene bounds, keep angle

p.camera                 // OrbitCamera instance
p.animation              // Animation instance (or null)
p.callout('id')          // returns the HTMLElement for a callout card
```

---

## Scroll-driven animation

### HTML structure

```html
<div class="hs-scene">

  <!-- The player mounts here -->
  <div class="hs-stage"
       data-holosplat="/scenes/scene.spz"
       data-holosplat-anim="/scenes/anim.json">
  </div>

  <!-- Scrollable track — one act per section of the page -->
  <div class="hs-track">

    <div class="hs-act"
         data-from="intro"
         data-to="desk_reveal"
         style="height: 300vh">
      <div class="hs-caption" data-at="0.15">Here is the desk</div>
    </div>

    <div class="hs-hold"
         data-frame="desk_reveal"
         style="height: 120vh">
      <div class="hs-caption">Notice the details</div>
    </div>

    <div class="hs-act"
         data-from="pingpong-start"
         data-to="pingpong-end"
         style="height: 200vh">
    </div>

    <div class="hs-act"
         data-from="freecamera-start"
         data-to="freecamera-end"
         style="height: 150vh">
    </div>

  </div>
</div>

<script src="holosplat.iife.js"></script>
```

`data-from`, `data-to`, and `data-frame` accept either a **Blender marker name** (from the exported JSON) or a raw **frame number**.

### Act types

There are four act types, selected by the element class and marker names:

---

#### `hs-act` — Scroll-driven playback

```html
<div class="hs-act"
     data-from="intro"
     data-to="desk_reveal"
     style="height: 300vh">
</div>
```

Plays the animation linearly from `data-from` to `data-to` as the user scrolls through the act's height. The camera follows the baked path exactly.

- `data-from` → start frame / marker name
- `data-to` → end frame / marker name
- `data-loop="3"` → repeat the range N times within the scroll distance (omit or `"1"` for no repeat)
- If `data-from` > `data-to` the range plays in reverse.

---

#### `hs-hold` — Freeze at a frame

```html
<div class="hs-hold"
     data-frame="desk_reveal"
     style="height: 120vh">
</div>
```

Holds the animation frozen at a single frame while the visitor reads. The camera does not move. Use this after an `hs-act` to give reading time at a key moment.

- `data-frame` → frame number or marker name to hold at

---

#### `hs-act` with pingpong markers — Autonomous loop

```html
<div class="hs-act"
     data-from="pingpong-start"
     data-to="pingpong-end"
     style="height: 200vh">
</div>
```

When both markers are named `*-start` / `*-end` (or any pair you configure in the editor as type **pingpong**), the act plays the frame range back and forth autonomously while the visitor is scrolled into it. The loop speed is independent of scroll position — the visitor lingers here and the scene plays by itself.

Transition in/out is smooth: the animation completes its current direction before handing off to the next act.

---

#### `hs-act` with freecamera markers — User orbit

```html
<div class="hs-act"
     data-from="freecamera-start"
     data-to="freecamera-end"
     style="height: 150vh">
</div>
```

Within this act's scroll range, the animation camera is released and the user can orbit, pan, and zoom freely (mouse drag / touch). The frame range still determines the transition frames used to enter and exit the act smoothly, but the camera is not driven by them while the user is inside.

---

### Captions

Inside any act or hold, add `.hs-caption` elements. They fade in when the scroll progress through that act reaches `data-at` (0–1, default 0).

```html
<div class="hs-act" data-from="intro" data-to="reveal" style="height:300vh">
  <div class="hs-caption" data-at="0.1">Opening line</div>
  <div class="hs-caption" data-at="0.6">Second beat</div>
</div>
```

Style `.hs-caption` and `.hs-caption--hidden` (added when not yet visible) yourself. Example:

```css
.hs-caption {
  position: absolute;
  bottom: 10vh;
  left: 50%;
  transform: translateX(-50%);
  color: white;
  font-size: 1.4rem;
  transition: opacity 0.4s;
}
.hs-caption--hidden { opacity: 0; pointer-events: none; }
```

---

## Callouts

Callouts are annotated world-space points anchored to 3D positions in the scene. The player projects each point to screen coordinates every frame, draws a dot and a connecting line to a card element, and hides the card when the point goes off-screen.

### Web page side

Add `.hs-callout` elements **inside the player container** (the element you passed to `player()`). Give each one a `data-id` matching the callout id exported from Blender.

```html
<div id="viewer" style="width:100%; height:600px">

  <div class="hs-callout" data-id="keyboard"
       data-offset-x="90" data-offset-y="-35">
    <h3>Mechanical keyboard</h3>
    <p>Cherry MX Browns, 65% layout</p>
  </div>

  <div class="hs-callout" data-id="screen"
       data-offset-x="-120" data-offset-y="20">
    <h3>Monitor</h3>
    <p>4K, 144 Hz</p>
  </div>

</div>

<script src="holosplat.iife.js"></script>
<script>
  HoloSplat.player('#viewer', {
    src: '/scenes/desk.spz',
    animation: '/scenes/desk.json',
  });
</script>
```

| Attribute | Description |
|-----------|-------------|
| `data-id` | Must match the callout id from Blender (the part after `hs.`) |
| `data-offset-x` | Horizontal offset in px from the dot to the card anchor point |
| `data-offset-y` | Vertical offset in px |

The player adds `.hs-callout--hidden` when the 3D point is behind the camera or off-screen.

### Styling callouts

The player injects minimal structural CSS. You provide the visual design:

```css
/* Card */
.hs-callout {
  position: absolute;           /* required — player positions it */
  background: rgba(0,0,0,0.7);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 8px;
  padding: 12px 16px;
  color: #fff;
  font-size: 0.85rem;
  pointer-events: auto;
  transition: opacity 0.2s;
}
.hs-callout--hidden { opacity: 0; pointer-events: none; }

/* Dot and line (SVG drawn by the player) */
.hs-lines circle { fill: #fff; }
.hs-lines line   { stroke: rgba(255,255,255,0.5); stroke-width: 1px; }
```

Access a callout element programmatically with `p.callout('keyboard')`.

---

## Blender workflow

### Export script

The script lives at `blender/export_holosplat.py`. It exports:
- The active camera's position and forward direction, one entry per frame
- The camera's field of view (always vertical/fovY)
- Near and far clip distances
- All timeline markers within the export range
- All callout anchor positions

**Steps:**

1. Open Blender and load your `.blend` file.
2. Go to the **Scripting** workspace.
3. Click **Open** and select `export_holosplat.py` (or paste it into a new text block).
4. Edit the `CONFIG` section at the top of the script as needed (see below).
5. Click **Run Script** (or press Alt+P with the cursor in the text editor).

The JSON file is saved next to your `.blend` file by default. Copy or symlink it into your project's `scenes/` folder.

### CONFIG options

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_PATH` | `"//"` | Output folder. `"//"` = same folder as the `.blend` file |
| `OUTPUT_NAME` | `None` | Output filename without `.json`. `None` = use the `.blend` filename |
| `CAMERA_NAME` | `None` | Name of the camera object. `None` = use the scene's active camera |
| `FRAME_START` | `0` | First frame to export. Defaults to `0` (not Blender's scene start) so that markers at frame 0 are included |
| `FRAME_END` | `None` | Last frame. `None` = use `scene.frame_end` |
| `GS_OBJECT_NAME` | `None` | Name of the imported Gaussian Splat mesh object. See [below](#coordinate-systems-and-gs-object) |
| `FLIP_Y` | `False` | Set `True` if loading the splat with `flipY: true` in the player |

### Timeline markers

Markers become the named reference points you use in `data-from`, `data-to`, and `data-frame` attributes on your scroll acts, and also control runtime camera behaviour when they carry an `hs-` prefix.

**How to add a marker in Blender:**

1. In Blender's **Timeline** or **Dopesheet**, scrub to the frame you want to mark.
2. Press **M** to place a marker.
3. With the marker selected, press **F2** (or double-click) to rename it.

After export the markers appear in the JSON:

```json
{
  "markers": {
    "intro":           0,
    "desk_reveal":    72,
    "pingpong-start": 90,
    "pingpong-end":  150,
    "hs-free":       155,
    "hs-locked":     220,
    "outro":         240
  }
}
```

Frame numbers are **0-based relative to `FRAME_START`**.

---

#### Scroll-act markers (any name you choose)

These are plain markers with no special naming rules. You reference them in `data-from`, `data-to`, `data-frame`, and in `hs-config.json`.

| Typical name | Convention |
|---|---|
| `intro` | First frame of the opening move |
| `desk_reveal` | Key moment — use as a hold target |
| `pingpong-start` / `pingpong-end` | Loop range for a pingpong act |
| `freecam-start` / `freecam-end` | Entry/exit frames for a freecamera act |
| `outro` | Final camera position |

Any name works — the names above are only a convention. Each scroll act needs a `from` + `to` pair (or a single `frame` for a hold).

---

#### `hs-*` camera control markers

Markers whose names begin with `hs-` are intercepted by the player at runtime and switch the camera between animation-driven and user-controlled modes. The **most recently passed** `hs-*` marker determines the current mode; earlier ones are superseded.

These markers have no effect when `scrollScene()` is driving the camera — they apply only during self-playing (non-scroll) animation.

| Marker | Effect |
|--------|--------|
| `hs-free` | Releases the camera to full free orbit. The user can orbit, zoom, and (unless a `focal-point` empty exists) pan. Camera snaps to the animation eye/target at the moment of transition. |
| `hs-locked` | Returns the camera to animation-driven mode. Camera snaps back to the baked path. |
| `hs-h{deg}` | Free orbit restricted to ±*deg* ° horizontally from the entry angle. Example: `hs-h45` allows 90 ° of horizontal sweep. |
| `hs-v{deg}` | Free orbit restricted to ±*deg* ° vertically from the entry angle. Example: `hs-v20` prevents the user from looking too far up or down. |
| `hs-h{deg}-v{deg}` | Both restrictions combined. Example: `hs-h30-v15` gives a tight look-around window. |

**Focal-point orbit anchor** — place a Blender Empty named `focal-point` (or `hs-focal-point`) anywhere in your scene. When free-camera mode is active, the orbit target locks to that world position instead of the animation look-at point, and panning is disabled. This keeps the object centred in frame while the user spins around it.

**Typical placement:**

```
frame  0  ──── animation plays, camera locked to baked path
       :
frame 155  ── hs-free   → user can orbit freely around focal-point
       :
frame 220  ── hs-locked → camera returns to baked path
       :
frame 240  ── outro
```

### Adding part Empties

See [Animated multi-part scenes → Blender setup](#blender-setup) for the full workflow. Quick summary: put each part's Empty in the **`HoloSplat Parts`** collection (or prefix names with `hs-part.`), animate them, then run the export script.

### Adding callout anchors

A callout anchor is a world-space point in Blender that the player will project to screen coordinates on every frame. Create one for each annotation you want.

**Method 1 — name prefix `hs.`** (recommended for a few callouts):

1. In the viewport, press **Shift+A → Empty → Plain Axes**.
2. Move it to the exact point you want to annotate (e.g. the corner of a keyboard).
3. Rename it to `hs.keyboard` — the part after `hs.` becomes the callout `id`.

**Method 2 — collection** (for many callouts):

1. Create a collection named exactly **`HoloSplat Callouts`**.
2. Add your Empty objects to it. The object name becomes the `id` directly (no prefix needed).

The exported positions are in HoloSplat's coordinate space and match the splat file exactly (assuming `GS_OBJECT_NAME` is set correctly — see below).

### Coordinate systems and GS object

Blender uses a **Z-up** coordinate system (X right, Y forward, Z up).  
HoloSplat uses **Y-up** (X right, Y up, Z back).

If you imported the Gaussian Splat using the **3D Gaussian Splatting** addon (or any addon that applies a rotation/scale transform to the object), the imported mesh object has a world-space transform baked in. To get camera and callout positions that align with the actual vertices in the `.spz`/`.ply` file, set `GS_OBJECT_NAME` to the name of that mesh object. The script will then work in the object's local space, which is the same coordinate space the splat file uses.

```python
GS_OBJECT_NAME = "desk_2"   # name of the imported GS mesh in your scene
```

If your scene has no per-object transform on the GS mesh (or you placed the camera manually), leave `GS_OBJECT_NAME = None`. The script will apply the default Blender → HoloSplat axis conversion: `hs_x = bl_x`, `hs_y = bl_z`, `hs_z = -bl_y`.

If you load the splat with `flipY: true` in the player, also set `FLIP_Y = True` in the script so the camera and callout coordinates are rotated to match.

---

## Art-direction editor

The editor is a local web app served at `/holosplat/`. It is **never deployed** — it is excluded from all production builds.

### Starting the editor

```bash
python server.py
# then open: http://localhost:8080/holosplat/
```

The editor needs the `/hs-api` routes to read and write files. `server.py` provides them. If you're using a Node.js server instead, see [Node.js server middleware](#nodejs-server-middleware).

### Editor workflow

1. **Load files** — Enter paths to your scene file and animation JSON in the Files panel (relative to the project root, e.g. `scenes/desk.spz`). Click the reload arrows.

2. **Preview** — The scene renders in the right panel. Use the scrubber at the bottom to seek through frames. The marker list on the left shows all markers exported from Blender.

3. **Build the timeline** — Use the `+ Act`, `+ Hold`, `+ Pingpong`, `+ Freecam` buttons to add acts.

4. **Assign markers** — Each act row has dropdowns for the from/to/frame markers. Select the Blender markers you want each act to span.

5. **Set heights** — Drag the height field (the number at the right of each act row) to set how many viewport heights of scroll that act takes.

6. **Save** — Click **Save** to write `hs-config.json` back to the server. The dot indicator in the title turns on when there are unsaved changes.

When the API server is offline (no `server.py` running), the Save button becomes **Export** and downloads a `hs-config.json` file instead.

---

## hs-config.json reference

```json
{
  "version": 1,
  "scene": "scenes/scene.spz",
  "animation": "scenes/anim.json",
  "acts": [
    {
      "id": "intro",
      "type": "act",
      "from": "intro",
      "to": "desk_reveal",
      "height": 300
    },
    {
      "id": "hold1",
      "type": "hold",
      "frame": "desk_reveal",
      "height": 120
    },
    {
      "id": "loop",
      "type": "pingpong",
      "from": "pingpong-start",
      "to": "pingpong-end",
      "height": 200
    },
    {
      "id": "explore",
      "type": "freecamera",
      "from": "freecam-start",
      "to": "freecam-end",
      "height": 150
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `type` | `"act"` \| `"hold"` \| `"pingpong"` \| `"freecamera"` |
| `from` / `to` | Blender marker name or frame number (used by `act`, `pingpong`, `freecamera`) |
| `frame` | Blender marker name or frame number (used by `hold`) |
| `height` | Scroll distance in `vh` units |

---

## Node.js server middleware

Install the package, then mount the middleware in development only.

```js
import { createHsApiHandler } from 'holosplat/server';
```

**Express:**
```js
if (process.env.NODE_ENV !== 'production') {
  app.use('/hs-api', createHsApiHandler());
}
```

**Vite (`vite.config.js`):**
```js
import { createHsApiHandler } from 'holosplat/server';

export default {
  server: {
    middlewares: [createHsApiHandler()]   // mounts at /hs-api
  }
}
```

**Next.js (`pages/api/hs-api/[...route].js`):**
```js
import { createHsApiHandler } from 'holosplat/server';
const handler = createHsApiHandler();

export default function hsApi(req, res) {
  req.url = '/' + (req.query.route ?? []).join('/')
    + (req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '');
  handler(req, res);
}

export const config = { api: { bodyParser: false } };
```

`createHsApiHandler(root?)` accepts an optional root directory (defaults to `process.cwd()`). Path traversal outside the root is rejected.

---

## Animated multi-part scenes

A multi-part scene loads several Gaussian Splat files as independent rigid bodies, each driven by an animated Empty in Blender. This lets you animate articulated objects — a folding headphone, a robot arm, a door opening — while keeping full splat quality on every part.

### How it works

1. Capture each part of the object as a separate `.spz`/`.ply` file (headband, hinge, cup…).
2. In Blender, import each part and parent it to an Empty with a matching name.
3. Animate the Empties on the Blender timeline.
4. Run the export script — it records each Empty's position + rotation quaternion per frame.
5. At runtime, HoloSplat merges all splat files into one GPU buffer, tags each splat with its part index, and applies the per-frame transforms on the GPU. Depth sorting works across all parts automatically.

### Blender setup

1. Import each splat part into your scene. Set `GS_OBJECT_NAME` in the export script to the root object (or the shared parent) so coordinate spaces align.

2. Create one Empty per part (**Add → Empty → Plain Axes**). Name each Empty to match what you'll use in `loadParts()`.

3. Parent each splat mesh to its Empty: select the mesh, Shift-click the Empty, press **Ctrl+P → Object**.

4. Put all part Empties into a collection named **`HoloSplat Parts`** (or prefix names with `hs-part.`):
   ```
   Collection: HoloSplat Parts
   ├── headband       → id "headband"
   ├── hinge_left     → id "hinge_left"
   ├── hinge_right    → id "hinge_right"
   └── cup_left       → id "cup_left"
   ```

5. Animate the Empties on the timeline. Add **timeline markers** (M key) for scroll acts as usual.

6. Run `export_holosplat.py`. The output JSON will include an `objects` array alongside the camera data.

> **Apply scale** (Ctrl+A → Scale) on all Empties before exporting. Uneven scale on an Empty is not exported and will cause misalignment at runtime.

### JavaScript

Pass a `parts` map instead of (or in addition to) `src`:

```js
// Script tag
HoloSplat.player('#viewer', {
  parts: {
    headband:    '/scenes/headband.spz',
    hinge_left:  '/scenes/hinge_left.spz',
    hinge_right: '/scenes/hinge_right.spz',
    cup_left:    '/scenes/cup_left.spz',
    cup_right:   '/scenes/cup_right.spz',
  },
  animation: '/scenes/headphones.json',
});
```

```js
// ESM
import { player } from 'holosplat';

const p = player('#viewer', {
  parts: {
    headband:   '/scenes/headband.spz',
    hinge_left: '/scenes/hinge_left.spz',
  },
  animation: '/scenes/headphones.json',
});

// Swap parts at runtime
p.loadParts({ headband: '/scenes/headband_v2.spz', ... });
```

The keys in `parts` must match the Empty names (or `hs-part.` suffixes) used during export.

### Animation JSON format (v2)

The export script writes version `2` when parts are present. The `objects` array is optional — v1 files (no `objects`) load normally as single-part scenes.

```json
{
  "version": 2,
  "fps": 24,
  "frameCount": 120,
  "fov": 45.0,
  "frames": [ ... ],
  "objects": [
    {
      "id": "headband",
      "frames": [
        0.0, 1.2, 0.0,  0.0, 0.0, 0.0, 1.0,
        0.0, 1.2, 0.0,  0.0, 0.0, 0.0, 1.0,
        ...
      ]
    },
    {
      "id": "hinge_left",
      "frames": [ ... ]
    }
  ],
  "markers": { "open": 0, "closed": 60 }
}
```

Each entry in `objects.frames` is 7 floats per frame:
```
px py pz   — position in splat-file coordinate space
qx qy qz qw — rotation quaternion (XYZW)
```

---

## Building the library

```bash
node build.js             # builds dist/holosplat.esm.js and dist/holosplat.iife.js
node build.js --watch     # rebuild on every change
```

Source is in `src/`. Entry point is `src/index.js`. Bundle target is browser (esbuild).  
`src/server.js` is Node.js-only and is **not bundled** — it is exported as the `holosplat/server` subpath.

---

## JavaScript API reference

### `create(options)` — low-level viewer

```js
import { create } from 'holosplat';

const viewer = await create({
  canvas:     '#myCanvas',   // CSS selector or HTMLCanvasElement
  src:        '/scenes/scene.spz',
  background: '#111111',
  splatScale: 1.0,
  onLoad:     () => console.log('ready'),
  onProgress: p => console.log(p * 100 + '%'),
  onError:    err => console.error(err),
});

viewer.setBackground('#222');
viewer.setSplatScale(1.5);
viewer.resetCamera();
viewer.focusCamera();      // same as resetCamera but preserves camera angle
viewer.destroy();
```

### `Viewer` class

```js
import { Viewer } from 'holosplat';

const viewer = new Viewer({ canvas: '#c', background: '#000' });
await viewer.init();
await viewer.load('/scenes/scene.spz');
viewer.start();

// Animation
await viewer.loadAnimationUrl('/scenes/anim.json');
viewer.setAnimationPaused(true);
viewer.setCameraFree(true);

// Frame callback — called every render tick
viewer.onFrame = (viewMatrix, projMatrix, width, height) => { … };

// Project world-space points to screen
const hits = viewer.projectCallouts([{ id: 'dot', pos: [1, 2, 3] }]);
// → [{ id: 'dot', visible: true, x: 540, y: 320 }]
```

### `Animation` class

```js
import { Animation, loadAnimation } from 'holosplat';

const anim = await loadAnimation('/scenes/anim.json');

anim.fps          // number
anim.frameCount   // number
anim.markers      // { name: frameNumber, … }
anim.callouts     // [{ id, pos: [x,y,z] }]

anim.seekFrame(42);
anim.tick(deltaSeconds);      // advance playback
anim.getCameraFrame();        // → { eye: [x,y,z], target: [x,y,z] }
```

### `scrollScene(sceneEl, playerInstance, opts?)`

```js
import { player, scrollScene } from 'holosplat';

const p = player('.hs-stage', { src: '…', animation: '…' });
const sc = scrollScene(document.querySelector('.hs-scene'), p);

sc.rebuild();   // re-read DOM after programmatic changes
sc.destroy();   // remove scroll listeners
```

### `compressToSpz(data, count, opts?)` → `Promise<ArrayBuffer>`

Converts canonical Gaussian data (Float32Array, 16 floats/splat) to a gzip-compressed `.spz` file.

```js
import { compressToSpz } from 'holosplat';

const buffer = await compressToSpz(gaussianData, numSplats);
const blob = new Blob([buffer], { type: 'application/octet-stream' });
```
