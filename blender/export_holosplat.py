"""
HoloSplat — Blender Camera + Parts Export Script
─────────────────────────────────────────
Exports the active camera animation and callout objects to a JSON file
readable by the HoloSplat player (HoloSplat.player / data-holosplat-anim).

HOW TO USE
──────────
1. Open the Scripting workspace in Blender.
2. Open this file (or paste it into a new text block).
3. Edit the CONFIG section below if needed — including LIB_DIR, if this repo
   doesn't live at the default path.
4. Press "Run Script".

This file only holds config + a small loader — the actual export logic
lives in _holosplat_export_lib.py, loaded fresh off disk every run, so
editing that file takes effect immediately without reopening/repasting this
one. See its module docstring if you want to read the implementation.

The output .json file is saved next to your .blend file by default.

ANIMATED PARTS
──────────────
Place an Empty in the "HoloSplat Parts" collection for each animated part —
its name is just an identifier and is never parsed, so name it anything
(e.g. "ctrl.fork-left"). What splat file gets loaded is determined entirely
by a child mesh object parented to that Empty, named "splat.<splat-name>":

    ctrl.fork-left  (Empty, in "HoloSplat Parts")
      └─ splat.fork-left  (mesh, parented to the Empty)  →  loads fork-left.spz

Color/material variants are declared the same way, as sibling children with
an extra suffix segment:

    splat.fork-left          →  primary file: fork-left.spz
    splat.fork-left.orange   →  variant "orange": fork-left.orange.spz

A part with no "splat.<name>" child is skipped (with a console warning) — it
has nothing telling HoloSplat which file to load. Objects outside the
"HoloSplat Parts" collection are always ignored, even if similarly named.

Animate the Empties. The script exports their world transform (position +
rotation) per frame. At runtime HoloSplat applies each Empty's transform to
the corresponding splat model (loaded separately by the web player — the
Blender mesh itself is just a placeholder/preview, never used at runtime).

Important: apply any scale on the Empties (Ctrl+A → Scale) before exporting,
and ensure GS_OBJECT_NAME is set so coordinate spaces align correctly.

REFERENCE OBJECTS (ignored entirely)
─────────────────────────────────────
If you're loading objects purely as visual references (e.g. the actual
physical product, for lining things up while animating) and they shouldn't
be touched by this script at all, put them — and all their children — in a
collection named "reference" (anywhere in the scene; nested collections are
fine, and so is having more than one "reference" collection — Blender
auto-renames duplicates "reference.001" etc., and those are matched too).
Everything inside it is skipped by every part of this script, regardless of
what it's named, even if it happens to collide with one of the naming
conventions below (e.g. a child mesh accidentally named "splat.x", or an
object accidentally named "focal-point"). Select the
reference object(s) and press M → New Collection (with "Select Hierarchy"
on) to move the whole hierarchy in at once.

CALLOUT OBJECTS
───────────────
To mark a 3D point as a callout anchor, add an Empty object (Add → Empty →
Plain Axes is recommended) and name it with the prefix "hs." followed by
the callout id:

    hs.keyboard   →  id "keyboard"
    hs.screen     →  id "screen"

Alternatively, put any objects into a collection named "HoloSplat Callouts"
(no prefix needed; the object name becomes the id directly).

In the web page, place a matching card div inside the player container:

    <div class="hs-callout hs-callout--right" data-id="keyboard"
         data-offset-x="90" data-offset-y="-35">
      <h3>Keyboard</h3>
      <p>Mechanical switches, 65% layout.</p>
    </div>

The player projects the Empty's position, draws a dot and a line, and
positions the card at (dot + offset). See examples/callouts.css for
styling defaults and directional variants.

CLIPS (product customization — color/size/add-on buttons)
───────────────────────────────────────────────────────────
Button-triggered, independent per-object animations (e.g. color/size/add-on
swaps) are exported separately, by export_holosplat_asset.py, from a
dedicated asset .blend file with its own timeline — not from this script.
See that file's docstring for setup. (Earlier versions of this tried
scoping clips to NLA action-local "pose markers", then to reserved negative
frames in this same file's timeline — neither panned out, the former
wasn't reachable/usable the way it needed to be and the latter hit a
Blender restriction on negative frames — hence the separate file.)

COORDINATE SYSTEM
─────────────────
Blender uses Z-up (X right, Y forward, Z up).
HoloSplat uses Y-up (X right, Y up, Z back).

If you imported your Gaussian Splat using the "3D Gaussian Splatting" addon
(or any addon that applies a world-space rotation/scale to the object), set
GS_OBJECT_NAME to the name of that object. The script will then transform the
camera into the object's local space, which matches the coordinate system of
the exported .spz/.ply file. Callout objects are transformed the same way.

If GS_OBJECT_NAME is None the script falls back to a simple Blender → HoloSplat
axis conversion (hs_x = bl_x, hs_y = bl_z, hs_z = -bl_y).
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────

OUTPUT_PATH  = "//"    # "//" = same folder as the .blend file
# Output filename (without .json). None = use the .blend file's own name.
OUTPUT_NAME  = None

# Leave None to use the scene's active camera
CAMERA_NAME  = None

# Frame range. FRAME_START defaults to 0 (not scene.frame_start which is
# usually 1 in Blender) so that markers placed at frame 0 are included.
FRAME_START  = 0
FRAME_END    = None

# Set this to the name of your imported Gaussian Splat mesh object, e.g. "desk_2".
# When set, the camera (and callouts) are transformed into the object's local space
# so that coordinates match the exported splat file exactly.
# Leave None if your scene has no per-object transform on the GS object.
GS_OBJECT_NAME = None

# Set True when loading the .spz/.ply in the HoloSplat player with flipY: true.
# Applies the same 180° X-axis rotation to all exported camera positions and
# callout positions so the animation matches the flipped scene.
FLIP_Y = False

# Folder holding _holosplat_export_lib.py — the actual export logic, loaded
# fresh from disk every run. Only needs changing if this repo isn't checked
# out at this path.
LIB_DIR = r"D:\dev\holosplat\blender"

# ─────────────────────────────────────────────────────────────────────────────

import types
from pathlib import Path

# Manual compile+exec instead of importlib's spec_from_file_location/
# exec_module — that path goes through SourceFileLoader, which caches
# compiled bytecode in __pycache__ and can serve a stale version if the
# cache's staleness check ever misfires. This never touches disk for
# anything but reading the source, so it's always fresh.
_lib_path = Path(LIB_DIR) / "_holosplat_export_lib.py"
_lib = types.ModuleType("_holosplat_export_lib")
exec(compile(_lib_path.read_text(encoding="utf-8"), str(_lib_path), "exec"), _lib.__dict__)

_lib.run(dict(
    OUTPUT_PATH=OUTPUT_PATH, OUTPUT_NAME=OUTPUT_NAME, CAMERA_NAME=CAMERA_NAME,
    FRAME_START=FRAME_START, FRAME_END=FRAME_END, GS_OBJECT_NAME=GS_OBJECT_NAME,
    FLIP_Y=FLIP_Y,
))
