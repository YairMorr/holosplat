"""
HoloSplat — Asset Export Script
──────────────────────────────────────
Exports a product/asset's variant data and button-triggered clip animations
from a dedicated asset .blend file's own timeline (e.g. the headphones rig,
in a file separate from the main camera/scene). Output is loaded at runtime
via HoloSplat.player({ clips: '...' }) or player.api.loadClips(url) —
completely independent of the main camera/scene animation exported by
export_holosplat.py (no camera required in this file at all).

HOW TO USE
──────────
1. Open the Scripting workspace in this asset's .blend file.
2. Open this file (or paste it into a new text block).
3. Edit the CONFIG section below if needed — including LIB_DIR, if this repo
   doesn't live at the default path.
4. Press "Run Script".

This file only holds config + a small loader — the actual export logic
lives in _holosplat_asset_lib.py, loaded fresh off disk every run, so
editing that file takes effect immediately without reopening/repasting this
one. See its module docstring if you want to read the implementation.

VARIANT AXES (color/size/feature — mix-and-match, Figma-component style)
───────────────────────────────────────────────────────────────────────────
A part's splat children declare which variant values they belong to via
dot-separated "<axis>=<value>" segments after the part's base name:

    splat.cup-left.color=blue
    splat.cup-left.color=orange

Everything before the first "<axis>=<value>" segment is the part's base
splat name (here "cup-left" — loads cup-left.color=blue.spz etc.). A part
can use more than one axis (e.g. "cup-left.color=blue.size=big"); different
parts don't need to share the same axes.

This script scans every object in the "HoloSplat Parts" collection (not
just splat children — any auxiliary object, e.g. a mask volume, can carry
an "<axis>=<value>" tag too) and collects every axis and value it finds
into the exported "axes" field: { "color": ["blue","orange",...], ... }.
The editor reads this to build a default-value picker per axis. How
switching between values actually animates (which objects move, on what
markers) isn't handled by this script — see the player/editor docs once
that's implemented.

SETTING UP A CLIP (independent of variant axes — button-triggered motion)
───────────────────────────────────────────────────────────────────────────
Clips are for animations triggered by a page button that aren't a variant
swap (e.g. an add-on sliding into place) — orthogonal to the axes above.
Each clip needs:
  - A part Empty already in the "HoloSplat Parts" collection (same
    convention as export_holosplat.py — a child "splat.<name>" mesh tells
    HoloSplat which file to load).
  - Keyframes on that Empty across a frame range you pick (this file's
    timeline is independent, so use any range you like, e.g. 0..100) —
    frame 0 = neutral/start pose, some frame in the middle = the settled
    pose ("hold"), the last frame = back to neutral.
  - 3 timeline markers (Timeline/Dope Sheet, press M) sharing the clip's id
    "[productName]-[variant]" (e.g. "headphones-blue") — this same string
    should be a page button's element id; the player wires the click
    automatically:
        "headphones-blue.in"   — start of the "in" animation
        "headphones-blue"      — the "hold" frame
        "headphones-blue.out"  — end of the "out" animation

Any "HoloSplat Parts" Empty with at least one keyframe inside [.in, .out] is
included in that clip automatically — so multiple parts can move together
for one clip just by all having keyframes in that range.

PROPERTIES (named bags of float values — e.g. a hue/sat/val color tweak)
───────────────────────────────────────────────────────────────────────────
Put an Empty in a collection named "properties" (create it if it doesn't
exist), and name the Empty "property: <name>" — e.g. "property: color".
Add Custom Properties to that Empty (Object Properties panel → Custom
Properties → +), one per float value you want to export, e.g.:

    property: color   (Empty, in "properties")
      custom properties:  hue = 280.0
                           sat = 1.0
                           val = 1.0

Exports as data["properties"]["color"] = {"hue": 280.0, "sat": 1.0, "val": 1.0}.
Non-numeric custom properties are ignored. Add as many "property: <name>"
Empties as you like — this is generic key/value storage, not tied to any
specific feature; what reads a given name/key at runtime decides what it
means.

REFERENCE OBJECTS (ignored entirely)
─────────────────────────────────────
Same convention as export_holosplat.py: put reference-only objects (and all
their children) in a collection named "reference" and this script skips
them completely, regardless of name.

COORDINATE SYSTEM
─────────────────
Same as export_holosplat.py: Blender Z-up → HoloSplat Y-up. Set
GS_OBJECT_NAME if this asset's parts need to align with a GS-imported
mesh's local space (see export_holosplat.py's docstring for details).
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────

OUTPUT_PATH = "//"    # "//" = same folder as the .blend file
OUTPUT_NAME = None    # None = use the .blend file's own name

# Set this to the name of an imported Gaussian Splat mesh object if this
# asset's parts need to align with its local space. Leave None otherwise —
# see export_holosplat.py's docstring for the full explanation.
GS_OBJECT_NAME = None

# Set True to match flipY: true on the HoloSplat player.
FLIP_Y = False

# Folder holding _holosplat_asset_lib.py — the actual export logic, loaded
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
_lib_path = Path(LIB_DIR) / "_holosplat_asset_lib.py"
_lib = types.ModuleType("_holosplat_asset_lib")
exec(compile(_lib_path.read_text(encoding="utf-8"), str(_lib_path), "exec"), _lib.__dict__)

_lib.run(dict(
    OUTPUT_PATH=OUTPUT_PATH, OUTPUT_NAME=OUTPUT_NAME, GS_OBJECT_NAME=GS_OBJECT_NAME,
    FLIP_Y=FLIP_Y,
))
