"""
HoloSplat — Blender Camera Export Script
─────────────────────────────────────────
Exports the active camera animation and callout objects to a JSON file
readable by the HoloSplat player (HoloSplat.player / data-holosplat-anim).

HOW TO USE
──────────
1. Open the Scripting workspace in Blender.
2. Open this file (or paste it into a new text block).
3. Edit the CONFIG section below if needed.
4. Press "Run Script".

The output .json file is saved next to your .blend file by default.

CALLOUT OBJECTS
───────────────
Any object you want to appear as a callout label in the web player should
either be placed in a collection named  "HoloSplat Callouts",  or be named
with the prefix  "hs."  (e.g. "hs.keyboard", "hs.screen").

The object's name (without the "hs." prefix) becomes the callout id.
In the web page, style it with:

    .hs-callout[data-id="keyboard"] { ... }

COORDINATE SYSTEM
─────────────────
Blender uses Z-up (X right, Y forward, Z up).
HoloSplat uses Y-up (X right, Y up, Z back).

This script converts automatically:
    hs_x = bl_x
    hs_y = bl_z
    hs_z = -bl_y
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────

OUTPUT_PATH  = "//"          # "//" = same folder as the .blend file
OUTPUT_NAME  = "scene_anim"  # output will be <OUTPUT_NAME>.json

# Leave None to use the scene's active camera
CAMERA_NAME  = None

# Leave None to use the scene's frame range
FRAME_START  = None
FRAME_END    = None

# ─────────────────────────────────────────────────────────────────────────────

import bpy, json, math
from pathlib import Path

scene = bpy.context.scene

# ── Resolve camera ────────────────────────────────────────────────────────────
cam_obj = scene.objects.get(CAMERA_NAME) if CAMERA_NAME else scene.camera
if cam_obj is None or cam_obj.type != 'CAMERA':
    raise RuntimeError(
        "HoloSplat export: no camera found. "
        "Set CAMERA_NAME or make sure the scene has an active camera."
    )

# ── Frame range ───────────────────────────────────────────────────────────────
f_start = FRAME_START if FRAME_START is not None else scene.frame_start
f_end   = FRAME_END   if FRAME_END   is not None else scene.frame_end
fps     = scene.render.fps

# ── FOV from camera focal length ─────────────────────────────────────────────
cam_data     = cam_obj.data
focal_length = cam_data.lens
sensor_h     = cam_data.sensor_height if cam_data.sensor_fit != 'HORIZONTAL' else cam_data.sensor_width
fov_deg      = round(2 * math.degrees(math.atan(sensor_h * 0.5 / focal_length)), 4)

# ── Coordinate conversion ─────────────────────────────────────────────────────

def bl_to_hs(v):
    """Blender (X right, Y fwd, Z up)  →  HoloSplat (X right, Y up, Z back)"""
    return [v.x, v.z, -v.y]

def normalize(v):
    l = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return [v[0]/l, v[1]/l, v[2]/l] if l > 1e-9 else [0, 0, -1]

# ── Collect callout objects ───────────────────────────────────────────────────
callout_objects = {}   # id → object (deduped)

# Method 1: "HoloSplat Callouts" collection
if "HoloSplat Callouts" in bpy.data.collections:
    for obj in bpy.data.collections["HoloSplat Callouts"].all_objects:
        callout_objects[obj.name] = obj

# Method 2: names starting with "hs."
for obj in scene.objects:
    if obj.name.startswith("hs."):
        callout_objects[obj.name] = obj

callouts = []
for name, obj in callout_objects.items():
    cid = name.removeprefix("hs.")
    # Use the object's world position at the current frame
    pos = bl_to_hs(obj.matrix_world.translation)
    callouts.append({"id": cid, "pos": [round(x, 6) for x in pos]})

print(f"HoloSplat: found {len(callouts)} callout(s): {[c['id'] for c in callouts]}")

# ── Export camera per frame ───────────────────────────────────────────────────
saved_frame = scene.frame_current
frames_flat = []

for f in range(f_start, f_end + 1):
    scene.frame_set(f)

    mw  = cam_obj.matrix_world
    pos = mw.translation

    # Camera looks in its local -Z direction.
    # matrix_world.col[2].xyz is the camera's Z axis in world space.
    fwd_bl = -mw.col[2].xyz  # negate → forward direction

    eye     = bl_to_hs(pos)
    forward = normalize(bl_to_hs(fwd_bl))

    frames_flat.extend([round(x, 6) for x in eye + forward])

scene.frame_set(saved_frame)   # restore original frame

frame_count = f_end - f_start + 1
print(f"HoloSplat: exported {frame_count} frames at {fps} fps  (duration {frame_count/fps:.2f}s)")

# ── Write JSON ────────────────────────────────────────────────────────────────
data = {
    "version"    : 1,
    "fps"        : fps,
    "frameCount" : frame_count,
    "fov"        : fov_deg,
    "frames"     : frames_flat,
    "callouts"   : callouts,
}

out_dir  = bpy.path.abspath(OUTPUT_PATH)
out_file = str(Path(out_dir) / f"{OUTPUT_NAME}.json")

with open(out_file, "w") as fp:
    json.dump(data, fp, separators=(",", ":"))

print(f"HoloSplat: saved → {out_file}")
