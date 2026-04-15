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

If you imported your Gaussian Splat using the "3D Gaussian Splatting" addon
(or any addon that applies a world-space rotation/scale to the object), set
GS_OBJECT_NAME to the name of that object. The script will then transform the
camera into the object's local space, which matches the coordinate system of
the exported .spz/.ply file. Callout objects are transformed the same way.

If GS_OBJECT_NAME is None the script falls back to a simple Blender → HoloSplat
axis conversion (hs_x = bl_x, hs_y = bl_z, hs_z = -bl_y).
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────

OUTPUT_PATH  = "//"          # "//" = same folder as the .blend file
OUTPUT_NAME  = "scene_anim"  # output will be <OUTPUT_NAME>.json

# Leave None to use the scene's active camera
CAMERA_NAME  = None

# Leave None to use the scene's frame range
FRAME_START  = None
FRAME_END    = None

# Set this to the name of your imported Gaussian Splat mesh object, e.g. "desk_2".
# When set, the camera (and callouts) are transformed into the object's local space
# so that coordinates match the exported splat file exactly.
# Leave None if your scene has no per-object transform on the GS object.
GS_OBJECT_NAME = None

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

# ── Resolve GS object (optional) ─────────────────────────────────────────────
gs_obj = scene.objects.get(GS_OBJECT_NAME) if GS_OBJECT_NAME else None
gs_inv = gs_obj.matrix_world.inverted() if gs_obj else None

if gs_obj:
    print(f"HoloSplat: using GS object '{gs_obj.name}' — camera will be in its local space")
else:
    print("HoloSplat: no GS object set — using default Blender → Y-up axis conversion")

# ── Frame range ───────────────────────────────────────────────────────────────
f_start = FRAME_START if FRAME_START is not None else scene.frame_start
f_end   = FRAME_END   if FRAME_END   is not None else scene.frame_end
fps     = scene.render.fps

# ── FOV from camera focal length ─────────────────────────────────────────────
cam_data     = cam_obj.data
focal_length = cam_data.lens
sensor_h     = cam_data.sensor_height if cam_data.sensor_fit != 'HORIZONTAL' else cam_data.sensor_width
fov_deg      = round(2 * math.degrees(math.atan(sensor_h * 0.5 / focal_length)), 4)

# ── Coordinate conversion helpers ────────────────────────────────────────────

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
    world_pos = obj.matrix_world.translation

    if gs_inv:
        # Transform callout position into GS object's local space (= splat file space)
        local_pos = gs_inv @ world_pos
        pos = [local_pos.x, local_pos.y, local_pos.z]
    else:
        pos = bl_to_hs(world_pos)

    callouts.append({"id": cid, "pos": [round(x, 6) for x in pos]})

print(f"HoloSplat: found {len(callouts)} callout(s): {[c['id'] for c in callouts]}")

# ── Export camera per frame ───────────────────────────────────────────────────
saved_frame = scene.frame_current
frames_flat = []

for f in range(f_start, f_end + 1):
    scene.frame_set(f)

    mw = cam_obj.matrix_world

    if gs_inv:
        # Transform camera into GS object's local space (= splat file coordinate system)
        # This undoes the rotation/scale the import addon applied to the GS object,
        # so the exported positions match the actual vertex positions in the file.
        local_pos = gs_inv @ mw.translation
        fwd_world = -mw.col[2].xyz          # camera looks in its local -Z direction
        local_fwd = (gs_inv.to_3x3() @ fwd_world).normalized()
        eye     = [local_pos.x, local_pos.y, local_pos.z]
        forward = normalize([local_fwd.x, local_fwd.y, local_fwd.z])
    else:
        pos    = mw.translation
        fwd_bl = -mw.col[2].xyz
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
