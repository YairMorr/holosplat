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

# ── FOV from camera — vertical (fovY), matching WebGPU perspective matrix ─────
# cam_data.angle is Blender's "active" FOV:
#   HORIZONTAL / AUTO-landscape  → horizontal FOV
#   VERTICAL   / AUTO-portrait   → vertical FOV
# We always need fovY, so convert horizontal→vertical using the render aspect.
cam_data  = cam_obj.data
render    = scene.render
aspect    = (render.resolution_x * render.pixel_aspect_x) / \
            (render.resolution_y * render.pixel_aspect_y)
cam_angle = cam_data.angle   # radians

if cam_data.sensor_fit == 'VERTICAL':
    vfov_rad = cam_angle
elif cam_data.sensor_fit == 'HORIZONTAL':
    vfov_rad = 2 * math.atan(math.tan(cam_angle * 0.5) / aspect)
else:  # AUTO — landscape uses hFOV, portrait uses vFOV
    if aspect >= 1.0:
        vfov_rad = 2 * math.atan(math.tan(cam_angle * 0.5) / aspect)
    else:
        vfov_rad = cam_angle

fov_deg = round(math.degrees(vfov_rad), 4)

# ── Clip distances — converted to SPZ-local space if a GS object is used ──────
# Blender clip values are in world units. When the GS object has a scale
# transform, local-space distances = world distances / object_scale.
if gs_obj:
    obj_scale = gs_obj.matrix_world.to_scale()
    inv_scale = 3.0 / (obj_scale.x + obj_scale.y + obj_scale.z)  # 1 / avg_scale
    clip_near = cam_data.clip_start * inv_scale
    clip_far  = cam_data.clip_end   * inv_scale
else:
    clip_near = cam_data.clip_start
    clip_far  = cam_data.clip_end

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

# ── Apply FLIP_Y (180° X rotation: negate Y and Z) ───────────────────────────
if FLIP_Y:
    for i in range(frame_count):
        b = i * 6
        frames_flat[b + 1] = -frames_flat[b + 1]  # eye.y
        frames_flat[b + 2] = -frames_flat[b + 2]  # eye.z
        frames_flat[b + 4] = -frames_flat[b + 4]  # forward.y
        frames_flat[b + 5] = -frames_flat[b + 5]  # forward.z
    for c in callouts:
        c["pos"][1] = -c["pos"][1]
        c["pos"][2] = -c["pos"][2]
    print("HoloSplat: FLIP_Y applied to camera and callout positions")

# ── Export timeline markers ───────────────────────────────────────────────────
# Blender timeline markers (added with M in the Timeline/Dopesheet editor) are
# exported as a dict { markerName: frameNumber } where frame numbers are
# 0-based relative to the export start frame.
#
# In HTML, reference them with data-from / data-to / data-frame attributes:
#   <div class="hs-act" data-from="intro" data-to="desk_reveal" ...>
#
# Only markers within the exported frame range are included.
markers = {}
for marker in sorted(scene.timeline_markers, key=lambda m: m.frame):
    if f_start <= marker.frame <= f_end:
        # Frame is relative to export start (0-based)
        markers[marker.name] = marker.frame - f_start

print(f"HoloSplat: exported {len(markers)} marker(s): {list(markers.keys())}")

# ── Write JSON ────────────────────────────────────────────────────────────────
data = {
    "version"    : 1,
    "fps"        : fps,
    "frameCount" : frame_count,
    "fov"        : fov_deg,
    "near"       : round(clip_near, 6),
    "far"        : round(clip_far,  6),
    "frames"     : frames_flat,
    "callouts"   : callouts,
    "markers"    : markers,
}

out_dir  = bpy.path.abspath(OUTPUT_PATH)
name     = OUTPUT_NAME or Path(bpy.data.filepath).stem or "scene_anim"
out_file = str(Path(out_dir) / f"{name}.json")

with open(out_file, "w") as fp:
    json.dump(data, fp, separators=(",", ":"))

print(f"HoloSplat: saved → {out_file}")
