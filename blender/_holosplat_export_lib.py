"""
HoloSplat — Camera + Parts export logic (library module).

Not meant to be run directly. export_holosplat.py loads this file fresh off
disk every time it runs (via importlib, not a cached import) and calls
run(config), so editing this file takes effect on the next "Run Script" in
Blender — no need to re-open/re-paste the config script itself.

See export_holosplat.py's own docstring for the full usage guide and all
naming conventions (parts, reference objects, callouts, masks, clips,
coordinate system) — none of that is repeated here.
"""

import bpy, json, math, mathutils, re
from pathlib import Path


def parse_variant_segments(name):
    """
    Split a dot-separated name into (base_name, variant_dict) — see
    _holosplat_asset_lib.py's copy of this for the full naming convention
    ("<axis>=<value>" segments after the base name).
    """
    parts = name.split(".")
    base_parts = []
    variant = {}
    for p in parts:
        if "=" in p:
            axis, _, value = p.partition("=")
            variant[axis] = value
        else:
            base_parts.append(p)
    return ".".join(base_parts), variant


def run(config):
    OUTPUT_PATH    = config['OUTPUT_PATH']
    OUTPUT_NAME    = config['OUTPUT_NAME']
    CAMERA_NAME    = config['CAMERA_NAME']
    FRAME_START    = config['FRAME_START']
    FRAME_END      = config['FRAME_END']
    GS_OBJECT_NAME = config['GS_OBJECT_NAME']
    FLIP_Y         = config['FLIP_Y']

    scene = bpy.context.scene

    # ── Resolve reference objects (ignored entirely) ─────────────────────────
    # Matches "reference" and any Blender auto-dedup suffix ("reference.001",
    # "reference.002", ...) — multiple collections all named "reference" get
    # renamed that way by Blender, e.g. if you nest one per sub-rig instead
    # of using a single top-level one; missing those would silently let
    # reference objects back into the export. .all_objects is recursive, so
    # nested subcollections under any of them are covered too.
    reference_objects = set()
    for col in bpy.data.collections:
        if col.name == "reference" or re.match(r"^reference\.\d+$", col.name):
            reference_objects.update(col.all_objects)
    if reference_objects:
        print(f"HoloSplat: ignoring {len(reference_objects)} reference object(s): "
              f"{[o.name for o in reference_objects]}")

    # ── Resolve camera ────────────────────────────────────────────────────────
    cam_obj = scene.objects.get(CAMERA_NAME) if CAMERA_NAME else scene.camera
    if cam_obj is None or cam_obj.type != 'CAMERA':
        raise RuntimeError(
            "HoloSplat export: no camera found. "
            "Set CAMERA_NAME or make sure the scene has an active camera."
        )

    # ── Resolve GS object (optional) ──────────────────────────────────────────
    gs_obj = scene.objects.get(GS_OBJECT_NAME) if GS_OBJECT_NAME else None
    gs_inv = gs_obj.matrix_world.inverted() if gs_obj else None

    if gs_obj:
        print(f"HoloSplat: using GS object '{gs_obj.name}' — camera will be in its local space")
    else:
        print("HoloSplat: no GS object set — using default Blender → Y-up axis conversion")

    # ── Frame range ────────────────────────────────────────────────────────────
    f_start = FRAME_START if FRAME_START is not None else scene.frame_start
    f_end   = FRAME_END   if FRAME_END   is not None else scene.frame_end
    fps     = scene.render.fps

    # ── FOV from camera — vertical (fovY), matching WebGPU perspective matrix ─
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

    # ── Clip distances — converted to SPZ-local space if a GS object is used ──
    if gs_obj:
        obj_scale = gs_obj.matrix_world.to_scale()
        inv_scale = 3.0 / (obj_scale.x + obj_scale.y + obj_scale.z)  # 1 / avg_scale
        clip_near = cam_data.clip_start * inv_scale
        clip_far  = cam_data.clip_end   * inv_scale
    else:
        clip_near = cam_data.clip_start
        clip_far  = cam_data.clip_end

    # ── Coordinate conversion helpers ─────────────────────────────────────────

    def bl_to_hs(v):
        """Blender (X right, Y fwd, Z up)  →  HoloSplat (X right, Y up, Z back)"""
        return [v.x, v.z, -v.y]

    def normalize(v):
        l = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        return [v[0]/l, v[1]/l, v[2]/l] if l > 1e-9 else [0, 0, -1]

    # ── Collect callout objects ────────────────────────────────────────────────
    callout_objects = {}   # id → object (deduped)

    # Method 1: "HoloSplat Callouts" collection
    if "HoloSplat Callouts" in bpy.data.collections:
        for obj in bpy.data.collections["HoloSplat Callouts"].all_objects:
            if obj not in reference_objects:
                callout_objects[obj.name] = obj

    # Method 2: names starting with "hs."
    for obj in scene.objects:
        if obj.name.startswith("hs.") and obj not in reference_objects:
            callout_objects[obj.name] = obj

    callouts = []
    for name, obj in callout_objects.items():
        cid = name.removeprefix("hs.")
        world_pos = obj.matrix_world.translation

        if gs_inv:
            local_pos = gs_inv @ world_pos
            pos = [local_pos.x, local_pos.y, local_pos.z]
        else:
            pos = bl_to_hs(world_pos)

        callouts.append({"id": cid, "pos": [round(x, 6) for x in pos]})

    print(f"HoloSplat: found {len(callouts)} callout(s): {[c['id'] for c in callouts]}")

    # ── Collect mask volume objects ────────────────────────────────────────────
    mask_objects = {}  # prefix → object

    for obj in scene.objects:
        if obj.name.endswith('.mask') and obj not in reference_objects:
            prefix = obj.name[:-5]
            mask_objects[prefix] = obj

    if "HoloSplat Masks" in bpy.data.collections:
        for obj in bpy.data.collections["HoloSplat Masks"].all_objects:
            if obj in reference_objects:
                continue
            name   = obj.name
            prefix = name[:-5] if name.endswith('.mask') else name
            if prefix not in mask_objects:
                mask_objects[prefix] = obj

    print(f"HoloSplat: found {len(mask_objects)} mask volume(s): {list(mask_objects.keys())}")
    if not mask_objects:
        candidates = [o.name for o in scene.objects if 'mask' in o.name.lower()]
        if candidates:
            print(
                "HoloSplat: no mask volumes matched the naming convention "
                f"(object name must end with '.mask', case-sensitive). "
                f"Found similar object name(s): {candidates}"
            )

    mask_frame_data = {prefix: [] for prefix in mask_objects}

    # ── Collect property objects ─────────────────────────────────────────────
    # An Empty named "property: <name>" inside a collection named "properties"
    # carries a free-form bag of named float values as Blender custom
    # properties (Object Properties panel → Custom Properties) — e.g. a
    # "property: color" Empty with custom properties hue=280, sat=1, val=1.
    # Exported as data["properties"]["<name>"] = {key: float, ...}; nothing
    # here assigns meaning to specific names/keys — that's up to whatever
    # reads the JSON at runtime.
    PROPERTY_PREFIX = "property:"

    def custom_float_properties(obj):
        out = {}
        for key in obj.keys():
            if key.startswith("_"):  # skip Blender's "_RNA_UI" metadata etc.
                continue
            val = obj[key]
            if isinstance(val, (int, float)):
                out[key] = float(val)
        return out

    property_objects = {}  # name -> object
    props_collection = bpy.data.collections.get("properties")
    if props_collection:
        for obj in props_collection.all_objects:
            if obj in reference_objects or obj.type != 'EMPTY':
                continue
            if not obj.name.startswith(PROPERTY_PREFIX):
                print(f"HoloSplat: WARNING — object '{obj.name}' is in the 'properties' "
                      f"collection but its name doesn't start with 'property:'; skipping")
                continue
            pname = obj.name[len(PROPERTY_PREFIX):].strip()
            if not pname:
                print(f"HoloSplat: WARNING — object '{obj.name}' has an empty property "
                      f"name after 'property:'; skipping")
                continue
            property_objects[pname] = obj

    properties_out = {}
    for pname, obj in property_objects.items():
        values = custom_float_properties(obj)
        if not values:
            print(f"HoloSplat: WARNING — property object '{pname}' has no numeric "
                  f"custom properties; skipping")
            continue
        properties_out[pname] = values

    if properties_out:
        print(f"HoloSplat: found {len(properties_out)} propert"
              f"{'y' if len(properties_out) == 1 else 'ies'}: "
              f"{ {k: list(v.keys()) for k, v in properties_out.items()} }")

    # ── Collect part objects ───────────────────────────────────────────────────
    part_objects = {}   # id → object (deduped, insertion-ordered)

    if "HoloSplat Parts" in bpy.data.collections:
        for obj in bpy.data.collections["HoloSplat Parts"].all_objects:
            if obj.type == 'EMPTY' and obj not in reference_objects:
                part_objects[obj.name] = obj
    else:
        for obj in scene.objects:
            if obj.type == 'EMPTY' and obj.name.startswith("hs-part.") and obj not in reference_objects:
                part_objects[obj.name] = obj

    print(f"HoloSplat: found {len(part_objects)} part(s): {list(part_objects.keys())}")

    # ── Variant axes ───────────────────────────────────────────────────────
    # Scan every object in "HoloSplat Parts" (not just splat children — any
    # auxiliary object, e.g. a mask volume, can carry an "<axis>=<value>"
    # tag) and collect every axis/value pair found. Discovery only — see
    # export_holosplat_asset.py for where switching values actually drives
    # anything.
    axes = {}  # axis -> set of values
    if "HoloSplat Parts" in bpy.data.collections:
        for obj in bpy.data.collections["HoloSplat Parts"].all_objects:
            if obj in reference_objects:
                continue
            _, variant = parse_variant_segments(obj.name)
            for axis, value in variant.items():
                axes.setdefault(axis, set()).add(value)
    axes = {axis: sorted(values) for axis, values in sorted(axes.items())}
    if axes:
        print(f"HoloSplat: found {len(axes)} variant axis/axes: "
              f"{{{', '.join(f'{a}: {v}' for a, v in axes.items())}}}")

    # ── Determine each part's splat file + variants from its child objects ───
    # The base name is everything before the first "<axis>=<value>" segment
    # (see parse_variant_segments) — a part with no variants at all just has
    # one splat.<name> child and that whole name is the base. part_variants
    # stays a flat list of each variant's full suffix string (e.g.
    # "color=blue", or "color=blue.size=big" for multi-axis variants) since
    # that's exactly the filename suffix resolvePartPaths (editor.js) needs.
    part_splat_name = {}
    part_variants   = {pid: [] for pid in part_objects}

    for pid, empty_obj in part_objects.items():
        splat_children = [c for c in empty_obj.children
                           if c.name.startswith("splat.") and c not in reference_objects]
        if not splat_children:
            print(f"HoloSplat: WARNING — part '{pid}' has no child 'splat.<name>' object; "
                  f"skipping it (it has nothing telling HoloSplat which file to load)")
            continue

        base_names = set()
        suffixes   = []
        for child in splat_children:
            raw = child.name[len("splat."):]
            base, variant = parse_variant_segments(raw)
            base_names.add(base)
            if variant:
                suffixes.append(raw[len(base) + 1:])  # everything after "<base>."

        if len(base_names) > 1:
            print(f"HoloSplat: WARNING — part '{pid}' has splat children with "
                  f"inconsistent base names {sorted(base_names)}; using "
                  f"'{sorted(base_names)[0]}'")
        part_splat_name[pid] = sorted(base_names)[0]
        part_variants[pid]   = sorted(set(suffixes))

    for pid, variants in part_variants.items():
        if variants:
            print(f"HoloSplat: part '{pid}' has {len(variants)} variant(s): {variants}")

    # Quaternion for Blender → HoloSplat axis conversion (Rx −90°)
    Q_BL_TO_HS = mathutils.Euler((-math.pi / 2, 0, 0)).to_quaternion()
    _R_BL_TO_HS = Q_BL_TO_HS.to_matrix().to_4x4()

    # ── Bind-pose (rest) matrices — recorded at f_start ──────────────────────
    scene.frame_set(f_start)
    _part_rest_mats = {}
    for _pid, _obj in part_objects.items():
        if gs_inv:
            _part_rest_mats[_pid] = (gs_inv @ _obj.matrix_world).copy()
        else:
            _part_rest_mats[_pid] = (_R_BL_TO_HS @ _obj.matrix_world @ _R_BL_TO_HS.inverted()).copy()

    def transform_part(obj, pid):
        """Return (pos_list, quat_xyzw) as the delta from the bind pose at f_start."""
        if gs_inv:
            curr = gs_inv @ obj.matrix_world
        else:
            curr = _R_BL_TO_HS @ obj.matrix_world @ _R_BL_TO_HS.inverted()
        rel = curr @ _part_rest_mats[pid].inverted()
        p   = rel.translation
        q   = rel.to_quaternion()   # Blender (w,x,y,z)
        return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]

    part_frame_data = {pid: [] for pid in part_objects}

    # ── Resolve focal-point object (before frame loop) ────────────────────────
    focal_obj = None
    for obj in scene.objects:
        if obj.name in ("focal-point", "hs-focal-point") and obj.type == 'EMPTY' and obj not in reference_objects:
            focal_obj = obj
            break

    focal_frames_flat = []  # 3 floats per frame (x, y, z)

    # ── Resolve asset anchors (before frame loop) ─────────────────────────────
    # An Empty named "hs-anchor.<assetId>" is a pure parent transform for an
    # externally-loaded asset (e.g. the headphones rig) — no splat.* child of
    # its own. Its world transform is exported as-is (not bind-pose relative
    # like transform_part — there's no rest pose to subtract; the asset's own
    # parts already carry their own bind pose from their own export).
    anchor_objects = {}  # assetId -> object
    for obj in scene.objects:
        if obj.name.startswith("hs-anchor.") and obj not in reference_objects:
            anchor_objects[obj.name[len("hs-anchor."):]] = obj

    if anchor_objects:
        print(f"HoloSplat: found {len(anchor_objects)} asset anchor(s): {list(anchor_objects.keys())}")

    anchor_frame_data = {asset_id: [] for asset_id in anchor_objects}

    # ── Export camera per frame ────────────────────────────────────────────────
    saved_frame = scene.frame_current
    frames_flat = []

    for f in range(f_start, f_end + 1):
        scene.frame_set(f)

        mw = cam_obj.matrix_world

        if gs_inv:
            local_pos = gs_inv @ mw.translation
            fwd_world = -mw.col[2].xyz
            local_fwd = (gs_inv.to_3x3() @ fwd_world).normalized()
            eye     = [local_pos.x, local_pos.y, local_pos.z]
            forward = normalize([local_fwd.x, local_fwd.y, local_fwd.z])
        else:
            pos    = mw.translation
            fwd_bl = -mw.col[2].xyz
            eye     = bl_to_hs(pos)
            forward = normalize(bl_to_hs(fwd_bl))

        frames_flat.extend([round(x, 6) for x in eye + forward])

        for pid, obj in part_objects.items():
            pos_hs, quat_xyzw = transform_part(obj, pid)
            part_frame_data[pid].extend(
                [round(x, 6) for x in pos_hs + quat_xyzw]
            )

        for prefix, obj in mask_objects.items():
            if gs_inv:
                m = gs_inv @ obj.matrix_world
            else:
                m = _R_BL_TO_HS @ obj.matrix_world @ _R_BL_TO_HS.inverted()
            flat = [m[r][c] for c in range(4) for r in range(4)]
            mask_frame_data[prefix].extend([round(x, 6) for x in flat])

        if focal_obj:
            wp = focal_obj.matrix_world.translation
            if gs_inv:
                lp = gs_inv @ wp
                focal_frames_flat.extend([round(lp.x, 6), round(lp.y, 6), round(lp.z, 6)])
            else:
                focal_frames_flat.extend([round(x, 6) for x in bl_to_hs(wp)])

        for asset_id, obj in anchor_objects.items():
            if gs_inv:
                m = gs_inv @ obj.matrix_world
            else:
                m = _R_BL_TO_HS @ obj.matrix_world @ _R_BL_TO_HS.inverted()
            p = m.translation
            q = m.to_quaternion()
            anchor_frame_data[asset_id].extend(
                [round(x, 6) for x in [p.x, p.y, p.z, q.x, q.y, q.z, q.w]]
            )

    scene.frame_set(saved_frame)   # restore original frame

    frame_count = f_end - f_start + 1
    print(f"HoloSplat: exported {frame_count} frames at {fps} fps  (duration {frame_count/fps:.2f}s)")

    # ── Focal point (orbit anchor for the free-camera mode) ──────────────────
    focal_point  = None
    focal_frames = None   # only set when focal point is animated

    if focal_obj:
        focal_point = focal_frames_flat[:3]

        fp0 = focal_frames_flat[:3]
        animated = any(
            abs(focal_frames_flat[i * 3 + j] - fp0[j]) > 1e-4
            for i in range(1, frame_count)
            for j in range(3)
        )
        if animated:
            focal_frames = focal_frames_flat
            print(f"HoloSplat: focal point '{focal_obj.name}' is animated — exporting per-frame positions")
        else:
            print(f"HoloSplat: focal point '{focal_obj.name}' (static) → {focal_point}")

    # ── Apply FLIP_Y (180° X rotation: negate Y and Z) ────────────────────────
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
        for frames in part_frame_data.values():
            for i in range(frame_count):
                b = i * 7
                frames[b + 1] = -frames[b + 1]   # py
                frames[b + 2] = -frames[b + 2]   # pz
                qx, qy, qz, qw = frames[b+3], frames[b+4], frames[b+5], frames[b+6]
                frames[b+3] = qw
                frames[b+4] = -qz
                frames[b+5] = qy
                frames[b+6] = -qx
        for frames in anchor_frame_data.values():
            for i in range(frame_count):
                b = i * 7
                frames[b + 1] = -frames[b + 1]   # py
                frames[b + 2] = -frames[b + 2]   # pz
                qx, qy, qz, qw = frames[b+3], frames[b+4], frames[b+5], frames[b+6]
                frames[b+3] = qw
                frames[b+4] = -qz
                frames[b+5] = qy
                frames[b+6] = -qx
        if focal_point:
            focal_point[1] = -focal_point[1]
            focal_point[2] = -focal_point[2]
        if focal_frames:
            for i in range(frame_count):
                focal_frames[i * 3 + 1] = -focal_frames[i * 3 + 1]
                focal_frames[i * 3 + 2] = -focal_frames[i * 3 + 2]
        _flip_signs = [1,-1,-1,1, -1,1,1,-1, -1,1,1,-1, 1,-1,-1,1]
        for frames in mask_frame_data.values():
            for i in range(frame_count):
                b = i * 16
                for k in range(16):
                    frames[b + k] *= _flip_signs[k]
        print("HoloSplat: FLIP_Y applied to camera, callouts, part transforms, and mask volumes")

    # ── Export timeline markers ───────────────────────────────────────────────
    # Markers named "state: <assetId>.<axis>=<value>" call into an asset's own
    # state axis (see export_holosplat_asset.py) — they're instructions for an
    # asset, not scene boundaries, so they're collected into state_calls
    # instead of the generic markers dict. Resolving them at runtime (seeking
    # the named asset's state axis to the target value) is Viewer's job — see
    # playState()/_syncAssetStates() in src/viewer.js.
    state_call_re = re.compile(r'^state:\s*([^.=\s]+)\.([^=\s]+)=([^=\s]+)$')
    markers = {}
    state_calls = []
    state_markers_skipped = 0
    for marker in sorted(scene.timeline_markers, key=lambda m: m.frame):
        if marker.name.startswith("state:"):
            m = state_call_re.match(marker.name)
            if m and f_start <= marker.frame <= f_end:
                asset, axis, value = m.groups()
                state_calls.append({
                    "frame": marker.frame - f_start,
                    "asset": asset, "axis": axis, "value": value,
                })
            else:
                state_markers_skipped += 1
            continue
        if f_start <= marker.frame <= f_end:
            markers[marker.name] = marker.frame - f_start

    if state_markers_skipped:
        print(f"HoloSplat: ignoring {state_markers_skipped} malformed/out-of-range 'state:' marker(s) "
              f"(expected 'state: <assetId>.<axis>=<value>')")
    if state_calls:
        call_strs = [f"{c['asset']}.{c['axis']}={c['value']}" for c in state_calls]
        print(f"HoloSplat: exported {len(state_calls)} state call(s): {call_strs}")
    print(f"HoloSplat: exported {len(markers)} marker(s): {list(markers.keys())}")

    # ── Write JSON ──────────────────────────────────────────────────────────────
    objects_out = []
    for pid, frames in part_frame_data.items():
        if pid not in part_splat_name:
            continue
        entry = {"id": "ctrl." + part_splat_name[pid], "frames": frames}
        if part_variants.get(pid):
            entry["variants"] = sorted(part_variants[pid])
        objects_out.append(entry)

    volumes_out = [
        {"name": prefix, "softEdge": 0.05, "matrices": frames}
        for prefix, frames in mask_frame_data.items()
    ]

    anchors_out = [
        {"id": "hs-anchor." + asset_id, "asset": asset_id, "frames": frames}
        for asset_id, frames in anchor_frame_data.items()
    ]

    data = {
        "version"     : 2 if objects_out else 1,
        "fps"         : fps,
        "frameCount"  : frame_count,
        "fov"         : fov_deg,
        "near"        : round(clip_near, 6),
        "far"         : round(clip_far,  6),
        "frames"      : frames_flat,
        "objects"     : objects_out,
        "volumes"     : volumes_out,
        "callouts"    : callouts,
        "markers"     : markers,
        "stateCalls"  : state_calls,
        "axes"        : axes,
        "anchors"     : anchors_out,
        "focalPoint"  : focal_point,
        "focalFrames" : focal_frames,
        "properties"  : properties_out,
    }

    out_dir  = bpy.path.abspath(OUTPUT_PATH)
    name     = OUTPUT_NAME or Path(bpy.data.filepath).stem or "scene_anim"
    out_file = str(Path(out_dir) / f"{name}.json")

    with open(out_file, "w") as fp:
        json.dump(data, fp, separators=(",", ":"))

    print(f"HoloSplat: saved → {out_file}")
