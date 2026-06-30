"""
HoloSplat — Asset export logic (library module).

Not meant to be run directly. export_holosplat_asset.py loads this file
fresh off disk every time it runs (via importlib, not a cached import) and
calls run(config), so editing this file takes effect on the next "Run
Script" in Blender — no need to re-open/re-paste the config script itself.

See export_holosplat_asset.py's own docstring for the full usage guide,
including the "<axis>=<value>" variant-naming convention parsed below.
"""

import bpy, math, json, mathutils, re
from pathlib import Path


def parse_variant_segments(name):
    """
    Split a dot-separated name into (base_name, variant_dict).

    Segments before the first "<axis>=<value>" segment make up the base
    name; every "<axis>=<value>" segment from there on (in any position,
    any count) is a variant axis assignment:

        "cup-left"                       -> ("cup-left", {})
        "cup-left.color=blue"            -> ("cup-left", {"color": "blue"})
        "cup-left.color=blue.size=big"   -> ("cup-left", {"color": "blue", "size": "big"})

    A name can't have "=" segments before non-"=" ones once one axis has
    started — in practice this never gets exercised since names are
    authored left-to-right, base name first.
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


def action_fcurves(action):
    """F-curves for an Action, across Blender versions.

    Blender 4.4 moved F-curves off the Action itself and into
    layers → strips → channelbags (the "slotted actions" redesign), so
    plain `action.fcurves` raises AttributeError there. Older Blenders
    still expose `.fcurves` directly. This checks for the legacy
    attribute first and falls back to walking the new structure.
    """
    if hasattr(action, 'fcurves'):
        return action.fcurves
    fcurves = []
    for layer in getattr(action, 'layers', []):
        for strip in layer.strips:
            for channelbag in getattr(strip, 'channelbags', []):
                fcurves.extend(channelbag.fcurves)
    return fcurves


def run(config):
    OUTPUT_PATH    = config['OUTPUT_PATH']
    OUTPUT_NAME    = config['OUTPUT_NAME']
    GS_OBJECT_NAME = config['GS_OBJECT_NAME']
    FLIP_Y         = config['FLIP_Y']

    scene = bpy.context.scene
    fps   = scene.render.fps

    # ── Resolve reference objects (ignored entirely) ─────────────────────────
    # Matches "reference" and any Blender auto-dedup suffix ("reference.001",
    # "reference.002", ...) — see _holosplat_export_lib.py's copy of this for
    # why. .all_objects is recursive, so nested subcollections are covered.
    reference_objects = set()
    for col in bpy.data.collections:
        if col.name == "reference" or re.match(r"^reference\.\d+$", col.name):
            reference_objects.update(col.all_objects)
    if reference_objects:
        print(f"HoloSplat: ignoring {len(reference_objects)} reference object(s): "
              f"{[o.name for o in reference_objects]}")

    gs_obj = scene.objects.get(GS_OBJECT_NAME) if GS_OBJECT_NAME else None
    gs_inv = gs_obj.matrix_world.inverted() if gs_obj else None

    Q_BL_TO_HS  = mathutils.Euler((-math.pi / 2, 0, 0)).to_quaternion()
    _R_BL_TO_HS = Q_BL_TO_HS.to_matrix().to_4x4()

    # ── Collect part objects (same convention as export_holosplat.py) ───────
    part_objects = {}
    if "HoloSplat Parts" in bpy.data.collections:
        for obj in bpy.data.collections["HoloSplat Parts"].all_objects:
            if obj.type == 'EMPTY' and obj not in reference_objects:
                part_objects[obj.name] = obj
    else:
        for obj in scene.objects:
            if obj.type == 'EMPTY' and obj.name.startswith("hs-part.") and obj not in reference_objects:
                part_objects[obj.name] = obj

    print(f"HoloSplat: found {len(part_objects)} part(s): {list(part_objects.keys())}")

    # ── Collect mask volume objects (same convention as export_holosplat.py) ─
    # Unlike part_objects (rest-pose-relative pose deltas), masks are animated
    # for a single purpose here: fading between color variants on a clip's own
    # in/hold/out timeline (see clip discovery below) — there's no equivalent
    # "rest pose" to subtract, so their matrices are absolute world transforms.
    mask_objects = {}  # prefix → object
    for obj in scene.objects:
        if obj.name.endswith('.mask') and obj not in reference_objects:
            mask_objects[obj.name[:-5]] = obj
    if "HoloSplat Masks" in bpy.data.collections:
        for obj in bpy.data.collections["HoloSplat Masks"].all_objects:
            if obj in reference_objects:
                continue
            name = obj.name
            prefix = name[:-5] if name.endswith('.mask') else name
            mask_objects.setdefault(prefix, obj)

    print(f"HoloSplat: found {len(mask_objects)} mask volume(s): {list(mask_objects.keys())}")

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

    def mask_world_matrix(obj):
        if gs_inv:
            m = gs_inv @ obj.matrix_world
        else:
            m = _R_BL_TO_HS @ obj.matrix_world @ _R_BL_TO_HS.inverted()
        return [m[r][c] for c in range(4) for r in range(4)]

    # ── Variant axes ───────────────────────────────────────────────────────
    # Scan every object in "HoloSplat Parts" (not just splat children — any
    # auxiliary object, e.g. a mask volume, can carry an "<axis>=<value>"
    # tag) and collect every axis/value pair found into a single registry.
    # This is just discovery: it doesn't decide how switching values
    # animates anything, only what axes/values exist for the editor to
    # offer a default-value picker.
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

    # ── Determine each part's splat base name + variants from its splat.* ───
    # children. The base name is everything before the first "<axis>=<value>"
    # segment (see parse_variant_segments) — a part with no variants at all
    # just has one splat.<name> child and that whole name is the base.
    # part_variants stays a flat list of each variant's full suffix string
    # (e.g. "color=blue") — exactly the filename suffix the runtime needs to
    # actually load this part's geometry (see export_holosplat_asset.py).
    part_splat_name = {}
    part_variants   = {pid: [] for pid in part_objects}
    for pid, empty_obj in part_objects.items():
        splat_children = [c for c in empty_obj.children
                           if c.name.startswith("splat.") and c not in reference_objects]
        if not splat_children:
            print(f"HoloSplat: WARNING — part '{pid}' has no child 'splat.<name>' object; skipping it")
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

    # ── Bind-pose (rest) matrices — recorded at frame 0 ──────────────────────
    saved_frame = scene.frame_current
    scene.frame_set(0)
    _part_rest_mats = {}
    for pid, obj in part_objects.items():
        if gs_inv:
            _part_rest_mats[pid] = (gs_inv @ obj.matrix_world).copy()
        else:
            _part_rest_mats[pid] = (_R_BL_TO_HS @ obj.matrix_world @ _R_BL_TO_HS.inverted()).copy()

    def transform_part(obj, pid):
        """Return (pos_list, quat_xyzw) as the delta from the bind pose at frame 0."""
        if gs_inv:
            curr = gs_inv @ obj.matrix_world
        else:
            curr = _R_BL_TO_HS @ obj.matrix_world @ _R_BL_TO_HS.inverted()
        rel = curr @ _part_rest_mats[pid].inverted()
        p   = rel.translation
        q   = rel.to_quaternion()
        return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]

    # ── Variant axis transitions (button-triggered color switches) ──────────
    # One marker triple per axis — "<axis>.in" / "<axis>.hold" / "<axis>.out"
    # — shared by every value of that axis, not one triple per value. Every
    # animated part/mask tagged "<axis>=<value>" in its name gets its own
    # per-frame data sampled across that single range; at runtime, choosing a
    # value plays that value's objects from .in to .hold while whatever value
    # was previously active plays from .hold to .out, simultaneously.
    axis_marker_names = set()
    for axis in axes:
        axis_marker_names.update((f"{axis}.in", f"{axis}.hold", f"{axis}.out"))

    transitions_out = {}
    for axis in axes:
        in_name, hold_name, out_name = f"{axis}.in", f"{axis}.hold", f"{axis}.out"
        marker_frames = {m.name: m.frame for m in scene.timeline_markers if m.name in (in_name, hold_name, out_name)}
        missing = [n for n in (in_name, hold_name, out_name) if n not in marker_frames]
        if missing:
            print(f"HoloSplat: WARNING — axis '{axis}' is missing marker(s) {missing}; "
                  f"skipping transition export for this axis")
            continue

        t_start = int(marker_frames[in_name])
        t_hold  = int(marker_frames[hold_name])
        t_end   = int(marker_frames[out_name])
        if not (t_start <= t_hold <= t_end):
            print(f"HoloSplat: WARNING — axis '{axis}' markers aren't in order "
                  f"(.in={t_start}, .hold={t_hold}, .out={t_end}); skipping")
            continue

        t_frame_count = t_end - t_start + 1
        t_hold_frame  = t_hold - t_start

        t_parts_out = []
        for pid, obj in part_objects.items():
            if pid not in part_splat_name:
                continue
            _, variant = parse_variant_segments(obj.name)
            value = variant.get(axis)
            if value is None:
                continue  # this part's empty isn't itself tagged for this axis
            action = obj.animation_data.action if obj.animation_data else None
            if not action:
                continue
            has_keyframe = any(
                any(t_start <= kp.co.x <= t_end for kp in fc.keyframe_points)
                for fc in action_fcurves(action)
            )
            if not has_keyframe:
                continue
            frames = []
            for f in range(t_start, t_end + 1):
                scene.frame_set(f)
                pos_hs, quat_xyzw = transform_part(obj, pid)
                frames.extend(round(x, 6) for x in pos_hs + quat_xyzw)
            t_parts_out.append({"id": "ctrl." + part_splat_name[pid], "value": value, "frames": frames})

        t_masks_out = []
        for prefix, obj in mask_objects.items():
            _, variant = parse_variant_segments(prefix)
            value = variant.get(axis)
            if value is None:
                continue
            action = obj.animation_data.action if obj.animation_data else None
            if not action:
                continue
            has_keyframe = any(
                any(t_start <= kp.co.x <= t_end for kp in fc.keyframe_points)
                for fc in action_fcurves(action)
            )
            if not has_keyframe:
                continue
            matrices = []
            for f in range(t_start, t_end + 1):
                scene.frame_set(f)
                matrices.extend(round(x, 6) for x in mask_world_matrix(obj))
            t_masks_out.append({"name": prefix, "value": value, "softEdge": 0.05, "matrices": matrices})

        if not t_parts_out and not t_masks_out:
            print(f"HoloSplat: WARNING — axis '{axis}' has transition markers but no "
                  f"tagged part/mask is animated in range ({t_start}..{t_end}); skipping")
            continue

        if FLIP_Y:
            for entry in t_parts_out:
                frames = entry["frames"]
                for i in range(t_frame_count):
                    b = i * 7
                    frames[b + 1] = -frames[b + 1]
                    frames[b + 2] = -frames[b + 2]
                    qx, qy, qz, qw = frames[b+3], frames[b+4], frames[b+5], frames[b+6]
                    frames[b+3] = qw
                    frames[b+4] = -qz
                    frames[b+5] = qy
                    frames[b+6] = -qx
            _flip_signs = [1,-1,-1,1, -1,1,1,-1, -1,1,1,-1, 1,-1,-1,1]
            for entry in t_masks_out:
                matrices = entry["matrices"]
                for i in range(t_frame_count):
                    b = i * 16
                    for k in range(16):
                        matrices[b + k] *= _flip_signs[k]

        transitions_out[axis] = {
            "fps": fps, "frameCount": t_frame_count, "holdFrame": t_hold_frame,
            "parts": t_parts_out, "masks": t_masks_out,
        }
        print(f"HoloSplat: axis '{axis}' transition — {t_frame_count} frames "
              f"(scene frames {t_start}..{t_end}), hold @ {t_hold_frame}, "
              f"{len(t_parts_out)} part(s), {len(t_masks_out)} mask(s)")

    # ── Asset states (declared via "state: <axis>=<value>" markers) ─────────
    # Each such marker is a single point on this asset's own timeline naming
    # the frame at which some state axis (e.g. "fold") reaches a particular
    # value (e.g. "folded"). Unlike the axis transitions above (one shared
    # in/hold/out triple, two values cross-fading), a state axis is one
    # continuous timeline spanning every one of its markers — switching
    # values seeks along it, forward or backward, through whatever
    # intermediate frames lie between (see playState() in src/viewer.js).
    state_axis_markers = {}  # axis -> {value: frame}
    state_marker_re = re.compile(r'^state:\s*([^.=\s]+)=([^=\s]+)$')
    for marker in scene.timeline_markers:
        m = state_marker_re.match(marker.name)
        if not m:
            continue
        axis, value = m.group(1), m.group(2)
        state_axis_markers.setdefault(axis, {})[value] = marker.frame

    states_out = {}
    for axis, value_frames in state_axis_markers.items():
        if len(value_frames) < 2:
            print(f"HoloSplat: WARNING — state axis '{axis}' has only one marker "
                  f"({list(value_frames)}); need at least two values to switch between; skipping")
            continue

        s_start = int(min(value_frames.values()))
        s_end   = int(max(value_frames.values()))
        s_frame_count = s_end - s_start + 1
        default_value = min(value_frames, key=lambda v: value_frames[v])

        s_parts_out = []
        for pid, obj in part_objects.items():
            if pid not in part_splat_name:
                continue
            action = obj.animation_data.action if obj.animation_data else None
            if not action:
                continue
            has_keyframe = any(
                any(s_start <= kp.co.x <= s_end for kp in fc.keyframe_points)
                for fc in action_fcurves(action)
            )
            if not has_keyframe:
                continue
            frames = []
            for f in range(s_start, s_end + 1):
                scene.frame_set(f)
                pos_hs, quat_xyzw = transform_part(obj, pid)
                frames.extend(round(x, 6) for x in pos_hs + quat_xyzw)
            s_parts_out.append({"id": "ctrl." + part_splat_name[pid], "frames": frames})

        s_masks_out = []
        for prefix, obj in mask_objects.items():
            action = obj.animation_data.action if obj.animation_data else None
            if not action:
                continue
            has_keyframe = any(
                any(s_start <= kp.co.x <= s_end for kp in fc.keyframe_points)
                for fc in action_fcurves(action)
            )
            if not has_keyframe:
                continue
            matrices = []
            for f in range(s_start, s_end + 1):
                scene.frame_set(f)
                matrices.extend(round(x, 6) for x in mask_world_matrix(obj))
            s_masks_out.append({"name": prefix, "softEdge": 0.05, "matrices": matrices})

        if not s_parts_out and not s_masks_out:
            print(f"HoloSplat: WARNING — state axis '{axis}' has markers but no tagged "
                  f"part/mask is animated in range ({s_start}..{s_end}); skipping")
            continue

        if FLIP_Y:
            for entry in s_parts_out:
                frames = entry["frames"]
                for i in range(s_frame_count):
                    b = i * 7
                    frames[b + 1] = -frames[b + 1]
                    frames[b + 2] = -frames[b + 2]
                    qx, qy, qz, qw = frames[b+3], frames[b+4], frames[b+5], frames[b+6]
                    frames[b+3] = qw
                    frames[b+4] = -qz
                    frames[b+5] = qy
                    frames[b+6] = -qx
            _flip_signs = [1,-1,-1,1, -1,1,1,-1, -1,1,1,-1, 1,-1,-1,1]
            for entry in s_masks_out:
                matrices = entry["matrices"]
                for i in range(s_frame_count):
                    b = i * 16
                    for k in range(16):
                        matrices[b + k] *= _flip_signs[k]

        states_out[axis] = {
            "fps": fps, "frameCount": s_frame_count,
            "markers": {v: int(f - s_start) for v, f in value_frames.items()},
            "default": default_value,
            "parts": s_parts_out, "masks": s_masks_out,
        }
        print(f"HoloSplat: state axis '{axis}' — {s_frame_count} frames "
              f"(scene frames {s_start}..{s_end}), default '{default_value}', "
              f"{len(s_parts_out)} part(s), {len(s_masks_out)} mask(s), "
              f"values: {list(value_frames.keys())}")

    # ── Discover clips from timeline markers (independent of variant axes —
    # see export_holosplat_asset.py's docstring) ─────────────────────────────
    # Markers named "state:..." declare/consume asset states (above) — skipped
    # here too so they don't get misread as a malformed/bare clip id.
    # Axis-transition markers (above) are also skipped here — they're not a
    # clip id, just shared in/hold/out points.
    clip_markers = {}  # clip_id -> {'in': frame, 'hold': frame, 'out': frame}

    for marker in scene.timeline_markers:
        name = marker.name
        if name.startswith("state:") or name in axis_marker_names:
            continue
        if name.endswith(".in"):
            clip_markers.setdefault(name[:-3], {})['in'] = marker.frame
        elif name.endswith(".out"):
            clip_markers.setdefault(name[:-4], {})['out'] = marker.frame
        else:
            clip_markers.setdefault(name, {})['hold'] = marker.frame

    print(f"HoloSplat: found {len(clip_markers)} clip marker group(s): {list(clip_markers.keys())}")

    clips_out = []
    for clip_id, roles in clip_markers.items():
        missing = [r for r in ('in', 'hold', 'out') if r not in roles]
        if missing:
            print(f"HoloSplat: WARNING — clip '{clip_id}' is missing marker(s) {missing} "
                  f"(need '{clip_id}.in', '{clip_id}', '{clip_id}.out'); skipping")
            continue

        clip_start = int(roles['in'])
        clip_end   = int(roles['out'])
        hold_abs   = int(roles['hold'])
        if not (clip_start <= hold_abs <= clip_end):
            print(f"HoloSplat: WARNING — clip '{clip_id}' markers aren't in order "
                  f"(.in={clip_start}, hold={hold_abs}, .out={clip_end}); skipping")
            continue

        hold_frame       = hold_abs - clip_start
        clip_frame_count = clip_end - clip_start + 1
        clip_objects_out = []

        for pid, obj in part_objects.items():
            if pid not in part_splat_name:
                continue
            action = obj.animation_data.action if obj.animation_data else None
            if not action:
                continue
            has_keyframe = any(
                any(clip_start <= kp.co.x <= clip_end for kp in fc.keyframe_points)
                for fc in action_fcurves(action)
            )
            if not has_keyframe:
                continue  # this part isn't animated in this clip's range

            frames = []
            for f in range(clip_start, clip_end + 1):
                scene.frame_set(f)
                pos_hs, quat_xyzw = transform_part(obj, pid)
                frames.extend([round(x, 6) for x in pos_hs + quat_xyzw])
            clip_objects_out.append({"id": "ctrl." + part_splat_name[pid], "frames": frames})

        # Masks fade between color variants on this clip's own timeline —
        # a clip may animate masks only (parts stay put while a mask reveals/
        # hides geometry), so this is independent of clip_objects_out below.
        clip_masks_out = []
        for prefix, obj in mask_objects.items():
            action = obj.animation_data.action if obj.animation_data else None
            if not action:
                continue
            has_keyframe = any(
                any(clip_start <= kp.co.x <= clip_end for kp in fc.keyframe_points)
                for fc in action_fcurves(action)
            )
            if not has_keyframe:
                continue  # this mask isn't animated in this clip's range

            matrices = []
            for f in range(clip_start, clip_end + 1):
                scene.frame_set(f)
                matrices.extend(round(x, 6) for x in mask_world_matrix(obj))
            clip_masks_out.append({"name": prefix, "softEdge": 0.05, "matrices": matrices})

        if not clip_objects_out and not clip_masks_out:
            print(f"HoloSplat: WARNING — clip '{clip_id}' has markers but no part or mask is "
                  f"animated in its range ({clip_start}..{clip_end}); skipping")
            continue

        if FLIP_Y:
            for entry in clip_objects_out:
                frames = entry["frames"]
                for i in range(clip_frame_count):
                    b = i * 7
                    frames[b + 1] = -frames[b + 1]   # py
                    frames[b + 2] = -frames[b + 2]   # pz
                    qx, qy, qz, qw = frames[b+3], frames[b+4], frames[b+5], frames[b+6]
                    frames[b+3] = qw
                    frames[b+4] = -qz
                    frames[b+5] = qy
                    frames[b+6] = -qx
            _flip_signs = [1,-1,-1,1, -1,1,1,-1, -1,1,1,-1, 1,-1,-1,1]
            for entry in clip_masks_out:
                matrices = entry["matrices"]
                for i in range(clip_frame_count):
                    b = i * 16
                    for k in range(16):
                        matrices[b + k] *= _flip_signs[k]

        clips_out.append({
            "id":         clip_id,
            "fps":        fps,
            "frameCount": clip_frame_count,
            "holdFrame":  hold_frame,
            "objects":    clip_objects_out,
            "masks":      clip_masks_out,
        })
        print(f"HoloSplat: clip '{clip_id}' — {clip_frame_count} frames "
              f"(scene frames {clip_start}..{clip_end}), hold @ {hold_frame}, "
              f"{len(clip_objects_out)} part(s), {len(clip_masks_out)} mask(s)")

    scene.frame_set(saved_frame)

    # ── Write JSON ──────────────────────────────────────────────────────────────
    # parts: id ("ctrl.<splatName>") -> { splatName, variants } — lets the
    # runtime actually load this asset's geometry (resolve <splatsDir>/
    # <splatName>.<variant>.<ext> per the asset's selected axis defaults),
    # independent of clips/axes which are about behaviour, not loading.
    parts_out = {
        "ctrl." + part_splat_name[pid]: {
            "splatName": part_splat_name[pid],
            "variants":  part_variants[pid],
        }
        for pid in part_splat_name
    }

    data = {
        "version":     1,
        "fps":         fps,
        "axes":        axes,
        "parts":       parts_out,
        "clips":       clips_out,
        "transitions": transitions_out,
        "states":      states_out,
        "properties":  properties_out,
    }

    out_dir  = bpy.path.abspath(OUTPUT_PATH)
    name     = OUTPUT_NAME or Path(bpy.data.filepath).stem or "asset"
    out_file = str(Path(out_dir) / f"{name}.json")

    with open(out_file, "w") as fp:
        json.dump(data, fp, separators=(",", ":"))

    print(f"HoloSplat: saved → {out_file}")
