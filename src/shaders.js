// WGSL shader for Gaussian splatting.
//
// GPU layout (must match CPU Float32Array layout in loaders):
//
//   struct Gaussian {         byte offset
//     pos   : vec3<f32>,   //  0  (12 bytes)
//     part  : f32,         // 12  (part/object index, stored as float 0.0/1.0/2.0…)
//     color : vec4<f32>,   // 16  (rgba — DC-only base color: 0.5 + SH_C0 * f_dc)
//     scale : vec3<f32>,   // 32  (12 bytes + 4 implicit padding = 16)
//     quat  : vec4<f32>,   // 48  (xyzw)
//   }; // stride = 64 bytes = 16 floats/gaussian
//
// Binding 3: per-part model-space → world-space transforms (col-major mat4, one per part).
// For single-part scenes part=0 and transforms[0] is the identity matrix.
//
// Binding 6: SH rest coefficients — packed f32 array (no vec3 padding).
//   Layout: shCoeffs[(gi * numSHBases + basis) * 3 + ch]  ch=0→r, 1→g, 2→b.
//   Bound to a tiny dummy buffer (12 bytes) when shDegree == 0.
//
// Uniform layout (192 bytes = 48 floats):
//   [0-15]  view matrix (col-major)
//   [16-31] proj matrix (col-major)
//   [32-33] viewport (width, height) in pixels
//   [34-35] focal (fx, fy) in pixels
//   [36]    splatScale multiplier
//   [37]    near (view-space units)
//   [38]    gamma (1.0 = linear, 2.2 = sRGB)
//   [39]    radiusCap (screen-space radius cap, fraction of viewport.y)
//   [40]    shDegree    (0/1/2/3, stored as f32, read as u32 in shader)
//   [41]    numSHBases  (0/3/8/15)
//   [42]    aaDilation  (covariance low-pass filter, default 0.3)
//   [43]    (padding)
//   [44]    hueShift    (turns, 0..1 — added to HSV hue; 0 = no-op)
//   [45]    satMul      (HSV saturation multiplier; 1 = no-op)
//   [46]    valMul      (HSV value multiplier; 1 = no-op)
//   [47]    (padding)

export const SHADER = /* wgsl */`

struct Uniforms {
  view       : mat4x4<f32>,
  proj       : mat4x4<f32>,
  viewport   : vec2<f32>,
  focal      : vec2<f32>,
  params     : vec4<f32>,   // .x = splatScale  .y = near  .z = gamma  .w = radiusCap
  shParams   : vec4<f32>,   // .x = shDegree  .y = numSHBases  .z = aaDilation
  colorParams: vec4<f32>,   // .x = hueShift (turns)  .y = satMul  .z = valMul
};

struct Gaussian {
  pos   : vec3<f32>,
  part  : f32,             // part index (0.0, 1.0, 2.0…) — cast to u32 in shader
  color : vec4<f32>,       // DC-only base color (0.5 + SH_C0 * f_dc)
  scale : vec3<f32>,
  quat  : vec4<f32>,
};

@group(0) @binding(0) var<uniform>       uniforms   : Uniforms;
@group(0) @binding(1) var<storage, read> gaussians  : array<Gaussian>;
@group(0) @binding(2) var<storage, read> order      : array<u32>;
@group(0) @binding(3) var<storage, read> transforms : array<mat4x4<f32>>;

// Mask volume: a world-space box that clips splats — splats inside render at
// full size, splats outside fade to zero over a softEdge band.
struct MaskVolume {
  invMatrix : mat4x4<f32>,  // world → local transform (precomputed inverse)
  params    : vec4<f32>,    // .x = softEdge (world-space fade width)
};
struct MaskUniforms {
  header  : vec4<u32>,             // .x = active volume count (0..8)
  volumes : array<MaskVolume, 8>,
};
@group(0) @binding(4) var<uniform>       maskUnif    : MaskUniforms;
@group(0) @binding(5) var<storage, read> partVolMask : array<u32>;

// SH rest coefficients — packed f32 array, no padding.
// Layout: for Gaussian gi, basis b: shCoeffs[(gi*numSHBases + b)*3 + channel].
// Bound to a tiny dummy buffer when shDegree == 0.
@group(0) @binding(6) var<storage, read> shCoeffs : array<f32>;

// TEMP debug instrumentation — writes intermediate per-splat values for the
// one gaussian index named in uniforms.shParams.w (−1 = disabled), so the
// GPU-computed math for a specific splat can be cross-checked on the CPU
// against hand-computed expected values from the raw source file.
@group(0) @binding(7) var<storage, read_write> debugOut : array<f32, 16>;

struct VOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) color         : vec4<f32>,
  // Raw ±1 quad-corner UV. The quad geometry itself is pre-warped to align
  // with the 2-D covariance's eigenvectors (see cs_preprocess), so this UV
  // directly parameterizes the ellipse — no per-pixel conic/Mahalanobis math
  // needed.
  @location(1) uv            : vec2<f32>,
};

// Per-splat results of the heavy per-gaussian math (covariance projection,
// eigendecomposition, SH), computed once per splat by cs_preprocess instead
// of once per *vertex* (6x redundant — every vertex of a splat's quad used
// to redo the same work). vs_main just reads this and places the quad.
//   ndc.xy = screen-space center (NDC, pre-offset)
//   ndc.z  = NDC depth (clip.z/clip.w) — ndc.z > 1.5 marks a degenerate/
//            discarded splat (near-plane, edge-on, etc.), matching the old
//            degen() sentinel.
//   v1/v2  = quad corner basis vectors (pixels), already scaled by sizeFade.
//   color  = final straight-alpha rgba (SH-evaluated rgb, alpha folded with
//            nearFade/sizeFade/maskFade/aaFactor).
struct SplatGeom {
  ndc   : vec4<f32>,
  v1    : vec2<f32>,
  v2    : vec2<f32>,
  color : vec4<f32>,
};
// read_write (compute-only) and read (vertex-only) views of the same buffer
// get distinct binding numbers — WGSL doesn't allow one module-scope
// resource to change access mode per entry point, and read_write storage
// isn't permitted outside the compute stage anyway.
@group(0) @binding(8)  var<storage, read_write> splatGeomOut : array<SplatGeom>;
@group(0) @binding(9)  var<storage, read>       splatGeomIn  : array<SplatGeom>;

struct PreprocessParams { count : vec4<u32> };
@group(0) @binding(10) var<uniform> ppParams : PreprocessParams;

// ── SH constants (match ply-loader.js) ──────────────────────────────────────
const SH_C1    = 0.4886025119029199;
const SH_C2_0  =  1.0925484305920792;
const SH_C2_1  = -1.0925484305920792;
const SH_C2_2  =  0.31539156525252005;
const SH_C2_3  = -1.0925484305920792;
const SH_C2_4  =  0.5462742152960396;
const SH_C3_0  = -0.5900435899266435;
const SH_C3_1  =  2.890611442640554;
const SH_C3_2  = -0.4570457994644658;
const SH_C3_3  =  0.3731763325901154;
const SH_C3_4  = -0.4570457994644658;
const SH_C3_5  =  1.445305721320277;
const SH_C3_6  = -0.5900435899266435;

// Uniform hue/sat/value tweak (see Uniforms.colorParams) — lets one splat
// model stand in for several color variants instead of baking each as
// separate geometry. Standard GLSL-style HSV round-trip (Sam Hocevar's
// branchless form), ported to WGSL.
fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
  let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), step(c.b, c.g));
  let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));
  let d = q.x - min(q.w, q.y);
  let e = 1.0e-10;
  return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
  let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

// Read one SH basis vector (3 packed floats) from the flat shCoeffs array.
// Each basis occupies 3 consecutive floats: [r, g, b].
fn shBasis(gi: u32, numBases: u32, b: u32) -> vec3<f32> {
  let o = (gi * numBases + b) * 3u;
  return vec3<f32>(shCoeffs[o], shCoeffs[o + 1u], shCoeffs[o + 2u]);
}

// Evaluate higher-order SH (degree 1–3) for Gaussian gi at direction dir.
// Returns the additive contribution to the DC color — caller clamps final sum.
fn evalSH(gi: u32, dir: vec3<f32>, shDeg: u32, numBases: u32) -> vec3<f32> {
  var result = vec3<f32>(0.0);
  if (shDeg == 0u) { return result; }

  let x = dir.x; let y = dir.y; let z = dir.z;

  let sh0 = shBasis(gi, numBases, 0u);
  let sh1 = shBasis(gi, numBases, 1u);
  let sh2 = shBasis(gi, numBases, 2u);
  result += SH_C1 * (-sh0 * y + sh1 * z - sh2 * x);
  if (shDeg == 1u) { return result; }

  let sh3 = shBasis(gi, numBases, 3u);
  let sh4 = shBasis(gi, numBases, 4u);
  let sh5 = shBasis(gi, numBases, 5u);
  let sh6 = shBasis(gi, numBases, 6u);
  let sh7 = shBasis(gi, numBases, 7u);
  let xx = x*x; let yy = y*y; let zz = z*z;
  let xy = x*y; let xz = x*z; let yz = y*z;
  result += SH_C2_0 * xy * sh3
          + SH_C2_1 * yz * sh4
          + SH_C2_2 * (2.0*zz - xx - yy) * sh5
          + SH_C2_3 * xz * sh6
          + SH_C2_4 * (xx - yy) * sh7;
  if (shDeg == 2u) { return result; }

  let sh8  = shBasis(gi, numBases,  8u);
  let sh9  = shBasis(gi, numBases,  9u);
  let sh10 = shBasis(gi, numBases, 10u);
  let sh11 = shBasis(gi, numBases, 11u);
  let sh12 = shBasis(gi, numBases, 12u);
  let sh13 = shBasis(gi, numBases, 13u);
  let sh14 = shBasis(gi, numBases, 14u);
  result += SH_C3_0 * y * (3.0*xx - yy)              * sh8
          + SH_C3_1 * xy * z                           * sh9
          + SH_C3_2 * y * (4.0*zz - xx - yy)          * sh10
          + SH_C3_3 * z * (2.0*zz - 3.0*xx - 3.0*yy) * sh11
          + SH_C3_4 * x * (4.0*zz - xx - yy)          * sh12
          + SH_C3_5 * (xx - yy) * z                   * sh13
          + SH_C3_6 * x * (xx - 3.0*yy)               * sh14;
  return result;
}

// Quaternion (xyzw) → column-major rotation matrix
fn quatToMat3(q: vec4<f32>) -> mat3x3<f32> {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0*(y*y+z*z),  2.0*(x*y+w*z),       2.0*(x*z-w*y)),
    vec3<f32>(2.0*(x*y-w*z),         1.0 - 2.0*(x*x+z*z), 2.0*(y*z+w*x)),
    vec3<f32>(2.0*(x*z+w*y),         2.0*(y*z-w*x),       1.0 - 2.0*(x*x+y*y))
  );
}

// Writes the degenerate-splat sentinel (moved outside clip space — its quad
// gets clipped away entirely, matching the old per-vertex degen() early-out).
fn writeDegenerate(gi: u32) {
  var o: SplatGeom;
  o.ndc   = vec4<f32>(0.0, 0.0, 2.0, 0.0);
  o.v1    = vec2<f32>(0.0);
  o.v2    = vec2<f32>(0.0);
  o.color = vec4<f32>(0.0);
  splatGeomOut[gi] = o;
}

// ── Per-splat preprocess (compute) ───────────────────────────────────────────
// Runs the heavy per-gaussian math — covariance projection, eigendecomposition,
// SH evaluation — exactly once per splat (not once per vertex). vs_main used
// to redo all of this 6x per splat (once per quad vertex); on weaker mobile
// GPUs that redundant vertex-shader work, not fragment/overdraw cost, turned
// out to be the dominant per-frame cost (lowering render resolution didn't
// help fps at all, which only makes sense if the bottleneck was vertex-bound).
@compute @workgroup_size(64)
fn cs_preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gi = gid.x;
  if gi >= ppParams.count.x { return; }

  let g = gaussians[gi];

  // ── Part transform: local → world → view ────────────────────────────────
  let partId    = u32(g.part);
  let partMat   = transforms[partId];
  let worldPos4 = partMat * vec4<f32>(g.pos, 1.0);
  let viewPos4  = uniforms.view * worldPos4;
  let t = viewPos4.xyz;

  // Discard if at or behind near plane
  let near = uniforms.params.y;
  if t.z > -near { writeDegenerate(gi); return; }

  // Near-depth fade: fade out splats within 3× the near distance so that
  // close-up splats don't cover the entire screen (matches Blender's behaviour).
  let depth     = -t.z;
  let nearFade  = clamp((depth - near) / (near * 2.0), 0.0, 1.0);

  // ── Mask volumes: fade splats outside assigned volume boxes ──────────────
  let volBits  = partVolMask[partId];
  var maskFade = 1.0;
  for (var vi = 0u; vi < maskUnif.header.x; vi++) {
    if (((volBits >> vi) & 1u) != 0u) {
      let vol  = maskUnif.volumes[vi];
      let lp   = vol.invMatrix * worldPos4;
      let q    = abs(lp.xyz) - vec3<f32>(0.5);
      let sdf  = length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
      let e    = max(vol.params.x, 0.001);
      maskFade *= 1.0 - smoothstep(-e, e, sdf);
    }
  }

  // ── 3-D covariance ────────────────────────────────────────────────────────
  let splatScale = uniforms.params.x;
  let s = g.scale * (splatScale * maskFade);
  let R = quatToMat3(g.quat);
  // M = R * diag(s); Cov3D = M * Mᵀ
  let M    = mat3x3<f32>(R[0]*s.x, R[1]*s.y, R[2]*s.z);
  let cov3 = M * transpose(M);

  // ── Project covariance to 2-D ─────────────────────────────────────────────
  // Perspective Jacobian at view-space point t (camera looks down –Z)
  let tz2 = t.z * t.z;
  let fx  = uniforms.focal.x;
  let fy  = uniforms.focal.y;
  let J = mat3x3<f32>(
    vec3<f32>(fx / (-t.z),  0.0,           0.0),
    vec3<f32>(0.0,          fy / (-t.z),   0.0),
    vec3<f32>(fx*t.x / tz2, fy*t.y / tz2, 0.0)
  );
  // Combined rotation: view × part — maps splat-local covariance to view space.
  // For single-part (identity partMat) this reduces to just the view rotation.
  let R_part = mat3x3<f32>(partMat[0].xyz, partMat[1].xyz, partMat[2].xyz);
  let W_view = mat3x3<f32>(uniforms.view[0].xyz, uniforms.view[1].xyz, uniforms.view[2].xyz);
  let W      = W_view * R_part;
  let T      = J * W;
  let cov2   = T * cov3 * transpose(T);

  // Extract 2×2 + low-pass filter (anti-aliasing dilation)
  let aaDil  = uniforms.shParams.z;
  let detRaw = cov2[0][0] * cov2[1][1] - cov2[0][1] * cov2[0][1];  // pre-dilation det(Σ)
  let a = cov2[0][0] + aaDil;  // Σ_xx  (cov2[col][row])
  let b = cov2[0][1];           // Σ_xy
  let c = cov2[1][1] + aaDil;  // Σ_yy

  let det = a*c - b*b;
  if det < 1e-4 { writeDegenerate(gi); return; }

  // Mip-Splatting opacity compensation: dilating the covariance for AA
  // lowers the Gaussian's peak density (det grows), which silently dims
  // every splat — small/thin splats (fine surface detail) the most, since
  // aaDil is a larger fraction of their det. Rescale alpha by sqrt(det(Σ)/det(Σ+aaDil·I))
  // to preserve the splat's original peak opacity under the dilated kernel.
  let aaFactor = sqrt(max(detRaw / det, 0.0));

  // ── Eigen-decomposition of the 2×2 covariance ─────────────────────────────
  // Quad geometry is aligned to the covariance's eigenvectors (matches
  // PlayCanvas/SuperSplat's actual gsplat shader) so each splat rasterizes
  // only the minimal area its ellipse actually covers — an axis-aligned
  // isotropic bound (sized to the larger eigenvalue in both screen axes) was
  // tried instead to dodge a suspected eigenvector-orientation instability,
  // but that didn't fix anything and meaningfully increased overdraw on
  // anisotropic splats (every elongated splat rasterizing a much bigger
  // square than it needs), which is a likely contributor to the overall
  // softness compared to SuperSplat.
  //
  // Degenerate (near edge-on, disk-like) splats — where the minor-axis
  // variance goes non-positive — are discarded outright, matching
  // GaussianSplats3D's eigenValue2 <= 0.0 early return, rather than forced
  // to a minimum width. A previous version floored lambda2 to 0.1, which
  // renders these as a faint but visible ~1px sliver instead of nothing —
  // on a curved/reflective surface, many splats sit near this grazing-angle
  // case, and the accumulated phantom slivers are a plausible source of the
  // "cloudy" look on specular surfaces specifically.
  let mid    = 0.5 * (a + c);
  let evDist = length(vec2<f32>((a - c) * 0.5, b));
  let lambda1 = mid + evDist;
  let lambda2 = mid - evDist;
  if lambda2 <= 0.0 { writeDegenerate(gi); return; }

  let vmin = min(1024.0, min(uniforms.viewport.x, uniforms.viewport.y));
  let l1 = 2.0 * min(sqrt(2.0 * lambda1), vmin);
  let l2 = 2.0 * min(sqrt(2.0 * lambda2), vmin);

  let eigenDir = normalize(vec2<f32>(b, lambda1 - a) + vec2<f32>(1e-6, 0.0));
  let v1 = l1 * eigenDir;
  let v2 = l2 * vec2<f32>(eigenDir.y, -eigenDir.x);

  // Clamp the major-axis extent: splats larger than radiusCap × viewport
  // height are faded out and capped, preventing a handful of close-up splats
  // from overdrawing the whole screen. radiusCap shrinks adaptively under
  // load — see Viewer#_updateAdaptiveQuality. l1/l2 are corner-to-center
  // offset magnitudes (already radii, not diameters — see v1/v2 below).
  let maxRadius = uniforms.viewport.y * uniforms.params.w;
  let extent    = max(l1, l2);
  // Same factor both shrinks the quad geometry and fades its alpha — capped
  // splats fade out as they're clamped, rather than abruptly losing their
  // true aspect ratio.
  let sizeFade = clamp(maxRadius / max(extent, 1.0), 0.0, 1.0);

  // ── View-dependent SH colour ─────────────────────────────────────────────
  // Camera world-space position from the view matrix:
  //   view = [R | -R*cam], so cam = -Rᵀ * t  where t = view[3].xyz
  // In column-major WGSL: cam.x = -dot(view[0].xyz, view[3].xyz), etc.
  let shDeg   = u32(uniforms.shParams.x);
  let shBases = u32(uniforms.shParams.y);
  var rgb = g.color.rgb;
  if (shDeg > 0u) {
    let camWorld = vec3<f32>(
      -dot(uniforms.view[0].xyz, uniforms.view[3].xyz),
      -dot(uniforms.view[1].xyz, uniforms.view[3].xyz),
      -dot(uniforms.view[2].xyz, uniforms.view[3].xyz)
    );
    // Reference convention (3DGS eval_sh): dir = point - camera (outward
    // along the view ray), not camera - point. Degree-1/3 terms are odd in
    // dir, so this sign was previously inverting half the SH contribution.
    let shDir = normalize(worldPos4.xyz - camWorld);
    rgb = rgb + evalSH(gi, shDir, shDeg, shBases);
  }
  // Single clamp after the full DC + SH-rest sum — g.color.rgb (the DC term)
  // is intentionally left unclamped by the loader, since clamping it before
  // adding the higher-order terms would bake in the wrong value.
  rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));

  // Uniform hue/sat/value tweak (see Uniforms.colorParams) — a uniform branch
  // (same for every thread in the dispatch), so the default no-op case costs
  // nothing beyond the comparisons.
  let cp = uniforms.colorParams;
  if (cp.x != 0.0 || cp.y != 1.0 || cp.z != 1.0) {
    var hsv = rgb2hsv(rgb);
    hsv.x = fract(hsv.x + cp.x);
    hsv.y = clamp(hsv.y * cp.y, 0.0, 1.0);
    hsv.z = clamp(hsv.z * cp.z, 0.0, 1.0);
    rgb = hsv2rgb(hsv);
  }

  let clip  = uniforms.proj * viewPos4;
  let ndcXY = clip.xy / clip.w;

  var o: SplatGeom;
  o.ndc   = vec4<f32>(ndcXY, clip.z / clip.w, 0.0);
  o.v1    = v1 * sizeFade;
  o.v2    = v2 * sizeFade;
  o.color = vec4<f32>(rgb, g.color.a * nearFade * sizeFade * maskFade * aaFactor);
  splatGeomOut[gi] = o;

  // TEMP debug instrumentation — see binding(7) declaration above. Writable
  // directly here (compute allows read_write storage; the old vertex-shader
  // version had to smuggle these out via varyings into fs_main instead).
  if gi == u32(uniforms.shParams.w) {
    debugOut[0]  = t.z; debugOut[1]  = a;       debugOut[2]  = b;        debugOut[3]  = c;
    debugOut[4]  = det; debugOut[5]  = lambda1; debugOut[6]  = lambda2;  debugOut[7]  = extent;
    debugOut[8]  = sizeFade; debugOut[9] = nearFade; debugOut[10] = aaFactor; debugOut[11] = o.color.a;
    debugOut[12] = rgb.x; debugOut[13] = rgb.y; debugOut[14] = rgb.z; debugOut[15] = worldPos4.x;
  }
}

@vertex
fn vs_main(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> VOut {
  // Quad corners (two triangles, CCW)
  const corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0)
  );

  let gi     = order[ii];
  let geo    = splatGeomIn[gi];
  let corner = corners[vi];

  var o: VOut;
  if geo.ndc.z > 1.5 {
    o.clipPos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    o.color   = vec4<f32>(0.0);
    o.uv      = vec2<f32>(0.0);
    return o;
  }

  // corner (±1,±1) is split across the two eigenvector axes (v1, v2) instead
  // of a single axis-aligned radius — this is what lets fs_main use a plain
  // circular falloff over the unwarped corner UV.
  let pixOff = corner.x * geo.v1 + corner.y * geo.v2;
  let ndcOff = pixOff / (uniforms.viewport * 0.5);

  o.clipPos = vec4<f32>(geo.ndc.xy + ndcOff, geo.ndc.z, 1.0);
  o.color   = geo.color;
  o.uv      = corner;
  return o;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  // Plain Gaussian falloff — matches the original 3DGS reference renderer and
  // mkkellogg/GaussianSplats3D (an independent, widely-used implementation),
  // not PlayCanvas's hard-edged normExp, which turned out to be a
  // PlayCanvas-specific stylistic choice rather than a more "correct" shape.
  // in.uv (raw ±1 quad corner) is built so dot(uv,uv)=1 at the edge
  // corresponds to a true Mahalanobis² of 8 (since the quad extent is
  // sqrt(8·λ) — see cs_preprocess), so the equivalent exp(-0.5·8·A) = exp(-4·A).
  let A = dot(in.uv, in.uv);
  if A > 1.0 { discard; }
  let alpha = in.color.a * exp(-4.0 * A);
  if alpha < 1.0/255.0 { discard; }
  let gamma = max(uniforms.params.z, 0.01);
  let rgb = pow(max(in.color.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / gamma));
  return vec4<f32>(rgb * alpha, alpha);
}
`;
