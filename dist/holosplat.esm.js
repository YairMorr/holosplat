var Jt=`

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
  part  : f32,             // part index (0.0, 1.0, 2.0\u2026) \u2014 cast to u32 in shader
  color : vec4<f32>,       // DC-only base color (0.5 + SH_C0 * f_dc)
  scale : vec3<f32>,
  quat  : vec4<f32>,
};

@group(0) @binding(0) var<uniform>       uniforms   : Uniforms;
@group(0) @binding(1) var<storage, read> gaussians  : array<Gaussian>;
@group(0) @binding(2) var<storage, read> order      : array<u32>;
@group(0) @binding(3) var<storage, read> transforms : array<mat4x4<f32>>;

// Mask volume: a world-space box that clips splats \u2014 splats inside render at
// full size, splats outside fade to zero over a softEdge band.
struct MaskVolume {
  invMatrix : mat4x4<f32>,  // world \u2192 local transform (precomputed inverse)
  params    : vec4<f32>,    // .x = softEdge (world-space fade width)
};
struct MaskUniforms {
  header  : vec4<u32>,             // .x = active volume count (0..8)
  volumes : array<MaskVolume, 8>,
};
@group(0) @binding(4) var<uniform>       maskUnif    : MaskUniforms;
@group(0) @binding(5) var<storage, read> partVolMask : array<u32>;

// SH rest coefficients \u2014 packed f32 array, no padding.
// Layout: for Gaussian gi, basis b: shCoeffs[(gi*numSHBases + b)*3 + channel].
// Bound to a tiny dummy buffer when shDegree == 0.
@group(0) @binding(6) var<storage, read> shCoeffs : array<f32>;

// TEMP debug instrumentation \u2014 writes intermediate per-splat values for the
// one gaussian index named in uniforms.shParams.w (\u22121 = disabled), so the
// GPU-computed math for a specific splat can be cross-checked on the CPU
// against hand-computed expected values from the raw source file.
@group(0) @binding(7) var<storage, read_write> debugOut : array<f32, 16>;

struct VOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) color         : vec4<f32>,
  // Raw \xB11 quad-corner UV. The quad geometry itself is pre-warped to align
  // with the 2-D covariance's eigenvectors (see cs_preprocess), so this UV
  // directly parameterizes the ellipse \u2014 no per-pixel conic/Mahalanobis math
  // needed.
  @location(1) uv            : vec2<f32>,
};

// Per-splat results of the heavy per-gaussian math (covariance projection,
// eigendecomposition, SH), computed once per splat by cs_preprocess instead
// of once per *vertex* (6x redundant \u2014 every vertex of a splat's quad used
// to redo the same work). vs_main just reads this and places the quad.
//   ndc.xy = screen-space center (NDC, pre-offset)
//   ndc.z  = NDC depth (clip.z/clip.w) \u2014 ndc.z > 1.5 marks a degenerate/
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
// get distinct binding numbers \u2014 WGSL doesn't allow one module-scope
// resource to change access mode per entry point, and read_write storage
// isn't permitted outside the compute stage anyway.
@group(0) @binding(8)  var<storage, read_write> splatGeomOut : array<SplatGeom>;
@group(0) @binding(9)  var<storage, read>       splatGeomIn  : array<SplatGeom>;

struct PreprocessParams { count : vec4<u32> };
@group(0) @binding(10) var<uniform> ppParams : PreprocessParams;

// \u2500\u2500 SH constants (match ply-loader.js) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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

// Uniform hue/sat/value tweak (see Uniforms.colorParams) \u2014 lets one splat
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

// Evaluate higher-order SH (degree 1\u20133) for Gaussian gi at direction dir.
// Returns the additive contribution to the DC color \u2014 caller clamps final sum.
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

// Quaternion (xyzw) \u2192 column-major rotation matrix
fn quatToMat3(q: vec4<f32>) -> mat3x3<f32> {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0*(y*y+z*z),  2.0*(x*y+w*z),       2.0*(x*z-w*y)),
    vec3<f32>(2.0*(x*y-w*z),         1.0 - 2.0*(x*x+z*z), 2.0*(y*z+w*x)),
    vec3<f32>(2.0*(x*z+w*y),         2.0*(y*z-w*x),       1.0 - 2.0*(x*x+y*y))
  );
}

// Writes the degenerate-splat sentinel (moved outside clip space \u2014 its quad
// gets clipped away entirely, matching the old per-vertex degen() early-out).
fn writeDegenerate(gi: u32) {
  var o: SplatGeom;
  o.ndc   = vec4<f32>(0.0, 0.0, 2.0, 0.0);
  o.v1    = vec2<f32>(0.0);
  o.v2    = vec2<f32>(0.0);
  o.color = vec4<f32>(0.0);
  splatGeomOut[gi] = o;
}

// \u2500\u2500 Per-splat preprocess (compute) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
// Runs the heavy per-gaussian math \u2014 covariance projection, eigendecomposition,
// SH evaluation \u2014 exactly once per splat (not once per vertex). vs_main used
// to redo all of this 6x per splat (once per quad vertex); on weaker mobile
// GPUs that redundant vertex-shader work, not fragment/overdraw cost, turned
// out to be the dominant per-frame cost (lowering render resolution didn't
// help fps at all, which only makes sense if the bottleneck was vertex-bound).
@compute @workgroup_size(64)
fn cs_preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gi = gid.x;
  if gi >= ppParams.count.x { return; }

  let g = gaussians[gi];

  // \u2500\u2500 Part transform: local \u2192 world \u2192 view \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  let partId    = u32(g.part);
  let partMat   = transforms[partId];
  let worldPos4 = partMat * vec4<f32>(g.pos, 1.0);
  let viewPos4  = uniforms.view * worldPos4;
  let t = viewPos4.xyz;

  // Discard if at or behind near plane
  let near = uniforms.params.y;
  if t.z > -near { writeDegenerate(gi); return; }

  // Near-depth fade: fade out splats within 3\xD7 the near distance so that
  // close-up splats don't cover the entire screen (matches Blender's behaviour).
  let depth     = -t.z;
  let nearFade  = clamp((depth - near) / (near * 2.0), 0.0, 1.0);

  // \u2500\u2500 Mask volumes: fade splats outside assigned volume boxes \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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

  // \u2500\u2500 3-D covariance \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  let splatScale = uniforms.params.x;
  let s = g.scale * (splatScale * maskFade);
  let R = quatToMat3(g.quat);
  // M = R * diag(s); Cov3D = M * M\u1D40
  let M    = mat3x3<f32>(R[0]*s.x, R[1]*s.y, R[2]*s.z);
  let cov3 = M * transpose(M);

  // \u2500\u2500 Project covariance to 2-D \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  // Perspective Jacobian at view-space point t (camera looks down \u2013Z)
  let tz2 = t.z * t.z;
  let fx  = uniforms.focal.x;
  let fy  = uniforms.focal.y;
  let J = mat3x3<f32>(
    vec3<f32>(fx / (-t.z),  0.0,           0.0),
    vec3<f32>(0.0,          fy / (-t.z),   0.0),
    vec3<f32>(fx*t.x / tz2, fy*t.y / tz2, 0.0)
  );
  // Combined rotation: view \xD7 part \u2014 maps splat-local covariance to view space.
  // For single-part (identity partMat) this reduces to just the view rotation.
  let R_part = mat3x3<f32>(partMat[0].xyz, partMat[1].xyz, partMat[2].xyz);
  let W_view = mat3x3<f32>(uniforms.view[0].xyz, uniforms.view[1].xyz, uniforms.view[2].xyz);
  let W      = W_view * R_part;
  let T      = J * W;
  let cov2   = T * cov3 * transpose(T);

  // Extract 2\xD72 + low-pass filter (anti-aliasing dilation)
  let aaDil  = uniforms.shParams.z;
  let detRaw = cov2[0][0] * cov2[1][1] - cov2[0][1] * cov2[0][1];  // pre-dilation det(\u03A3)
  let a = cov2[0][0] + aaDil;  // \u03A3_xx  (cov2[col][row])
  let b = cov2[0][1];           // \u03A3_xy
  let c = cov2[1][1] + aaDil;  // \u03A3_yy

  let det = a*c - b*b;
  if det < 1e-4 { writeDegenerate(gi); return; }

  // Mip-Splatting opacity compensation: dilating the covariance for AA
  // lowers the Gaussian's peak density (det grows), which silently dims
  // every splat \u2014 small/thin splats (fine surface detail) the most, since
  // aaDil is a larger fraction of their det. Rescale alpha by sqrt(det(\u03A3)/det(\u03A3+aaDil\xB7I))
  // to preserve the splat's original peak opacity under the dilated kernel.
  let aaFactor = sqrt(max(detRaw / det, 0.0));

  // \u2500\u2500 Eigen-decomposition of the 2\xD72 covariance \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  // Quad geometry is aligned to the covariance's eigenvectors (matches
  // PlayCanvas/SuperSplat's actual gsplat shader) so each splat rasterizes
  // only the minimal area its ellipse actually covers \u2014 an axis-aligned
  // isotropic bound (sized to the larger eigenvalue in both screen axes) was
  // tried instead to dodge a suspected eigenvector-orientation instability,
  // but that didn't fix anything and meaningfully increased overdraw on
  // anisotropic splats (every elongated splat rasterizing a much bigger
  // square than it needs), which is a likely contributor to the overall
  // softness compared to SuperSplat.
  //
  // Degenerate (near edge-on, disk-like) splats \u2014 where the minor-axis
  // variance goes non-positive \u2014 are discarded outright, matching
  // GaussianSplats3D's eigenValue2 <= 0.0 early return, rather than forced
  // to a minimum width. A previous version floored lambda2 to 0.1, which
  // renders these as a faint but visible ~1px sliver instead of nothing \u2014
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

  // Clamp the major-axis extent: splats larger than radiusCap \xD7 viewport
  // height are faded out and capped, preventing a handful of close-up splats
  // from overdrawing the whole screen. radiusCap shrinks adaptively under
  // load \u2014 see Viewer#_updateAdaptiveQuality. l1/l2 are corner-to-center
  // offset magnitudes (already radii, not diameters \u2014 see v1/v2 below).
  let maxRadius = uniforms.viewport.y * uniforms.params.w;
  let extent    = max(l1, l2);
  // Same factor both shrinks the quad geometry and fades its alpha \u2014 capped
  // splats fade out as they're clamped, rather than abruptly losing their
  // true aspect ratio.
  let sizeFade = clamp(maxRadius / max(extent, 1.0), 0.0, 1.0);

  // \u2500\u2500 View-dependent SH colour \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  // Camera world-space position from the view matrix:
  //   view = [R | -R*cam], so cam = -R\u1D40 * t  where t = view[3].xyz
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
  // Single clamp after the full DC + SH-rest sum \u2014 g.color.rgb (the DC term)
  // is intentionally left unclamped by the loader, since clamping it before
  // adding the higher-order terms would bake in the wrong value.
  rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));

  // Uniform hue/sat/value tweak (see Uniforms.colorParams) \u2014 a uniform branch
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

  // TEMP debug instrumentation \u2014 see binding(7) declaration above. Writable
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

  // corner (\xB11,\xB11) is split across the two eigenvector axes (v1, v2) instead
  // of a single axis-aligned radius \u2014 this is what lets fs_main use a plain
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
  // Plain Gaussian falloff \u2014 matches the original 3DGS reference renderer and
  // mkkellogg/GaussianSplats3D (an independent, widely-used implementation),
  // not PlayCanvas's hard-edged normExp, which turned out to be a
  // PlayCanvas-specific stylistic choice rather than a more "correct" shape.
  // in.uv (raw \xB11 quad corner) is built so dot(uv,uv)=1 at the edge
  // corresponds to a true Mahalanobis\xB2 of 8 (since the quad extent is
  // sqrt(8\xB7\u03BB) \u2014 see cs_preprocess), so the equivalent exp(-0.5\xB78\xB7A) = exp(-4\xB7A).
  let A = dot(in.uv, in.uv);
  if A > 1.0 { discard; }
  let alpha = in.color.a * exp(-4.0 * A);
  if alpha < 1.0/255.0 { discard; }
  let gamma = max(uniforms.params.z, 0.01);
  let rgb = pow(max(in.color.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / gamma));
  return vec4<f32>(rgb * alpha, alpha);
}
`;var ke=`

struct Uniforms {
  view       : mat4x4<f32>,
  proj       : mat4x4<f32>,
  viewport   : vec2<f32>,
  focal      : vec2<f32>,
  params     : vec4<f32>,
};

struct Gaussian {
  pos   : vec3<f32>,
  part  : f32,
  color : vec4<f32>,
  scale : vec3<f32>,
  quat  : vec4<f32>,
};

struct SortParams {
  numElements   : u32,
  passIndex     : u32,
  numWorkgroups : u32,
  numChunks     : u32,
};

@group(0) @binding(0)  var<uniform>             uniforms   : Uniforms;
@group(0) @binding(1)  var<storage, read>       gaussians  : array<Gaussian>;
@group(0) @binding(2)  var<storage, read>       transforms : array<mat4x4<f32>>;
@group(0) @binding(3)  var<uniform>             sortParams : SortParams;
@group(0) @binding(4)  var<storage, read_write> keysA      : array<u32>;
@group(0) @binding(5)  var<storage, read_write> keysB      : array<u32>;
@group(0) @binding(6)  var<storage, read_write> idxA       : array<u32>;
@group(0) @binding(7)  var<storage, read_write> idxB       : array<u32>;
@group(0) @binding(8)  var<storage, read_write> histo      : array<u32>;
@group(0) @binding(9)  var<storage, read_write> blockPfx   : array<u32>;
@group(0) @binding(10) var<storage, read_write> globalPfx  : array<u32>;
@group(0) @binding(11) var<storage, read_write> chunkTotal : array<u32>;
@group(0) @binding(12) var<storage, read_write> chunkPfx   : array<u32>;

// \u2500\u2500 Ping-pong helpers: pass even -> A is source, B is dest; pass odd -> reversed \u2500\u2500

fn readKey(i: u32, p: u32) -> u32 {
  if (p % 2u == 0u) { return keysA[i]; } else { return keysB[i]; }
}
fn writeKey(i: u32, p: u32, v: u32) {
  if (p % 2u == 0u) { keysB[i] = v; } else { keysA[i] = v; }
}
fn readIdx(i: u32, p: u32) -> u32 {
  if (p % 2u == 0u) { return idxA[i]; } else { return idxB[i]; }
}
fn writeIdx(i: u32, p: u32, v: u32) {
  if (p % 2u == 0u) { idxB[i] = v; } else { idxA[i] = v; }
}

// \u2500\u2500 cs_depth_key: seed keys/indices from view-space depth \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

@compute @workgroup_size(256)
fn cs_depth_key(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sortParams.numElements) { return; }

  let g      = gaussians[i];
  let partId = u32(g.part);
  let world  = transforms[partId] * vec4<f32>(g.pos, 1.0);
  let view   = uniforms.view * world;

  // Flip all bits of the (always-negative) IEEE-754 depth so ascending u32
  // order matches back-to-front draw order - same trick as src/sorter.js.
  let bits = bitcast<u32>(view.z);
  keysA[i] = bits ^ 0xffffffffu;
  idxA[i]  = i;
}

// \u2500\u2500 cs_histogram: per-workgroup 256-bin histogram of the current byte \u2500\u2500\u2500\u2500\u2500\u2500\u2500

var<workgroup> localHist: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn cs_histogram(
  @builtin(global_invocation_id) gid:  vec3<u32>,
  @builtin(local_invocation_id)  lid:  vec3<u32>,
  @builtin(workgroup_id)         wgid: vec3<u32>,
) {
  atomicStore(&localHist[lid.x], 0u);
  workgroupBarrier();

  let i = gid.x;
  if (i < sortParams.numElements) {
    let key = readKey(i, sortParams.passIndex);
    let bin = (key >> (sortParams.passIndex * 8u)) & 0xffu;
    atomicAdd(&localHist[bin], 1u);
  }
  workgroupBarrier();

  histo[wgid.x * 256u + lid.x] = atomicLoad(&localHist[lid.x]);
}

// \u2500\u2500 cs_scan_reduce: per-chunk per-bin exclusive prefixes + chunk totals \u2500\u2500\u2500\u2500\u2500
//
// Workgroups of histogram blocks are grouped into chunks of 256. Each
// dispatched workgroup (one per chunk) computes, for its chunk, the
// chunk-local exclusive prefix sum of histo[] into blockPfx[] (relative to
// the chunk's first block) plus the chunk's total per bin (chunkTotal[]).
// Bounded to 256 iterations per thread regardless of scene size.

@compute @workgroup_size(256)
fn cs_scan_reduce(
  @builtin(local_invocation_id) lid:  vec3<u32>,
  @builtin(workgroup_id)        wgid: vec3<u32>,
) {
  let bin = lid.x;
  let c   = wgid.x;
  let nwg = sortParams.numWorkgroups;
  var running: u32 = 0u;
  for (var j = 0u; j < 256u; j = j + 1u) {
    let wg = c * 256u + j;
    if (wg < nwg) {
      blockPfx[wg * 256u + bin] = running;
      running = running + histo[wg * 256u + bin];
    }
  }
  chunkTotal[c * 256u + bin] = running;
}

// \u2500\u2500 cs_scan_global: scan chunk totals + compute global per-bin offsets \u2500\u2500\u2500\u2500\u2500
//
// Single workgroup. Each thread (one per bin) scans chunkTotal across all
// chunks into chunkPfx (bounded to numChunks <= 256 iterations even for
// huge scenes), then thread 0 scans the 256 per-bin totals into globalPfx.

var<workgroup> binTotal: array<u32, 256>;

@compute @workgroup_size(256)
fn cs_scan_global(@builtin(local_invocation_id) lid: vec3<u32>) {
  let bin = lid.x;
  let numChunks = sortParams.numChunks;
  var running: u32 = 0u;
  for (var c = 0u; c < numChunks; c = c + 1u) {
    chunkPfx[c * 256u + bin] = running;
    running = running + chunkTotal[c * 256u + bin];
  }
  binTotal[bin] = running;
  workgroupBarrier();

  if (bin == 0u) {
    var acc: u32 = 0u;
    for (var b = 0u; b < 256u; b = b + 1u) {
      globalPfx[b] = acc;
      acc = acc + binTotal[b];
    }
  }
}

// \u2500\u2500 cs_scan_combine: fold chunk offsets into blockPfx \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
//
// blockPfx[] currently holds chunk-local exclusive prefixes (from
// cs_scan_reduce); add each block's chunk offset (chunkPfx) to make it
// globally-relative. One thread per (workgroup, bin) - O(1) each.

@compute @workgroup_size(256)
fn cs_scan_combine(
  @builtin(local_invocation_id) lid:  vec3<u32>,
  @builtin(workgroup_id)        wgid: vec3<u32>,
) {
  let bin   = lid.x;
  let wg    = wgid.x;
  let chunk = wg / 256u;
  blockPfx[wg * 256u + bin] = blockPfx[wg * 256u + bin] + chunkPfx[chunk * 256u + bin];
}

// \u2500\u2500 cs_scatter: stable scatter to sorted positions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

var<workgroup> wgBin:  array<u32, 256>;
var<workgroup> wgRank: array<u32, 256>;
var<workgroup> counts: array<u32, 256>;

@compute @workgroup_size(256)
fn cs_scatter(
  @builtin(global_invocation_id) gid:  vec3<u32>,
  @builtin(local_invocation_id)  lid:  vec3<u32>,
  @builtin(workgroup_id)         wgid: vec3<u32>,
) {
  let i = gid.x;
  let p = sortParams.passIndex;
  let inBounds = i < sortParams.numElements;

  var key: u32 = 0u;
  var idx: u32 = 0u;
  var bin: u32 = 256u; // sentinel for out-of-bounds lanes
  if (inBounds) {
    key = readKey(i, p);
    idx = readIdx(i, p);
    bin = (key >> (p * 8u)) & 0xffu;
  }
  wgBin[lid.x]  = bin;
  counts[lid.x] = 0u;
  workgroupBarrier();

  // Single-threaded, in-order pass over this workgroup's 256 lanes: assigns
  // each element a stable rank among same-bin elements (required so each
  // 8-bit pass preserves the ordering established by previous passes).
  // counts lives in workgroup memory (zeroed above by all 256 lanes in
  // parallel) rather than a per-invocation private array, which would be
  // allocated 256x over on some mobile GPU drivers.
  if (lid.x == 0u) {
    for (var k = 0u; k < 256u; k = k + 1u) {
      let b = wgBin[k];
      if (b < 256u) {
        wgRank[k] = counts[b];
        counts[b] = counts[b] + 1u;
      }
    }
  }
  workgroupBarrier();

  if (inBounds) {
    let dst = globalPfx[bin] + blockPfx[wgid.x * 256u + bin] + wgRank[lid.x];
    writeKey(dst, p, key);
    writeIdx(dst, p, idx);
  }
}
`;function Dt(a){return Math.ceil(a/256)}function te(a){return Math.ceil(Dt(a)/256)}var _s=`
@group(0) @binding(0) var hdrTex : texture_2d<f32>;
@group(0) @binding(1) var hdrSampler : sampler;

struct VOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv        : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VOut {
  const pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
  );
  var o: VOut;
  o.pos = vec4<f32>(pos[vi], 0.0, 1.0);
  o.uv  = (pos[vi] * vec2<f32>(0.5, -0.5)) + vec2<f32>(0.5, 0.5);
  return o;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  return textureSample(hdrTex, hdrSampler, in.uv);
}
`,gs=0,ys=16,Me=32,Ce=34,lt=36,kt=40,Ae=42,Mt=44,Fe=48,Rt=class{constructor(t,e){this.canvas=t,this.background=ze(e),this.device=null,this.context=null,this.pipeline=null,this.bindGroup=null,this._mainModule=null,this._hdrFormat="rgba16float",this._hdrTexture=null,this._hdrView=null,this._blitPipeline=null,this._blitSampler=null,this._blitBindGroup=null,this._uniformBuf=null,this._gaussianBuf=null,this._orderBuf=null,this._transformBuf=null,this._maskVolBuf=null,this._partVolMaskBuf=null,this._splatGeomBuf=null,this._preprocessParamsBuf=null,this._preprocessPipeline=null,this._preprocessBindGroup=null,this._preprocessCountData=new Uint32Array(4),this._shBuf=null,this._shDummyBuf=null,this._shNumBases=0,this._keysBufA=null,this._keysBufB=null,this._idxBufB=null,this._histoBuf=null,this._blockPfxBuf=null,this._globalPfxBuf=null,this._chunkTotalBuf=null,this._chunkPfxBuf=null,this._sortParamsBuf=null,this._sortParamsStaging=null,this._sortParamsData=new Uint32Array(20),this._depthKeyPipeline=null,this._histogramPipeline=null,this._scanReducePipeline=null,this._scanGlobalPipeline=null,this._scanCombinePipeline=null,this._scatterPipeline=null,this._depthKeyBindGroup=null,this._histogramBindGroup=null,this._scanReduceBindGroup=null,this._scanGlobalBindGroup=null,this._scanCombineBindGroup=null,this._scatterBindGroup=null,this._gpuSortFailed=!1,this._uniforms=new Float32Array(Fe),this._uniformsU32=new Uint32Array(this._uniforms.buffer),this._uniforms[lt]=1.08,this._uniforms[lt+2]=1,this._uniforms[lt+3]=1,this._uniforms[Ae]=.3,this._uniforms[Mt+1]=1,this._uniforms[Mt+2]=1,this._numSplats=0,this._maskWarned=new Set}async init(){if(!navigator.gpu)throw new Error("WebGPU is not supported in this browser.");let t=await navigator.gpu.requestAdapter();if(!t)throw new Error("No WebGPU adapter found.");this.device=await t.requestDevice({requiredLimits:{maxStorageBufferBindingSize:t.limits.maxStorageBufferBindingSize,maxBufferSize:t.limits.maxBufferSize}}),this.device.addEventListener("uncapturederror",s=>{console.error("[HoloSplat] GPU error:",s.error),this._gpuSortFailed=!0}),this.device.lost.then(s=>{console.error("[HoloSplat] WebGPU device lost:",s.reason,s.message)}),this.context=this.canvas.getContext("webgpu"),this._format=navigator.gpu.getPreferredCanvasFormat(),this.context.configure({device:this.device,format:this._format,alphaMode:"premultiplied"}),this._createPipeline(),this._createBlitPipeline(),this._createSortPipelines(),this._createPreprocessPipeline(),this._uniformBuf=this._createBuffer(Fe*4,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),this._preprocessParamsBuf=this._createBuffer(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),this._shDummyBuf=this._createBuffer(12,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),this._debugBuf=this._createBuffer(16*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),this._debugReadBuf=this._createBuffer(16*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),this._uniforms[kt+3]=-1,this._globalPfxBuf=this._createBuffer(256*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),this._sortParamsBuf=this._createBuffer(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),this._sortParamsStaging=this._createBuffer(16*5,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);let e=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);this._transformBuf=this._createBuffer(64,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),this.device.queue.writeBuffer(this._transformBuf,0,e),this._maskVolBuf=this._createBuffer(656,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),this._partVolMaskBuf=this._createBuffer(4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST)}uploadGaussians(t,e){this._numSplats=e,this._gaussianBuf=this.device.createBuffer({size:t.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this._gaussianBuf,0,t),this._orderBuf=this.device.createBuffer({size:e*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}),this._splatGeomBuf=this.device.createBuffer({size:Math.max(e*48,48),usage:GPUBufferUsage.STORAGE});let s=n=>this.device.createBuffer({size:Math.max(n*4,4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});this._keysBufA=s(e),this._keysBufB=s(e),this._idxBufB=s(e);let i=Dt(e),r=te(e);this._histoBuf=s(i*256),this._blockPfxBuf=s(i*256),this._chunkTotalBuf=s(r*256),this._chunkPfxBuf=s(r*256),this._rebuildBindGroup(),this._rebuildPreprocessBindGroup(),this._rebuildSortBindGroups()}updateUniforms({view:t,proj:e,width:s,height:i,focal:r,near:n=.01,radiusCap:o}){let l=this._uniforms;l.set(t,gs),l.set(e,ys),l[Me]=s,l[Me+1]=i,l[Ce]=r,l[Ce+1]=r,l[lt+1]=n,o!=null&&(l[lt+3]=o),this.device.queue.writeBuffer(this._uniformBuf,0,l)}updateOrder(t,e){let s=Math.min(e,this._orderBuf.size/4,t.length);s<e&&console.warn(`[HoloSplat] updateOrder: requested count (${e}) exceeds available buffer/source size (${s}) \u2014 clamping this frame's sort update`),!(s<=0)&&this.device.queue.writeBuffer(this._orderBuf,0,t.buffer,0,s*4)}uploadTransforms(t){let e=new Float32Array(t.length*16);for(let s=0;s<t.length;s++)e.set(t[s],s*16);this._transformBuf=this._createBuffer(e.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),this.device.queue.writeBuffer(this._transformBuf,0,e),this._gaussianBuf&&(this._rebuildPreprocessBindGroup(),this._rebuildSortBindGroups())}updateTransforms(t){this._transformBuf&&this.device.queue.writeBuffer(this._transformBuf,0,t)}uploadPartVolumeMask(t){this._partVolMaskBuf=this._createBuffer(Math.max(t.byteLength,4),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),this.device.queue.writeBuffer(this._partVolMaskBuf,0,t),this._gaussianBuf&&this._rebuildPreprocessBindGroup()}updateMaskVolumes(t){let e=Math.min(t.length,8),s=new ArrayBuffer(656),i=new Uint32Array(s),r=new Float32Array(s);i[0]=e;for(let n=0;n<e;n++){let l=16+n*80>>2,c=t[n],u=ee(c.matrix);u?r.set(u,l):this._maskWarned.has(c.name)||(this._maskWarned.add(c.name),console.warn(`[HoloSplat] mask volume "${c.name}" has a non-invertible (degenerate) transform \u2014 it will have no effect. Check that its Blender object has non-zero scale on all 3 axes.`)),r[l+16]=c.softEdge??.05}this.device.queue.writeBuffer(this._maskVolBuf,0,s)}patchGaussians(t,e){this._gaussianBuf&&this.device.queue.writeBuffer(this._gaussianBuf,e*64,t)}setSplatScale(t){this._uniforms[lt]=t}setGamma(t){this._uniforms[lt+2]=t}uploadSH(t,e){this._shBuf=null,this._shNumBases=e||0,this._uniforms[kt+1]=this._shNumBases,t&&this._shNumBases>0&&(this._shBuf=this._createBuffer(t.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),this.device.queue.writeBuffer(this._shBuf,0,t)),this._gaussianBuf&&this._rebuildPreprocessBindGroup()}allocateSH(t,e){this._shBuf=null,this._shNumBases=e||0,this._uniforms[kt+1]=this._shNumBases,this._shNumBases>0&&t>0&&(this._shBuf=this._createBuffer(t*this._shNumBases*12,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST)),this._gaussianBuf&&this._rebuildPreprocessBindGroup()}patchSH(t,e){this._shBuf&&t&&this.device.queue.writeBuffer(this._shBuf,e*this._shNumBases*12,t)}setShDegree(t){this._uniforms[kt]=t}setAaDilation(t){this._uniforms[Ae]=t}setHueShift(t){this._uniforms[Mt]=t/360%1}setSaturation(t){this._uniforms[Mt+1]=t}setValue(t){this._uniforms[Mt+2]=t}setDebugIndex(t){this._uniforms[kt+3]=t}async readDebug(){let t=this.device.createCommandEncoder();t.copyBufferToBuffer(this._debugBuf,0,this._debugReadBuf,0,16*4),this.device.queue.submit([t.finish()]),await this._debugReadBuf.mapAsync(GPUMapMode.READ);let e=new Float32Array(this._debugReadBuf.getMappedRange().slice(0));this._debugReadBuf.unmap();let[s,i,r,n,o,l,c,u,p,f,d,h,m,b,y,g]=e;return{tz:s,a:i,b:r,c:n,det:o,lambda1:l,lambda2:c,extent:u,sizeFade:p,nearFade:f,aaFactor:d,finalAlpha:h,rgb:[m,b,y],worldX:g}}setBackground(t){this.background=ze(t)}preprocess(t){if(!t||!this._preprocessBindGroup)return;this._preprocessCountData[0]=t,this.device.queue.writeBuffer(this._preprocessParamsBuf,0,this._preprocessCountData);let e=this.device.createCommandEncoder(),s=e.beginComputePass();s.setPipeline(this._preprocessPipeline),s.setBindGroup(0,this._preprocessBindGroup),s.dispatchWorkgroups(Math.ceil(t/64)),s.end(),this.device.queue.submit([e.finish()])}draw(t=this._numSplats){if(!t||!this.bindGroup)return;this._ensureHdrTexture();let e=this.device.createCommandEncoder(),s=e.beginRenderPass({colorAttachments:[{view:this._hdrView,clearValue:this.background,loadOp:"clear",storeOp:"store"}]});s.setPipeline(this.pipeline),s.setBindGroup(0,this.bindGroup),s.draw(6,t,0,0),s.end();let i=e.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),loadOp:"load",storeOp:"store"}]});i.setPipeline(this._blitPipeline),i.setBindGroup(0,this._blitBindGroup),i.draw(3,1,0,0),i.end(),this.device.queue.submit([e.finish()])}_ensureHdrTexture(){let t=this.canvas.width,e=this.canvas.height;this._hdrTexture&&this._hdrTexture.width===t&&this._hdrTexture.height===e||(this._hdrTexture?.destroy(),this._hdrTexture=this.device.createTexture({size:[t,e],format:this._hdrFormat,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING}),this._hdrView=this._hdrTexture.createView(),this._blitBindGroup=this.device.createBindGroup({layout:this._blitPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:this._hdrView},{binding:1,resource:this._blitSampler}]}))}destroy(){this._hdrTexture?.destroy(),this._debugBuf?.destroy(),this._debugReadBuf?.destroy(),this._uniformBuf?.destroy(),this._gaussianBuf?.destroy(),this._orderBuf?.destroy(),this._transformBuf?.destroy(),this._maskVolBuf?.destroy(),this._partVolMaskBuf?.destroy(),this._keysBufA?.destroy(),this._keysBufB?.destroy(),this._idxBufB?.destroy(),this._histoBuf?.destroy(),this._blockPfxBuf?.destroy(),this._globalPfxBuf?.destroy(),this._chunkTotalBuf?.destroy(),this._chunkPfxBuf?.destroy(),this._sortParamsBuf?.destroy(),this._sortParamsStaging?.destroy(),this._shBuf?.destroy(),this._shDummyBuf?.destroy(),this._splatGeomBuf?.destroy(),this._preprocessParamsBuf?.destroy(),this.context?.unconfigure()}_createPipeline(){this._mainModule=this._mainModule??this.device.createShaderModule({code:Jt});let t=this._mainModule;this.pipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:t,entryPoint:"vs_main"},fragment:{module:t,entryPoint:"fs_main",targets:[{format:this._hdrFormat,blend:{color:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}}]},primitive:{topology:"triangle-list",cullMode:"none"}})}_createBlitPipeline(){let t=this.device.createShaderModule({code:_s});this._blitPipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:t,entryPoint:"vs_main"},fragment:{module:t,entryPoint:"fs_main",targets:[{format:this._format}]},primitive:{topology:"triangle-list"}}),this._blitSampler=this.device.createSampler({magFilter:"linear",minFilter:"linear"})}_createBuffer(t,e){return this.device.createBuffer({size:t,usage:e})}_createPreprocessPipeline(){this._mainModule=this._mainModule??this.device.createShaderModule({code:Jt});let t=this._mainModule;this._preprocessPipeline=this.device.createComputePipeline({layout:"auto",compute:{module:t,entryPoint:"cs_preprocess"}})}_createSortPipelines(){let t=this.device.createShaderModule({code:ke}),e=s=>this.device.createComputePipeline({layout:"auto",compute:{module:t,entryPoint:s}});this._depthKeyPipeline=e("cs_depth_key"),this._histogramPipeline=e("cs_histogram"),this._scanReducePipeline=e("cs_scan_reduce"),this._scanGlobalPipeline=e("cs_scan_global"),this._scanCombinePipeline=e("cs_scan_combine"),this._scatterPipeline=e("cs_scatter")}_rebuildBindGroup(){this.bindGroup=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this._uniformBuf}},{binding:2,resource:{buffer:this._orderBuf}},{binding:9,resource:{buffer:this._splatGeomBuf}}]})}_rebuildPreprocessBindGroup(){this._preprocessBindGroup=this.device.createBindGroup({layout:this._preprocessPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this._uniformBuf}},{binding:1,resource:{buffer:this._gaussianBuf}},{binding:3,resource:{buffer:this._transformBuf}},{binding:4,resource:{buffer:this._maskVolBuf}},{binding:5,resource:{buffer:this._partVolMaskBuf}},{binding:6,resource:{buffer:this._shBuf??this._shDummyBuf}},{binding:7,resource:{buffer:this._debugBuf}},{binding:8,resource:{buffer:this._splatGeomBuf}},{binding:10,resource:{buffer:this._preprocessParamsBuf}}]})}_rebuildSortBindGroups(){let t={0:{buffer:this._uniformBuf},1:{buffer:this._gaussianBuf},2:{buffer:this._transformBuf},3:{buffer:this._sortParamsBuf},4:{buffer:this._keysBufA},5:{buffer:this._keysBufB},6:{buffer:this._orderBuf},7:{buffer:this._idxBufB},8:{buffer:this._histoBuf},9:{buffer:this._blockPfxBuf},10:{buffer:this._globalPfxBuf},11:{buffer:this._chunkTotalBuf},12:{buffer:this._chunkPfxBuf}},e=(s,i)=>this.device.createBindGroup({layout:s.getBindGroupLayout(0),entries:i.map(r=>({binding:r,resource:t[r]}))});this._depthKeyBindGroup=e(this._depthKeyPipeline,[0,1,2,3,4,6]),this._histogramBindGroup=e(this._histogramPipeline,[3,4,5,8]),this._scanReduceBindGroup=e(this._scanReducePipeline,[3,8,9,11]),this._scanGlobalBindGroup=e(this._scanGlobalPipeline,[3,10,11,12]),this._scanCombineBindGroup=e(this._scanCombinePipeline,[9,12]),this._scatterBindGroup=e(this._scatterPipeline,[3,4,5,6,7,9,10])}runGpuSort(t){if(!t)return;let e=Dt(t),s=te(t),i=this._sortParamsData;for(let r=0;r<5;r++){let n=r*4;i[n]=t,i[n+1]=r===0?0:r-1,i[n+2]=e,i[n+3]=s}this.device.queue.writeBuffer(this._sortParamsStaging,0,i);{let r=this.device.createCommandEncoder();r.copyBufferToBuffer(this._sortParamsStaging,0,this._sortParamsBuf,0,16);let n=r.beginComputePass();n.setPipeline(this._depthKeyPipeline),n.setBindGroup(0,this._depthKeyBindGroup),n.dispatchWorkgroups(e),n.end(),this.device.queue.submit([r.finish()])}for(let r=0;r<4;r++){let n=this.device.createCommandEncoder();n.copyBufferToBuffer(this._sortParamsStaging,(r+1)*16,this._sortParamsBuf,0,16);let o=n.beginComputePass();o.setPipeline(this._histogramPipeline),o.setBindGroup(0,this._histogramBindGroup),o.dispatchWorkgroups(e),o.end();let l=n.beginComputePass();l.setPipeline(this._scanReducePipeline),l.setBindGroup(0,this._scanReduceBindGroup),l.dispatchWorkgroups(s),l.end();let c=n.beginComputePass();c.setPipeline(this._scanGlobalPipeline),c.setBindGroup(0,this._scanGlobalBindGroup),c.dispatchWorkgroups(1),c.end();let u=n.beginComputePass();u.setPipeline(this._scanCombinePipeline),u.setBindGroup(0,this._scanCombineBindGroup),u.dispatchWorkgroups(e),u.end();let p=n.beginComputePass();p.setPipeline(this._scatterPipeline),p.setBindGroup(0,this._scatterBindGroup),p.dispatchWorkgroups(e),p.end(),this.device.queue.submit([n.finish()])}}async debugReadOrder(t){let e=t*4,s=this.device.createBuffer({size:e,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),i=this.device.createCommandEncoder();i.copyBufferToBuffer(this._orderBuf,0,s,0,e),this.device.queue.submit([i.finish()]),await s.mapAsync(GPUMapMode.READ);let r=new Uint32Array(s.getMappedRange()).slice();return s.unmap(),s.destroy(),r}};function ee(a){let t=a[0],e=a[1],s=a[2],i=a[3],r=a[4],n=a[5],o=a[6],l=a[7],c=a[8],u=a[9],p=a[10],f=a[11],d=a[12],h=a[13],m=a[14],b=a[15],y=t*n-e*r,g=t*o-s*r,S=t*l-i*r,_=e*o-s*n,v=e*l-i*n,B=s*l-i*o,x=c*h-u*d,P=c*m-p*d,k=c*b-f*d,U=u*m-p*h,I=u*b-f*h,V=p*b-f*m,E=y*V-g*I+S*U+_*k-v*P+B*x;if(!E)return null;let T=1/E,F=new Float32Array(16);return F[0]=(n*V-o*I+l*U)*T,F[1]=(s*I-e*V-i*U)*T,F[2]=(h*B-m*v+b*_)*T,F[3]=(p*v-u*B-f*_)*T,F[4]=(o*k-r*V-l*P)*T,F[5]=(t*V-s*k+i*P)*T,F[6]=(m*S-d*B-b*g)*T,F[7]=(c*B-p*S+f*g)*T,F[8]=(r*I-n*k+l*x)*T,F[9]=(e*k-t*I-i*x)*T,F[10]=(d*v-h*S+b*y)*T,F[11]=(u*S-c*v-f*y)*T,F[12]=(n*P-r*U-o*x)*T,F[13]=(t*U-e*P+s*x)*T,F[14]=(h*g-d*_-m*y)*T,F[15]=(c*_-u*g+p*y)*T,F}function ze(a){if(!a||a==="transparent")return{r:0,g:0,b:0,a:0};if(Array.isArray(a))return{r:a[0],g:a[1],b:a[2],a:a[3]??1};if(typeof a=="string"){let t=a.replace("#","");if(t.length===6)return{r:parseInt(t.slice(0,2),16)/255,g:parseInt(t.slice(2,4),16)/255,b:parseInt(t.slice(4,6),16)/255,a:1};if(t.length===8)return{r:parseInt(t.slice(0,2),16)/255,g:parseInt(t.slice(2,4),16)/255,b:parseInt(t.slice(4,6),16)/255,a:parseInt(t.slice(6,8),16)/255}}return{r:0,g:0,b:0,a:1}}var Gt=class{constructor({fov:t=60,near:e=.01,far:s=2e3}={}){this.fov=t*Math.PI/180,this.near=e,this.far=s,this.theta=0,this.phi=.2,this.radius=5,this.target=[0,0,0],this.enabled=!0,this.orbitEnabled=!0,this.panEnabled=!0,this.panSpeed=1,this.panButton=2,this.panRadius=null,this.panOrigin=null,this.thetaMin=null,this.thetaMax=null,this.phiMin=null,this.phiMax=null,this.zoomEnabled=!0,this.radiusMin=null,this.radiusMax=null,this.orbitDeltaCallback=null,this.dragStartCallback=null,this.dragEndCallback=null,this.panDeltaCallback=null,this.zoomDeltaCallback=null,this._drag=null,this._touches=[],this._allowTouchScroll=!1,this.viewMatrix=new Float32Array(16),this.projMatrix=new Float32Array(16)}get allowTouchScroll(){return this._allowTouchScroll}set allowTouchScroll(t){this._allowTouchScroll=!!t,typeof document<"u"&&(document.documentElement.style.touchAction=t?"pan-y":"")}attach(t){this._canvas=t;let e=(s,i)=>t.contains(document.elementFromPoint(s,i));this._onMouseDown=s=>{e(s.clientX,s.clientY)&&this._mouseDown(s)},this._onMouseMove=s=>this._mouseMove(s),this._onMouseUp=()=>{this._drag=null,this.dragEndCallback?.()},this._onWheel=s=>this._wheel(s),this._onTouchStart=s=>{e(s.touches[0]?.clientX,s.touches[0]?.clientY)&&this._touchStart(s)},this._onTouchMove=s=>this._touchMove(s),this._onTouchEnd=s=>this._touchEnd(s),this._onCtxMenu=s=>{e(s.clientX,s.clientY)&&s.preventDefault()},this._onMouseOutDoc=s=>{s.relatedTarget===null&&this._drag&&(this._drag=null,this.dragEndCallback?.())},document.addEventListener("mousedown",this._onMouseDown),document.addEventListener("mousemove",this._onMouseMove),document.addEventListener("mouseup",this._onMouseUp),document.addEventListener("mouseout",this._onMouseOutDoc),t.addEventListener("wheel",this._onWheel,{passive:!1}),document.addEventListener("touchstart",this._onTouchStart,{passive:!1}),document.addEventListener("touchmove",this._onTouchMove,{passive:!1}),document.addEventListener("touchend",this._onTouchEnd),document.addEventListener("contextmenu",this._onCtxMenu)}detach(){let t=this._canvas;t&&(document.removeEventListener("mousedown",this._onMouseDown),document.removeEventListener("mousemove",this._onMouseMove),document.removeEventListener("mouseup",this._onMouseUp),document.removeEventListener("mouseout",this._onMouseOutDoc),t.removeEventListener("wheel",this._onWheel),document.removeEventListener("touchstart",this._onTouchStart),document.removeEventListener("touchmove",this._onTouchMove),document.removeEventListener("touchend",this._onTouchEnd),document.removeEventListener("contextmenu",this._onCtxMenu),this._canvas=null)}_mouseDown(t){this.enabled&&(t.button===2&&!this.panEnabled||(this._drag={x:t.clientX,y:t.clientY,button:t.button},t.button===0&&this.dragStartCallback?.(),t.preventDefault()))}_mouseMove(t){if(this._drag){let e=t.clientX-this._drag.x,s=t.clientY-this._drag.y;this._drag.x=t.clientX,this._drag.y=t.clientY;let i=this._drag.button===this.panButton;if(this.panEnabled&&(i||!this.orbitEnabled&&this._drag.button===0)){this._pan(e,s);return}this._drag.button===0&&!i&&this._orbit(e,s)}}_wheel(t){if(!this.enabled||!this.zoomEnabled)return;t.preventDefault();let e=t.deltaY>0?1.1:.9;if(this.zoomDeltaCallback){this.zoomDeltaCallback(e);return}this.radius=Math.max(.01,this.radius*e),this.radiusMin!==null&&(this.radius=Math.max(this.radiusMin,this.radius)),this.radiusMax!==null&&(this.radius=Math.min(this.radiusMax,this.radius))}_touchStart(t){this.enabled&&(this.allowTouchScroll&&t.touches.length<2||(t.preventDefault(),this._touches=Array.from(t.touches).map(e=>({id:e.identifier,x:e.clientX,y:e.clientY}))))}_touchMove(t){if(!this.enabled)return;if(this.allowTouchScroll&&t.touches.length<2){this._touches=[];return}t.preventDefault();let e=this._touches,s=Array.from(t.touches).map(i=>({id:i.identifier,x:i.clientX,y:i.clientY}));if(s.length===1&&e.length===1){let i=s[0].x-e[0].x,r=s[0].y-e[0].y;this.panEnabled&&(this.panButton===0||!this.orbitEnabled)?this._pan(i,r):this._orbit(i,r)}else if(s.length===2&&e.length===2){let i=Math.hypot(e[1].x-e[0].x,e[1].y-e[0].y),r=Math.hypot(s[1].x-s[0].x,s[1].y-s[0].y);if(i>0&&r>0&&this.zoomEnabled){let n=i/r;this.zoomDeltaCallback?this.zoomDeltaCallback(n):(this.radius=Math.max(.01,this.radius*n),this.radiusMin!==null&&(this.radius=Math.max(this.radiusMin,this.radius)),this.radiusMax!==null&&(this.radius=Math.min(this.radiusMax,this.radius)))}if(this.panEnabled){let n=(e[0].x+e[1].x)*.5,o=(e[0].y+e[1].y)*.5,l=(s[0].x+s[1].x)*.5,c=(s[0].y+s[1].y)*.5;this._pan(l-n,c-o)}}this._touches=s}_touchEnd(t){this._touches=Array.from(t.touches).map(e=>({id:e.identifier,x:e.clientX,y:e.clientY})),this._touches.length===0&&this.dragEndCallback?.()}_orbit(t,e){let i=-t*.005,r=e*.005;if(this.orbitDeltaCallback){this.orbitDeltaCallback(i,r);return}this.theta+=i,this.phi=Math.max(-Math.PI/2+.01,Math.min(Math.PI/2-.01,this.phi+r)),this.thetaMin!==null&&(this.theta=Math.max(this.thetaMin,this.theta)),this.thetaMax!==null&&(this.theta=Math.min(this.thetaMax,this.theta)),this.phiMin!==null&&(this.phi=Math.max(this.phiMin,this.phi)),this.phiMax!==null&&(this.phi=Math.min(this.phiMax,this.phi))}constrainAngles(t,e){if(t!==null){let s=t*Math.PI/180;this.thetaMin=this.theta-s,this.thetaMax=this.theta+s}else this.thetaMin=null,this.thetaMax=null;if(e!==null){let s=e*Math.PI/180;this.phiMin=Math.max(-Math.PI/2+.01,this.phi-s),this.phiMax=Math.min(Math.PI/2-.01,this.phi+s)}else this.phiMin=null,this.phiMax=null}clearConstraints(){this.thetaMin=null,this.thetaMax=null,this.phiMin=null,this.phiMax=null}disableZoom(){this.zoomEnabled=!1,this.radiusMin=null,this.radiusMax=null}constrainZoom(t,e){let s=e/100;this.zoomEnabled=!0,this.radiusMin=Math.max(.01,t*(1-s)),this.radiusMax=t*(1+s)}enableZoom(){this.zoomEnabled=!0,this.radiusMin=null,this.radiusMax=null}_pan(t,e){let s=this.radius*.001*this.panSpeed,i=this._cameraRight(),r=this._cameraUp(),n=-(i[0]*t-r[0]*e)*s,o=-(i[1]*t-r[1]*e)*s,l=-(i[2]*t-r[2]*e)*s;if(this.panDeltaCallback){this.panDeltaCallback(n,o,l);return}if(this.target[0]+=n,this.target[1]+=o,this.target[2]+=l,this.panRadius!==null&&this.panOrigin){let[c,u,p]=this.panOrigin,f=Math.hypot(this.target[0]-c,this.target[1]-u,this.target[2]-p);if(f>this.panRadius){let d=this.panRadius/f;this.target[0]=c+(this.target[0]-c)*d,this.target[1]=u+(this.target[1]-u)*d,this.target[2]=p+(this.target[2]-p)*d}}}_cameraRight(){return[this.viewMatrix[0],this.viewMatrix[4],this.viewMatrix[8]]}_cameraUp(){return[this.viewMatrix[1],this.viewMatrix[5],this.viewMatrix[9]]}update(t,e){let s=this._eye();vs(s,this.target,[0,1,0],this.viewMatrix),xs(this.fov,t/e,this.near,this.far,this.projMatrix)}get eye(){return this._eye()}_eye(){let t=Math.cos(this.phi),e=Math.sin(this.phi),s=Math.cos(this.theta),i=Math.sin(this.theta);return[this.target[0]+this.radius*t*i,this.target[1]+this.radius*e,this.target[2]+this.radius*t*s]}focalLength(t){return t*.5/Math.tan(this.fov*.5)}setFromLookAt(t,e){this.target=[e[0],e[1],e[2]];let s=t[0]-e[0],i=t[1]-e[1],r=t[2]-e[2];this.radius=Math.hypot(s,i,r)||.001,this.phi=Math.asin(Math.max(-1,Math.min(1,i/this.radius))),this.theta=Math.atan2(s,r)}fitScene(t,e){this._sceneBounds(t,e),this.theta=0,this.phi=.2}focusScene(t,e){this._sceneBounds(t,e)}_sceneBounds(t,e){let s=1/0,i=1/0,r=1/0,n=-1/0,o=-1/0,l=-1/0;for(let u=0;u<e;u++){let p=u*16,f=t[p],d=t[p+1],h=t[p+2];f<s&&(s=f),f>n&&(n=f),d<i&&(i=d),d>o&&(o=d),h<r&&(r=h),h>l&&(l=h)}this.target=[(s+n)*.5,(i+o)*.5,(r+l)*.5];let c=Math.max(n-s,o-i,l-r)*.5;this.radius=c/Math.tan(this.fov*.5)*1.2}};function vs(a,t,e,s){let[i,r,n]=a,[o,l,c]=t,[u,p,f]=e,d=i-o,h=r-l,m=n-c,b=Math.hypot(d,h,m);d/=b,h/=b,m/=b;let y=p*m-f*h,g=f*d-u*m,S=u*h-p*d,_=Math.hypot(y,g,S);y/=_,g/=_,S/=_;let v=h*S-m*g,B=m*y-d*S,x=d*g-h*y;s[0]=y,s[1]=v,s[2]=d,s[3]=0,s[4]=g,s[5]=B,s[6]=h,s[7]=0,s[8]=S,s[9]=x,s[10]=m,s[11]=0,s[12]=-(y*i+g*r+S*n),s[13]=-(v*i+B*r+x*n),s[14]=-(d*i+h*r+m*n),s[15]=1}function xs(a,t,e,s,i){let r=1/Math.tan(a*.5),n=e-s;i[0]=r/t,i[1]=0,i[2]=0,i[3]=0,i[4]=0,i[5]=r,i[6]=0,i[7]=0,i[8]=0,i[9]=0,i[10]=s/n,i[11]=-1,i[12]=0,i[13]=0,i[14]=e*s/n,i[15]=0}var bs=`
function radixSort32(keys, idx, tmp_k, tmp_i, counts, pfx, N) {
  for (let pass = 0; pass < 4; pass++) {
    const shift = pass * 8;
    counts.fill(0);
    for (let i = 0; i < N; i++) counts[(keys[i] >>> shift) & 0xff]++;
    pfx[0] = 0;
    for (let i = 1; i < 256; i++) pfx[i] = pfx[i-1] + counts[i-1];
    for (let i = 0; i < N; i++) {
      const b = (keys[i] >>> shift) & 0xff;
      tmp_k[pfx[b]]   = keys[i];
      tmp_i[pfx[b]++] = idx[i];
    }
    // swap
    const k = keys; keys = tmp_k; tmp_k = k;
    const x = idx;  idx  = tmp_i; tmp_i = x;
  }
  // after 4 passes (even) idx holds the result
  return idx;
}
`,ws=bs+`
const u32 = new Uint32Array(1);
const f32 = new Float32Array(u32.buffer);

let keys0, keys1, idx0, idx1, counts, pfx, allocN = -1;
function alloc(N) {
  if (N <= allocN) return;
  allocN = N;
  keys0  = new Uint32Array(N);
  keys1  = new Uint32Array(N);
  idx0   = new Uint32Array(N);
  idx1   = new Uint32Array(N);
  counts = new Uint32Array(256);
  pfx    = new Uint32Array(256);
}

self.onmessage = function(e) {
  const depths = e.data.depths;
  const gIndex = e.data.gIndex;
  const N      = e.data.N;
  alloc(N);

  // Convert negative floats \u2192 sortable uint32 by flipping all bits
  const dv = new DataView(depths.buffer);
  for (let i = 0; i < N; i++) {
    keys0[i] = dv.getUint32(i * 4, true) ^ 0xffffffff;
    idx0[i]  = gIndex ? gIndex[i] : i;
  }

  const sorted = radixSort32(keys0, idx0, keys1, idx1, counts, pfx, N);
  const result = sorted.slice(0, N);
  self.postMessage({ order: result }, [result.buffer]);
};
`;function Ee(a){let t=new Uint32Array(a),e=new Uint32Array(a),s=new Uint32Array(a),i=new Uint32Array(a),r=new Uint32Array(256),n=new Uint32Array(256);function o(l,c,u=null){let p=new DataView(l.buffer);for(let f=0;f<c;f++){let d=u?u[f]:f;t[f]=p.getUint32(d*4,!0)^4294967295,s[f]=d}for(let f=0;f<4;f++){let d=f*8;r.fill(0);for(let h=0;h<c;h++)r[t[h]>>>d&255]++;n[0]=0;for(let h=1;h<256;h++)n[h]=n[h-1]+r[h-1];for(let h=0;h<c;h++){let m=t[h]>>>d&255;e[n[m]]=t[h],i[n[m]++]=s[h]}[t,e]=[e,t],[s,i]=[i,s]}return s}return o}function _t(a){if(typeof Worker>"u")return Ee(a);let t;try{let r=new Blob([ws],{type:"application/javascript"}),n=URL.createObjectURL(r);t=new Worker(n),URL.revokeObjectURL(n)}catch{return Ee(a)}let e=new Uint32Array(a);for(let r=0;r<a;r++)e[r]=r;let s=!1,i=null;return t.onmessage=r=>{if(e=r.data.order,s=!1,i){let{depths:n,N:o,gIndex:l}=i;i=null,s=!0;let c=l?[n.buffer,l.buffer]:[n.buffer];t.postMessage({depths:n,N:o,gIndex:l},c)}},function(n,o,l=null){let c=new Float32Array(o);if(l)for(let p=0;p<o;p++)c[p]=n[l[p]];else c.set(n.subarray(0,o));let u=l?l.slice(0,o):null;if(s)i={depths:c,N:o,gIndex:u};else{s=!0;let p=u?[c.buffer,u.buffer]:[c.buffer];t.postMessage({depths:c,N:o,gIndex:u},p)}return e.length!==o?u&&u.length===o?u:Uint32Array.from({length:o},(p,f)=>f):e}}async function at(a,t){let e=await fetch(a);if(!e.ok)throw new Error(`HTTP ${e.status} loading ${a}`);let s=parseInt(e.headers.get("content-length")||"0",10),i=e.body.getReader(),r=[],n=0;for(;;){let{done:c,value:u}=await i.read();if(c)break;r.push(u),n+=u.byteLength,t&&s>0&&t(n/s)}let o=new Uint8Array(n),l=0;for(let c of r)o.set(c,l),l+=c.byteLength;return o.buffer}async function Te(a,t){let e=await at(a,t);return Ue(e)}function Ue(a){let e=Math.floor(a.byteLength/32);if(e===0)throw new Error("Empty or invalid .splat file");let s=new DataView(a),i=new Float32Array(e*16);for(let r=0;r<e;r++){let n=r*32,o=r*16;i[o+0]=s.getFloat32(n+0,!0),i[o+1]=s.getFloat32(n+4,!0),i[o+2]=s.getFloat32(n+8,!0),i[o+4]=s.getUint8(n+24)/255,i[o+5]=s.getUint8(n+25)/255,i[o+6]=s.getUint8(n+26)/255,i[o+7]=s.getUint8(n+27)/255,i[o+8]=s.getFloat32(n+12,!0),i[o+9]=s.getFloat32(n+16,!0),i[o+10]=s.getFloat32(n+20,!0);let l=(s.getUint8(n+28)-128)/128,c=(s.getUint8(n+29)-128)/128,u=(s.getUint8(n+30)-128)/128,p=(s.getUint8(n+31)-128)/128,f=Math.hypot(c,u,p,l)||1;i[o+12]=c/f,i[o+13]=u/f,i[o+14]=p/f,i[o+15]=l/f}return{data:i,count:e,shData:null,numSHBases:0}}var se=.28209479177387814;async function De(a,t,e=0){let s=await at(a,t);return Re(s,e)}function Re(a,t=0){let e=new Uint8Array(a),s=Oe(e);if(s<0)throw new Error("Invalid PLY: end_header not found");let{numVertices:i,propMap:r,stride:n,hasColor:o}=He(new TextDecoder().decode(e.subarray(0,s)));if(i===0)throw new Error("PLY file contains no vertices");let l=new DataView(a,s),{data:c,shData:u,numSHBases:p}=Ge(l,r,n,o,i,t);return{data:c,count:i,shData:u,numSHBases:p}}async function ie(a,t,e=0){let s=await fetch(a);if(!s.ok)throw new Error(`HTTP ${s.status} loading ${a}`);let i=parseInt(s.headers.get("content-length")||"0",10),r=s.body.getReader(),n=new Uint8Array(0),o=0;for(;;){let{done:l,value:c}=await r.read();if(l)throw new Error("PLY stream ended before end_header");o+=c.byteLength,t&&i>0&&t(o/i);let u=new Uint8Array(n.length+c.length);u.set(n),u.set(c,n.length),n=u;let p=Oe(n);if(p>=0){let f=He(new TextDecoder().decode(n.subarray(0,p))),d=n.slice(p),{numVertices:h,propMap:m,stride:b,hasColor:y}=f,g=m.f_rest_44?15:m.f_rest_23?8:m.f_rest_8?3:0,S=g>=15?3:g>=8?2:g>=3?1:0,_=Math.min(e,S),v=_>=3?15:_>=2?8:_>=1?3:0;return{numVertices:h,numSHBases:v,consume:async(x,P)=>{let k=d;for(;;){let{done:U,value:I}=await r.read();if(!U){o+=I.byteLength,P&&i>0&&P(o/i);let E=new Uint8Array(k.length+I.length);E.set(k),E.set(I,k.length),k=E}let V=Math.floor(k.length/b);if(V>0){let E=V*b,{data:T,shData:F}=Ge(new DataView(k.buffer,k.byteOffset,E),m,b,y,V,e);x(T,V,F),k=k.slice(E)}if(U)break}}}}if(n.length>65536)throw new Error("PLY header exceeds 64 KB")}}function Ge(a,t,e,s,i,r=0){let n=new Float32Array(i*16),o=t.f_rest_44?15:t.f_rest_23?8:t.f_rest_8?3:0,l=o>=15?3:o>=8?2:o>=3?1:0,c=Math.min(r,l),u=c>=3?15:c>=2?8:c>=1?3:0,p=u>0&&s,f=[],d=[],h=[];if(p){let b=o,y=2*o;for(let g=0;g<u;g++)f[g]=t[`f_rest_${g}`],d[g]=t[`f_rest_${b+g}`],h[g]=t[`f_rest_${y+g}`]}let m=p?new Float32Array(i*u*3):null;for(let b=0;b<i;b++){let y=b*e,g=b*16;if(n[g+0]=q(a,y,t.x),n[g+1]=q(a,y,t.y),n[g+2]=q(a,y,t.z),s){if(n[g+4]=.5+se*q(a,y,t.f_dc_0),n[g+5]=.5+se*q(a,y,t.f_dc_1),n[g+6]=.5+se*q(a,y,t.f_dc_2),p){let P=b*u*3;for(let k=0;k<u;k++){let U=P+k*3;m[U+0]=q(a,y,f[k]),m[U+1]=q(a,y,d[k]),m[U+2]=q(a,y,h[k])}}}else n[g+4]=1,n[g+5]=1,n[g+6]=1;n[g+7]=Ps(q(a,y,t.opacity)),n[g+8]=Math.exp(q(a,y,t.scale_0)),n[g+9]=Math.exp(q(a,y,t.scale_1)),n[g+10]=Math.exp(q(a,y,t.scale_2));let S=q(a,y,t.rot_0),_=q(a,y,t.rot_1),v=q(a,y,t.rot_2),B=q(a,y,t.rot_3),x=Math.hypot(_,v,B,S)||1;n[g+12]=_/x,n[g+13]=v/x,n[g+14]=B/x,n[g+15]=S/x}return{data:n,shData:m,numSHBases:u}}function Oe(a){let t=[101,110,100,95,104,101,97,100,101,114];t:for(let e=0;e<=a.length-t.length;e++){for(let i=0;i<t.length;i++)if(a[e+i]!==t[i])continue t;let s=e+t.length;return a[s]===13&&s++,a[s]===10&&s++,s}return-1}function He(a){let t=a.split(`
`),e=0,s=!1,i=[];for(let l of t){let c=l.trim().split(/\s+/);c[0]==="element"?(s=c[1]==="vertex",s&&(e=parseInt(c[2],10))):c[0]==="property"&&s&&i.push({type:c[1],name:c[2]})}let r={},n=0;for(let l of i)r[l.name]={offset:n,type:l.type},n+=Ss(l.type);let o=["x","y","z","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","opacity"];for(let l of o)if(!r[l])throw new Error(`PLY missing required property: ${l}`);return{numVertices:e,propMap:r,stride:n,hasColor:!!(r.f_dc_0&&r.f_dc_1&&r.f_dc_2)}}function Ss(a){switch(a){case"float":case"float32":case"int":case"uint":return 4;case"double":case"int64":case"uint64":return 8;case"short":case"ushort":case"int16":case"uint16":return 2;case"char":case"uchar":case"int8":case"uint8":return 1;default:return 4}}function q(a,t,e){let s=t+e.offset;switch(e.type){case"float":case"float32":return a.getFloat32(s,!0);case"double":return a.getFloat64(s,!0);case"int":case"int32":return a.getInt32(s,!0);case"uint":case"uint32":return a.getUint32(s,!0);case"short":case"int16":return a.getInt16(s,!0);case"ushort":case"uint16":return a.getUint16(s,!0);case"char":case"int8":return a.getInt8(s);case"uchar":case"uint8":return a.getUint8(s);default:return a.getFloat32(s,!0)}}function Ps(a){return 1/(1+Math.exp(-a))}var Ie=1347635022,Ve=1/Math.SQRT2;async function Le(a,t,e=0){let s=await at(a,t),i=await Ct(s);return ae(i,e)}async function Bs(a,t=0){let e=await Ct(a);return ae(e,t)}var ks=5e4;function Ms(){return new Promise(a=>setTimeout(a,0))}async function ae(a,t=0){let e=new DataView(a),s=e.getUint32(0,!0),i=e.getUint32(4,!0),r=e.getUint32(8,!0),n=e.getUint8(12),o=e.getUint8(13);if(s!==Ie)throw new Error(`Invalid .spz magic: 0x${s.toString(16).toUpperCase()} (expected 0x${Ie.toString(16).toUpperCase()})`);(i<2||i>4)&&console.warn(`HoloSplat: .spz version ${i} is untested; attempting load anyway`);let l=1<<o,c=i>=3?4:3,u=x=>x>=4?24:x>=3?15:x>=2?8:x>=1?3:0,p=u(n),f=Math.min(t,n,3),d=u(f),h=d>0,m=16,b=m+r*9,y=b+r*1,g=y+r*3,S=g+r*3,_=S+r*c,v=new Float32Array(r*16),B=h?new Float32Array(r*d*3):null;for(let x=0;x<r;x++){let P=x*16,k=m+x*9;v[P+0]=ct(e,k+0)/l,v[P+1]=ct(e,k+3)/l,v[P+2]=ct(e,k+6)/l;let U=y+x*3;v[P+4]=e.getUint8(U+0)/255,v[P+5]=e.getUint8(U+1)/255,v[P+6]=e.getUint8(U+2)/255,v[P+7]=e.getUint8(b+x)/255;let I=g+x*3;v[P+8]=Math.exp((e.getUint8(I+0)-128)/16),v[P+9]=Math.exp((e.getUint8(I+1)-128)/16),v[P+10]=Math.exp((e.getUint8(I+2)-128)/16);let V=S+x*c,E,T,F,Y;if(i>=3){let j=e.getUint32(V,!0),bt=j&3,W=Ve/512,N=gt(j>>2&1023)*W,L=gt(j>>12&1023)*W,Q=gt(j>>22&1023)*W,ot=Math.sqrt(Math.max(0,1-N*N-L*L-Q*Q));switch(bt){case 0:E=ot,T=N,F=L,Y=Q;break;case 1:E=N,T=ot,F=L,Y=Q;break;case 2:E=N,T=L,F=ot,Y=Q;break;default:E=N,T=L,F=Q,Y=ot}}else E=e.getInt8(V+0)/128,T=e.getInt8(V+1)/128,F=e.getInt8(V+2)/128,Y=Math.sqrt(Math.max(0,1-E*E-T*T-F*F));let rt=Math.hypot(E,T,F,Y)||1;if(v[P+12]=E/rt,v[P+13]=T/rt,v[P+14]=F/rt,v[P+15]=Y/rt,h){let j=_+x*p*3,bt=x*d*3;for(let W=0;W<d*3;W++)B[bt+W]=Cs(e.getUint8(j+W))}(x+1)%ks===0&&await Ms()}return{data:v,count:r,shData:B,numSHBases:d}}function ct(a,t){let e=a.getUint8(t),s=a.getUint8(t+1),i=a.getInt8(t+2);return e|s<<8|i<<16}function gt(a){return a&512?a|4294966272:a}function Ot(a){return Math.exp((a-128)/16)}function Cs(a){return(a-128)/128}function qe(a){let t=a&3,e=Ve/512,s=gt(a>>2&1023)*e,i=gt(a>>12&1023)*e,r=gt(a>>22&1023)*e,n=Math.sqrt(Math.max(0,1-s*s-i*i-r*r)),o,l,c,u;switch(t){case 0:o=n,l=s,c=i,u=r;break;case 1:o=s,l=n,c=i,u=r;break;case 2:o=s,l=i,c=n,u=r;break;default:o=s,l=i,c=r,u=n}let p=Math.hypot(o,l,c,u)||1;return[o/p,l/p,c/p,u/p]}async function Ct(a){if(typeof DecompressionStream>"u")throw new Error("DecompressionStream is not available in this environment");let t=new DecompressionStream("gzip"),e=t.writable.getWriter();return e.write(a),e.close(),new Response(t.readable).arrayBuffer()}async function As(a,t,e={}){let s=je(a,t,e);return re(s)}function je(a,t,e={}){let{fractionalBits:s}=e;if(s==null){let _=0;for(let B=0;B<t;B++){let x=B*16;Math.abs(a[x])>_&&(_=Math.abs(a[x])),Math.abs(a[x+1])>_&&(_=Math.abs(a[x+1])),Math.abs(a[x+2])>_&&(_=Math.abs(a[x+2]))}let v=(1<<23)-1;s=_>0?Math.min(20,Math.max(0,Math.floor(Math.log2(v/_)))):12}let{shData:i,numSHBases:r=0}=e,n=r>=15?3:r>=8?2:r>=3?1:0,o=r*3,l=16,c=l+t*(20+o),u=new ArrayBuffer(c),p=new DataView(u),f=new Uint8Array(u);p.setUint32(0,1347635022,!0),p.setUint32(4,3,!0),p.setUint32(8,t,!0),p.setUint8(12,n),p.setUint8(13,s),p.setUint8(14,0),p.setUint8(15,0);let d=l,h=d+t*9,m=h+t*1,b=m+t*3,y=b+t*3,g=y+t*4,S=1<<s;for(let _=0;_<t;_++){let v=_*16;if(ht(p,d+_*9+0,a[v+0]*S),ht(p,d+_*9+3,a[v+1]*S),ht(p,d+_*9+6,a[v+2]*S),f[h+_]=K(a[v+7]*255),f[m+_*3+0]=K(a[v+4]*255),f[m+_*3+1]=K(a[v+5]*255),f[m+_*3+2]=K(a[v+6]*255),f[b+_*3+0]=ut(a[v+8]),f[b+_*3+1]=ut(a[v+9]),f[b+_*3+2]=ut(a[v+10]),p.setUint32(y+_*4,ne(a[v+12],a[v+13],a[v+14],a[v+15]),!0),i&&r>0){let B=_*o,x=_*r*3;for(let P=0;P<o;P++)f[g+B+P]=Fs(i[x+P])}}return f}function ht(a,t,e){let s=Math.max(-8388608,Math.min(8388607,Math.round(e)));a.setUint8(t,s&255),a.setUint8(t+1,s>>8&255),a.setUint8(t+2,s>>16&255)}function K(a){return Math.max(0,Math.min(255,Math.round(a)))}function ut(a){return K(Math.log(Math.max(1e-9,a))*16+128)}function Fs(a){return K(Math.round(a*128)+128)}function ne(a,t,e,s){let i=Math.hypot(a,t,e,s)||1,r=[a/i,t/i,e/i,s/i],n=0;for(let h=1;h<4;h++)Math.abs(r[h])>Math.abs(r[n])&&(n=h);let o=r[n]<0?-1:1,l=[0,1,2,3].filter(h=>h!==n),c=512*Math.SQRT2,u=h=>Math.max(-512,Math.min(511,Math.round(r[h]*o*c))),p=u(l[0]),f=u(l[1]),d=u(l[2]);return(n&3|(p&1023)<<2|(f&1023)<<12|(d&1023)<<22)>>>0}async function re(a){if(typeof CompressionStream>"u")throw new Error("CompressionStream API is not available in this environment");let t=new CompressionStream("gzip"),e=t.writable.getWriter();return e.write(a),e.close(),new Response(t.readable).arrayBuffer()}var Ht=1448104776,zs=1,ft=16;function $e(a,t,e={}){if(!a.length)throw new Error("HoloSplat: encodeSpzv requires at least one variant");if(a.length>255)throw new Error("HoloSplat: encodeSpzv supports at most 255 variants");let s=a[0].data,{fractionalBits:i}=e;if(i==null){let _=0;for(let B=0;B<t;B++){let x=B*16;_=Math.max(_,Math.abs(s[x]),Math.abs(s[x+1]),Math.abs(s[x+2]))}let v=(1<<23)-1;i=_>0?Math.min(20,Math.max(0,Math.floor(Math.log2(v/_)))):12}let r=a.length,n=16,o=r*ft,l=t*16,c=t*4,u=n+o+l+r*c,p=new ArrayBuffer(u),f=new DataView(p),d=new Uint8Array(p);f.setUint32(0,Ht,!0),f.setUint32(4,zs,!0),f.setUint32(8,t,!0),f.setUint8(12,r),f.setUint8(13,i),f.setUint8(14,0),f.setUint8(15,0);let h=n,m=new TextEncoder;for(let _ of a){let v=m.encode(_.name).slice(0,ft-1);d.set(v,h),h+=ft}let b=h,y=b+t*9,g=y+t*3;h=g+t*4;let S=1<<i;for(let _=0;_<t;_++){let v=_*16;ht(f,b+_*9+0,s[v+0]*S),ht(f,b+_*9+3,s[v+1]*S),ht(f,b+_*9+6,s[v+2]*S),d[y+_*3+0]=ut(s[v+8]),d[y+_*3+1]=ut(s[v+9]),d[y+_*3+2]=ut(s[v+10]),f.setUint32(g+_*4,ne(s[v+12],s[v+13],s[v+14],s[v+15]),!0)}for(let _ of a){let v=h,B=v+t;for(let x=0;x<t;x++){let P=x*16;d[v+x]=K(_.data[P+7]*255),d[B+x*3+0]=K(_.data[P+4]*255),d[B+x*3+1]=K(_.data[P+5]*255),d[B+x*3+2]=K(_.data[P+6]*255)}h=B+t*3}return d}async function Es(a,t,e={}){let s=$e(a,t,e);return re(s)}var Ts=5e4;function Ne(){return new Promise(a=>setTimeout(a,0))}async function Ye(a,t){let e=await at(a,t),s=await Ct(e);return oe(s)}async function Us(a){let t=await Ct(a);return oe(t)}async function oe(a){let t=new DataView(a),e=new Uint8Array(a),s=t.getUint32(0,!0),i=t.getUint32(4,!0),r=t.getUint32(8,!0),n=t.getUint8(12),o=t.getUint8(13);if(s!==Ht)throw new Error(`Invalid .spzv magic: 0x${s.toString(16).toUpperCase()} (expected 0x${Ht.toString(16).toUpperCase()})`);i!==1&&console.warn(`HoloSplat: .spzv version ${i} is untested; attempting load anyway`);let l=16,c=new TextDecoder,u=[];for(let y=0;y<n;y++){let g=e.subarray(l,l+ft),S=g.indexOf(0);u.push(c.decode(g.subarray(0,S<0?ft:S))),l+=ft}let p=l,f=p+r*9,d=f+r*3;l=d+r*4;let h=1<<o,m=new Float32Array(r*16);for(let y=0;y<r;y++){let g=y*16,S=p+y*9;m[g+0]=ct(t,S+0)/h,m[g+1]=ct(t,S+3)/h,m[g+2]=ct(t,S+6)/h;let _=f+y*3;m[g+8]=Ot(e[_+0]),m[g+9]=Ot(e[_+1]),m[g+10]=Ot(e[_+2]);let[v,B,x,P]=qe(t.getUint32(d+y*4,!0));m[g+12]=v,m[g+13]=B,m[g+14]=x,m[g+15]=P,(y+1)%Ts===0&&await Ne()}let b=[];for(let y=0;y<n;y++){let g=l,S=g+r,_=new Float32Array(r*4);for(let v=0;v<r;v++){let B=v*4;if(_[B+0]=e[S+v*3+0]/255,_[B+1]=e[S+v*3+1]/255,_[B+2]=e[S+v*3+2]/255,_[B+3]=e[g+v]/255,y===0){let x=v*16;m[x+4]=_[B+0],m[x+5]=_[B+1],m[x+6]=_[B+2],m[x+7]=_[B+3]}}l=S+r*3,b.push({name:u[y],palette:_}),(y+1)%1===0&&await Ne()}return{data:m,count:r,variants:b,shData:null,numSHBases:0}}function yt(a,t,e,s,i,r,n,o,l){let c=a*i+t*r+e*n+s*o;if(c<0&&(i=-i,r=-r,n=-n,o=-o,c=-c),c>.9995){let b=a+(i-a)*l,y=t+(r-t)*l,g=e+(n-e)*l,S=s+(o-s)*l,_=Math.sqrt(b*b+y*y+g*g+S*S);return[b/_,y/_,g/_,S/_]}let u=Math.acos(c),p=u*l,f=Math.sin(u),d=Math.sin(p),h=Math.cos(p)-c*d/f,m=d/f;return[h*a+m*i,h*t+m*r,h*e+m*n,h*s+m*o]}function le(a){let t=a.replace(/^hs-part\./,"").replace(/^ctrl\./,"");return t=t.replace(/(\.\d+)+$/,""),t}var It=class{constructor(t){if(!t.frames||t.frames.length===0)throw new Error("HoloSplat Animation: no frame data");this.fps=t.fps??24,this.frameCount=t.frameCount??Math.floor(t.frames.length/6),this.fov=t.fov??null,this.near=t.near??null,this.far=t.far??null,this.callouts=t.callouts??[],this.focalPoint=t.focalPoint??null,this._focalFrames=t.focalFrames?new Float32Array(t.focalFrames):null,this.loop=!0,Array.isArray(t.markers)?this.markers=Object.fromEntries(t.markers.map(e=>[e.name,e.frame])):this.markers=t.markers??{},this.stateCalls=t.stateCalls??[],this.properties=t.properties??{},this._frames=new Float32Array(t.frames),this._objects=(t.objects??[]).map(e=>({id:e.id,frames:new Float32Array(e.frames),variants:e.variants})),this._volumes=(t.volumes??[]).map(e=>({name:e.name,softEdge:e.softEdge??.05,matrices:new Float32Array(e.matrices)})),this._anchors=(t.anchors??[]).map(e=>({asset:e.asset,frames:new Float32Array(e.frames)})),this._time=0,this._playing=!0,this.direction=1,this.pingPong=!1}get duration(){return this.frameCount/this.fps}get time(){return this._time}get playing(){return this._playing}get objects(){return this._objects}get volumes(){return this._volumes}get anchors(){return this._anchors}play(){this._playing=!0}pause(){this._playing=!1}seek(t){this._time=Math.max(0,Math.min(this.duration,t))}seekFrame(t){this._time=Math.max(0,Math.min(this.frameCount-1,t))/this.fps}get frame(){return this._time*this.fps}tick(t){this._playing&&(this._time+=t*this.direction,this.direction>=0?this._time>=this.duration&&(this.pingPong?(this.direction=-1,this._time=2*this.duration-this._time):(this._time=this.loop?this._time%this.duration:this.duration,this.loop||(this._playing=!1))):this._time<=0&&(this.pingPong?(this.direction=1,this._time=-this._time):this.loop?this._time=this.duration+this._time:(this._time=0,this._playing=!1)))}getObjectFrames(t){let e=t!=null?Math.max(0,Math.min(t,this.frameCount-1)):Math.min(this._time*this.fps,this.frameCount-1),s=Math.min(Math.floor(e),this.frameCount-1),i=Math.min(s+1,this.frameCount-1),r=e-s;return this._objects.map(n=>{let o=n.frames,l=s*7,c=i*7;return{id:n.id,pos:[o[l]+(o[c]-o[l])*r,o[l+1]+(o[c+1]-o[l+1])*r,o[l+2]+(o[c+2]-o[l+2])*r],quat:yt(o[l+3],o[l+4],o[l+5],o[l+6],o[c+3],o[c+4],o[c+5],o[c+6],r)}})}getAnchorFrames(t){let e=t!=null?Math.max(0,Math.min(t,this.frameCount-1)):Math.min(this._time*this.fps,this.frameCount-1),s=Math.min(Math.floor(e),this.frameCount-1),i=Math.min(s+1,this.frameCount-1),r=e-s;return this._anchors.map(n=>{let o=n.frames,l=s*7,c=i*7;return{asset:n.asset,pos:[o[l]+(o[c]-o[l])*r,o[l+1]+(o[c+1]-o[l+1])*r,o[l+2]+(o[c+2]-o[l+2])*r],quat:yt(o[l+3],o[l+4],o[l+5],o[l+6],o[c+3],o[c+4],o[c+5],o[c+6],r)}})}getVolumeFrames(t){let e=t!=null?Math.max(0,Math.min(t,this.frameCount-1)):Math.min(this._time*this.fps,this.frameCount-1),s=Math.min(Math.floor(e),this.frameCount-1),i=Math.min(s+1,this.frameCount-1),r=e-s;return this._volumes.map(n=>{let o=n.matrices,l=s*16,c=i*16,u=new Float32Array(16);for(let p=0;p<16;p++)u[p]=o[l+p]+(o[c+p]-o[l+p])*r;return{name:n.name,softEdge:n.softEdge,matrix:u}})}getFocalPoint(){if(!this.focalPoint)return null;if(!this._focalFrames)return this.focalPoint;let t=Math.min(this._time*this.fps,this.frameCount-1),e=Math.min(Math.floor(t),this.frameCount-1),s=Math.min(e+1,this.frameCount-1),i=t-e,r=this._focalFrames,n=e*3,o=s*3;return[r[n]+(r[o]-r[n])*i,r[n+1]+(r[o+1]-r[n+1])*i,r[n+2]+(r[o+2]-r[n+2])*i]}getCameraFrame(t){let e=t!=null?Math.max(0,Math.min(t,this.frameCount-1)):Math.min(this._time*this.fps,this.frameCount-1),s=Math.min(Math.floor(e),this.frameCount-1),i=Math.min(s+1,this.frameCount-1),r=e-s,n=this._frames,o=s*6,l=i*6,c=n[o]+(n[l]-n[o])*r,u=n[o+1]+(n[l+1]-n[o+1])*r,p=n[o+2]+(n[l+2]-n[o+2])*r,f=n[o+3]+(n[l+3]-n[o+3])*r,d=n[o+4]+(n[l+4]-n[o+4])*r,h=n[o+5]+(n[l+5]-n[o+5])*r;return{eye:[c,u,p],target:[c+f,u+d,p+h]}}};function We(a,t){let e=Math.max(0,Math.min(Math.floor(t),a.frameCount-1)),s=Math.min(e+1,a.frameCount-1),i=Math.max(0,Math.min(1,t-e));return a.objects.map(r=>{let n=r.frames,o=e*7,l=s*7;return{id:r.id,pos:[n[o]+(n[l]-n[o])*i,n[o+1]+(n[l+1]-n[o+1])*i,n[o+2]+(n[l+2]-n[o+2])*i],quat:yt(n[o+3],n[o+4],n[o+5],n[o+6],n[l+3],n[l+4],n[l+5],n[l+6],i)}})}function Qe(a,t){let e=a.masks;if(!e||!e.length)return[];let s=Math.max(0,Math.min(Math.floor(t),a.frameCount-1)),i=Math.min(s+1,a.frameCount-1),r=Math.max(0,Math.min(1,t-s));return e.map(({name:n,softEdge:o,matrices:l})=>{let c=s*16,u=i*16,p=new Float32Array(16);for(let f=0;f<16;f++)p[f]=l[c+f]+(l[u+f]-l[c+f])*r;return{name:n,softEdge:o,matrix:p}})}function ce(a,t,e){let s=Math.max(0,Math.min(Math.floor(e),t-1)),i=Math.min(s+1,t-1),r=Math.max(0,Math.min(1,e-s)),n=s*7,o=i*7;return{pos:[a[n]+(a[o]-a[n])*r,a[n+1]+(a[o+1]-a[n+1])*r,a[n+2]+(a[o+2]-a[n+2])*r],quat:yt(a[n+3],a[n+4],a[n+5],a[n+6],a[o+3],a[o+4],a[o+5],a[o+6],r)}}function he(a,t,e){let s=Math.max(0,Math.min(Math.floor(e),t-1)),i=Math.min(s+1,t-1),r=Math.max(0,Math.min(1,e-s)),n=s*16,o=i*16,l=new Float32Array(16);for(let c=0;c<16;c++)l[c]=a[n+c]+(a[o+c]-a[n+c])*r;return l}async function ue(a){let t=await fetch(a);if(!t.ok)throw new Error(`HoloSplat: failed to load animation "${a}" (HTTP ${t.status})`);return new It(await t.json())}var Ke={low:{maxPixelRatio:1,shDegreeCap:1,lod:3,prefetchVariants:!1},medium:{maxPixelRatio:1.5,shDegreeCap:1,lod:2,prefetchVariants:!1},high:{maxPixelRatio:2,shDegreeCap:1/0,lod:0,prefetchVariants:!0}};function Vt(){if(typeof navigator>"u")return"high";if(navigator.connection?.saveData)return"low";let a=navigator.deviceMemory,t=navigator.hardwareConcurrency||4,e=/Android|iPhone|iPad|iPod/i.test(navigator.userAgent||"");return a!=null&&a<=2||t<=2||e?"low":a!=null&&a<=4||t<=4?"medium":"high"}function Lt(a){return Ke[a]||Ke.high}async function et(a,t){if(!t)return a;let e=a.match(/\.(spz|ply|splat)$/i);if(!e)return a;let s=a.slice(0,-e[0].length),i=s.lastIndexOf("/"),r=s.slice(0,i+1),n=s.slice(i+1),o=`${r}${n}.lods/${n}.lod${t}.spz`;try{if((await fetch(o,{method:"HEAD",cache:"no-store"})).ok)return o}catch{}return a}async function dt(a,t){if(!t)return a;let e={};for(let[s,i]of Object.entries(a))Array.isArray(i)?e[s]=await Promise.all(i.map(r=>et(r,t))):i&&typeof i=="object"?e[s]={...i,url:await et(i.url,t)}:e[s]=await et(i,t);return e}var xt=class{constructor(t={}){let{canvas:e,background:s="#000000",fov:i=60,near:r=.01,far:n=2e3,splatScale:o=1.08,autoRotate:l=!1,flipY:c=!1,shDegree:u=0,aaDilation:p=.3,maxPixelRatio:f=2,adaptiveQuality:d=!0,prefetchVariants:h=!0,gpuSort:m=!1,tier:b=null,onProgress:y,onError:g}=t;this._canvas=$s(e),this._onProgress=y,this._onError=g,this._autoRotate=l,this._splatScale=o,this._maxPixelRatio=f,this._flipY=c,this._shDegree=u,this._aaDilation=p,this._tier=b,this._adaptiveQuality=d,this._prefetchVariantsEnabled=h,this._gpuSort=m,this._effectivePixelRatio=f,this._minPixelRatio=.5,this._frameTimeEMA=null,this._lastQualityCheck=0,this._wasSceneReady=!1,this._qualityWarmupUntil=0,this._effectiveRadiusCap=1,this._minRadiusCap=.4,this._maxRadiusCap=1,this._renderer=new Rt(this._canvas,s),this._camera=new Gt({fov:i,near:r,far:n}),this._gaussians=null,this._numSplats=0,this._depths=null,this._sort=null,this._rafId=null,this._running=!1,this._resizeObs=null,this._resizeTimer=null,this._partIndex={},this._fileNames=[],this._partVariants={},this._partTransforms=[nt.slice()],this._partTransFlat=nt.slice(),this._partLocalPose={},this._fileRanges=null,this._fileAABB=null,this._fileMasks=null,this._activeIdx=null,this._lastActiveCount=0,this._currentSceneName=null,this._animation=null,this._animPaused=!1,this._cameraFree=!1,this._blendBack=null,this._sceneBlend=null,this._sceneReady=!1,this._lastUrl=null,this._lastParts=null,this._lastTick=null,this.onFrame=null,this._lastSortView=null,this._sortDirty=!0,this._lastRenderMs=0,this._panOffset=[0,0,0],this._panLimit=null,this._zoomFactor=1,this._zoomLimit=null,this._camMode=null,this._camModeType=null,this._maskFeather={},this._clips={},this._clipPlaybacks={},this._clipHeld={},this._clipMaskState={},this._clipMaskSoftEdge={},this._clipPartsRegistry={},this._clipPartsLoadQueue=Promise.resolve(),this._clipPartsFlush=null,this._urlCache=new Map,this._transitions={},this._transitionPlaybacks={},this._axisActive={},this._states={},this._stateFrame={},this._stateActive={},this._stateTarget={},this._statePlaybacks={},this._properties={}}setMaskFeather(t,e){this._maskFeather[t]=e}_mergeClipParts(t){return Object.assign(this._clipPartsRegistry,t),this._clipPartsFlush||(this._clipPartsFlush=new Promise((e,s)=>{setTimeout(()=>{this._clipPartsFlush=null;let i={...this._clipPartsRegistry},r=this._clipPartsLoadQueue.then(()=>this.loadParts(i),()=>this.loadParts(i));this._clipPartsLoadQueue=r.catch(()=>{}),r.then(e,s)},30)})),this._clipPartsFlush}async loadClips(t,e={}){let s=await fetch(t);if(!s.ok)throw new Error(`HoloSplat: failed to load clips "${t}" (HTTP ${s.status})`);let i=await s.json(),r=[];for(let n of i.clips??[]){let o=(n.masks??[]).map(l=>({name:l.name,softEdge:l.softEdge??.05,matrices:new Float32Array(l.matrices)}));this._clips[n.id]={id:n.id,fps:n.fps??i.fps??24,frameCount:n.frameCount,holdFrame:n.holdFrame,objects:(n.objects??[]).map(l=>({id:l.id,frames:new Float32Array(l.frames)})),masks:o},r.push(n.id);for(let l of o){if(this._clipMaskState[l.name])continue;let c=l.matrices.length/16-1;this._clipMaskState[l.name]=l.matrices.slice(c*16,c*16+16),this._clipMaskSoftEdge[l.name]=l.softEdge}}for(let[n,o]of Object.entries(i.transitions??{})){let l=(o.masks??[]).map(c=>({name:c.name,value:c.value,softEdge:c.softEdge??.05,matrices:new Float32Array(c.matrices)}));this._transitions[n]={fps:o.fps??i.fps??24,frameCount:o.frameCount,holdFrame:o.holdFrame,parts:(o.parts??[]).map(c=>({id:c.id,value:c.value,frames:new Float32Array(c.frames)})),masks:l};for(let c of l){if(this._clipMaskState[c.name])continue;let u=c.matrices.length/16-1;this._clipMaskState[c.name]=c.matrices.slice(u*16,u*16+16),this._clipMaskSoftEdge[c.name]=c.softEdge}}for(let[n,o]of Object.entries(i.states??{})){let l=(o.masks??[]).map(c=>({name:c.name,softEdge:c.softEdge??.05,matrices:new Float32Array(c.matrices)}));this._states[n]={fps:o.fps??i.fps??24,frameCount:o.frameCount,markers:o.markers??{},default:o.default,parts:(o.parts??[]).map(c=>({id:c.id,frames:new Float32Array(c.frames)})),masks:l}}if(Object.assign(this._properties,i.properties??{}),i.properties?.color&&this._applyColorProperty(i.properties.color),e.splatsDir&&i.parts&&Object.keys(i.parts).length){let n=e.splatsDir.replace(/\/?$/,"/"),o={};for(let[l,c]of Object.entries(i.parts))o[l]=`${n}${c.splatName}${Xe(c,e.defaults)}.spz`;e.lod&&(o=await dt(o,e.lod)),await this._mergeClipParts(o),Object.values(i.parts).some(l=>l.variants.length>1)&&setTimeout(()=>{(async()=>{let l={};for(let[c,u]of Object.entries(i.parts)){if(!u.variants.length){l[c]=`${n}${u.splatName}.spz`;continue}let p=Xe(u,e.defaults).slice(1);l[c]=await Promise.all(u.variants.map(f=>{let d=f===p?e.lod||0:Math.max(e.lod||0,Gs);return et(`${n}${u.splatName}.${f}.spz`,d)}))}try{await this._mergeClipParts(l)}catch(c){console.error(`[HoloSplat] background variant load failed for "${t}":`,c)}})()},Rs)}for(let[n,o]of Object.entries(e.defaults??{}))(this._transitions[n]||this._properties[`${n}=${o}`])&&this._setVariantInstant(n,o),this._states[n]&&this._setStateInstant(n,o);for(let[n,o]of Object.entries(this._states))this._stateActive[n]===void 0&&o.default!=null&&this._setStateInstant(n,o.default);return r}unloadClips(t){for(let e of t)delete this._clips[e]}playClip(t){let e=this._clips[t];if(!e){console.warn(`[HoloSplat] playClip: unknown clip "${t}"`);return}let s=t.lastIndexOf("-");if(s<=0){console.warn(`[HoloSplat] playClip: "${t}" doesn't match "productName-variant"`);return}let i=t.slice(0,s),r=this._clipHeld[i];if(r===t)return;let n=r?this._clips[r]:null;this._clipPlaybacks[i]=n?{clip:n,dir:"out",frame:n.holdFrame,nextClipId:t}:{clip:e,dir:"in",frame:0,nextClipId:null},this._clipHeld[i]=t}_tickClips(t){for(let e in this._clipPlaybacks){let s=this._clipPlaybacks[e],{clip:i,dir:r}=s,n=r==="in"?i.holdFrame:i.frameCount-1;s.frame+=t*i.fps;let o=s.frame>=n;o&&(s.frame=n),this._applyClipFrame(i,s.frame),o&&(r==="out"&&s.nextClipId?this._clipPlaybacks[e]={clip:this._clips[s.nextClipId],dir:"in",frame:0,nextClipId:null}:delete this._clipPlaybacks[e])}}_applyClipFrame(t,e){let s=!1;for(let{id:i,pos:r,quat:n}of We(t,e)){let o=this._partIndex[i];if(o&&o.length){let l=zt(r,n);this._partLocalPose[i]=l;for(let c of o)this._partTransFlat.set(l,c*16);s=!0}}for(let{name:i,softEdge:r,matrix:n}of Qe(t,e))this._clipMaskState[i]=n,this._clipMaskSoftEdge[i]=r,this._sortDirty=!0;s&&(this._renderer.updateTransforms(this._partTransFlat),this._sortDirty=!0)}playVariant(t,e){let s=this._transitions[t],i=this._properties[`${t}=${e}`];if(!s&&!i){console.warn(`[HoloSplat] playVariant: unknown axis "${t}"`);return}let r=this._axisActive[t];r!==e&&(this._axisActive[t]=e,s&&(this._transitionPlaybacks[t]={value:e,prevValue:r,elapsed:0}),i&&this._applyColorProperty(i))}_setVariantInstant(t,e){let s=this._transitions[t];this._axisActive[t]=e,s&&this._applyTransitionValue(s,e,s.holdFrame);let i=this._properties[`${t}=${e}`];i&&this._applyColorProperty(i)}_applyColorProperty({hue:t=0,sat:e=1,val:s=1}={}){this._renderer.setHueShift(t),this._renderer.setSaturation(e),this._renderer.setValue(s)}setColorProperty(t){this._applyColorProperty(t)}getProperties(){return{...this._properties}}_tickTransitions(t){for(let e in this._transitionPlaybacks){let s=this._transitionPlaybacks[e],i=this._transitions[e];s.elapsed+=t*i.fps;let r=Math.min(s.elapsed,i.holdFrame),n=Math.min(i.holdFrame+s.elapsed,i.frameCount-1);this._applyTransitionValue(i,s.value,r),s.prevValue&&this._applyTransitionValue(i,s.prevValue,n),r>=i.holdFrame&&n>=i.frameCount-1&&delete this._transitionPlaybacks[e]}}_applyTransitionValue(t,e,s){let i=!1;for(let r of t.parts){if(r.value!==e)continue;let n=this._partIndex[r.id];if(n&&n.length){let{pos:o,quat:l}=ce(r.frames,t.frameCount,s),c=zt(o,l);this._partLocalPose[r.id]=c;for(let u of n)this._partTransFlat.set(c,u*16);i=!0}}for(let r of t.masks)r.value===e&&(this._clipMaskState[r.name]=he(r.matrices,t.frameCount,s),this._clipMaskSoftEdge[r.name]=r.softEdge,this._sortDirty=!0);i&&(this._renderer.updateTransforms(this._partTransFlat),this._sortDirty=!0)}playState(t,e){let s=this._states[t];if(!s){console.warn(`[HoloSplat] playState: unknown state axis "${t}"`);return}if(!(e in s.markers)){console.warn(`[HoloSplat] playState: axis "${t}" has no value "${e}"`);return}if(this._stateTarget[t]===e)return;let i=s.markers[e],r=this._stateFrame[t]??i;if(this._stateTarget[t]=e,r===i){this._stateActive[t]=e,delete this._statePlaybacks[t];return}this._statePlaybacks[t]={dir:i>r?1:-1,frame:r,toFrame:i,value:e}}_setStateInstant(t,e){let s=this._states[t];if(!s||!(e in s.markers))return;let i=s.markers[e];this._stateFrame[t]=i,this._stateActive[t]=e,this._stateTarget[t]=e,delete this._statePlaybacks[t],this._applyStateFrame(s,i)}_tickStates(t){for(let e in this._statePlaybacks){let s=this._statePlaybacks[e],i=this._states[e];s.frame+=t*i.fps*s.dir;let r=s.dir>0?s.frame>=s.toFrame:s.frame<=s.toFrame;r&&(s.frame=s.toFrame),this._applyStateFrame(i,s.frame),this._stateFrame[e]=s.frame,r&&(this._stateActive[e]=s.value,delete this._statePlaybacks[e])}}_applyStateFrame(t,e){let s=!1;for(let i of t.parts){let r=this._partIndex[i.id];if(r&&r.length){let{pos:n,quat:o}=ce(i.frames,t.frameCount,e),l=zt(n,o);this._partLocalPose[i.id]=l;for(let c of r)this._partTransFlat.set(l,c*16);s=!0}}for(let i of t.masks)this._clipMaskState[i.name]=he(i.matrices,t.frameCount,e),this._clipMaskSoftEdge[i.name]=i.softEdge,this._sortDirty=!0;s&&(this._renderer.updateTransforms(this._partTransFlat),this._sortDirty=!0)}_syncAssetStates(){let t=this._animation.stateCalls;if(!t||!t.length)return;let e=this._animation.frame,s={};for(let i of t)this._states[i.axis]&&i.frame<=e&&(!s[i.axis]||i.frame>s[i.axis].frame)&&(s[i.axis]=i);for(let i in this._states){let r=s[i],n=r?r.value:this._states[i].default;n!=null&&this.playState(i,n)}}async init(){await this._renderer.init(),this._renderer.setSplatScale(this._splatScale),this._renderer.setAaDilation(this._aaDilation),this._camera.attach(this._canvas),this._observeResize(),this._updateSize()}async load(t){if(this._lastUrl=t,this._lastParts=null,this._sceneReady=!1,t.split("?")[0].split(".").pop().toLowerCase()==="ply")try{await this._loadPlyStreamSingle(t);return}catch(l){if(!/^HTTP 4/.test(l.message))throw l}let{data:s,count:i,variants:r,shData:n,numSHBases:o}=await jt(t,l=>this._onProgress?.(l),this._shDegree);this._flipY&&(vt(s,i),At(n,o,i));for(let l=0;l<i;l++)s[l*16+3]=0;if(this._lastShData=n,this._lastNumSHBases=o,this._gaussians=s,this._numSplats=i,this._depths=new Float32Array(i),this._sort=_t(i),this._partIndex={},this._partLocalPose={},this._partVariants=r?{0:{kind:"spzv",variants:r,active:r[0]?.name}}:{},this._fileNames=[qt(t)],this._partTransforms=[nt.slice()],this._partTransFlat=nt.slice(),this._fileRanges=[[0,i]],this._fileAABB=[null],this._activeIdx=null,this._renderer.uploadGaussians(s,i),this._renderer.uploadTransforms(this._partTransforms),this._renderer.uploadSH(n||null,o||0),this._renderer.setShDegree(o>0?this._shDegree:0),this._sortDirty=!0,this._camera.fitScene(s,i),this._animation){let{eye:l,target:c}=this._animation.getCameraFrame();this._camera.setFromLookAt(l,c)}this._buildPartVolumeMask(),this._sceneReady=!0}async _loadPlyStreamSingle(t){let{numVertices:e,numSHBases:s,consume:i}=await ie(t,o=>this._onProgress?.(o*.02),this._shDegree),r=new Float32Array(e*16);if(this._gaussians=r,this._numSplats=e,this._depths=new Float32Array(e),this._sort=_t(e),this._partIndex={},this._partLocalPose={},this._partVariants={},this._fileNames=[qt(t)],this._partTransforms=[nt.slice()],this._partTransFlat=nt.slice(),this._fileRanges=[[0,e]],this._fileAABB=[null],this._activeIdx=null,this._renderer.uploadGaussians(r,e),this._renderer.uploadTransforms(this._partTransforms),this._renderer.allocateSH(e,s),this._renderer.setShDegree(s>0?this._shDegree:0),this._buildPartVolumeMask(),this._animation){let{eye:o,target:l}=this._animation.getCameraFrame();this._camera.setFromLookAt(o,l)}this._lastShData=s>0?new Float32Array(e*s*3):null,this._lastNumSHBases=s;let n=0;await i((o,l,c)=>{for(let u=0;u<l;u++)o[u*16+3]=0;this._flipY&&(vt(o,l),At(c,s,l)),this._renderer.patchGaussians(o,n),this._renderer.patchSH(c,n),this._gaussians.set(o,n*16),this._lastShData&&c&&this._lastShData.set(c,n*s*3),n+=l},o=>this._onProgress?.(o)),this._animation||this._camera.fitScene(this._gaussians,this._numSplats),this._sceneReady=!0}async _loadUrlCached(t,e,s){let i=`${t}#${s}`,r=this._urlCache.get(i);r?e?.(1):(r=jt(t,e,s),this._urlCache.set(i,r),r.catch(()=>this._urlCache.delete(i)));let n=await r;return{...n,data:n.data.slice(),shData:n.shData?n.shData.slice():n.shData}}async loadParts(t){this._lastParts=t,this._lastUrl=null;let e=Object.keys(t);if(e.length===0)throw new Error("HoloSplat: loadParts called with empty map");this._sceneReady=!1;let s=[];e.forEach(h=>{let m=t[h];if(m&&typeof m=="object"&&!Array.isArray(m)){s.push({id:h,slot:s.length,url:m.url,variantNames:m.variants});return}let b=Array.isArray(m)?m:[m];for(let y of b)s.push({id:h,slot:s.length,url:y})});let i=new Array(s.length).fill(0),r=()=>{this._onProgress&&this._onProgress(i.reduce((h,m)=>h+m,0)/s.length)},n=await Promise.all(s.map(async({id:h,slot:m,url:b},y)=>{if(b.split("?")[0].split(".").pop().toLowerCase()==="ply")try{let{numVertices:P,numSHBases:k,consume:U}=await ie(b,I=>{i[y]=I*.03,r()},this._shDegree);return{id:h,slot:m,kind:"stream",numVertices:P,numSHBases:k,consume:U}}catch(P){if(!/^HTTP 4/.test(P.message))throw P}let{data:S,count:_,variants:v,shData:B,numSHBases:x}=await this._loadUrlCached(b,P=>{i[y]=P*.9,r()},this._shDegree);return i[y]=.9,{id:h,slot:m,kind:"loaded",data:S,count:_,variants:v,shData:B,numSHBases:x}})),o=n.map(h=>h.kind==="stream"?h.numVertices:h.count),l=o.reduce((h,m)=>h+m,0),c=[];{let h=0;for(let m of o)c.push(h),h+=m}let u=new Float32Array(l*16),p=new Array(s.length).fill(null);n.forEach((h,m)=>{if(h.kind!=="loaded")return;let{data:b,count:y,slot:g,shData:S,numSHBases:_}=h;for(let v=0;v<y;v++)b[v*16+3]=g;this._flipY&&(vt(b,y),At(S,_,y)),u.set(b,c[m]*16),p[g]=de(b,y),i[m]=1}),this._partIndex={},e.forEach(h=>{this._partIndex[h]=[]}),s.forEach(({id:h,slot:m})=>{this._partIndex[h].push(m)}),this._fileNames=s.map(h=>qt(h.url)),this._partVariants={},n.forEach(h=>{h.variants&&(this._partVariants[h.slot]={kind:"spzv",variants:h.variants,active:h.variants[0]?.name})}),s.forEach(({slot:h,url:m,variantNames:b})=>{if(!b?.length||this._partVariants[h])return;let g=m.split("?")[0].match(/^(.*)\.([^./]+)\.(spz|ply|splat)$/i);!g||!b.includes(g[2])||(this._partVariants[h]={kind:"file",names:b,active:g[2],baseUrl:g[1],ext:g[3],cache:{}})}),this._partTransforms=s.map(()=>nt.slice()),this._partTransFlat=new Float32Array(s.length*16);for(let h=0;h<s.length;h++)this._partTransFlat.set(nt,h*16);this._fileRanges=c.map((h,m)=>[h,h+o[m]]),this._fileAABB=p,this._activeIdx=new Uint32Array(l),this._gaussians=u,this._numSplats=l,this._depths=new Float32Array(l),this._sort=_t(l),this._renderer.uploadGaussians(u,l),this._renderer.uploadTransforms(this._partTransforms);let f=n.reduce((h,m)=>Math.max(h,m.numSHBases||0),0);if(this._renderer.allocateSH(l,f),this._renderer.setShDegree(f>0?this._shDegree:0),n.forEach((h,m)=>{h.kind==="loaded"&&h.shData&&h.numSHBases===f&&this._renderer.patchSH(h.shData,c[m])}),this._buildPartVolumeMask(),this._camera.fitScene(u,l),this._animation){let{eye:h,target:m}=this._animation.getCameraFrame();this._camera.setFromLookAt(h,m)}r();let d=n.map((h,m)=>({part:h,i:m})).filter(({part:h})=>h.kind==="stream");if(d.length===0){this._sceneReady=!0,this._prefetchVariants();return}await Promise.all(d.map(async({part:h,i:m})=>{let{slot:b,consume:y}=h,g=c[m];await y((S,_,v)=>{for(let B=0;B<_;B++)S[B*16+3]=b;this._flipY&&(vt(S,_),At(v,h.numSHBases,_)),this._renderer.patchGaussians(S,g),v&&h.numSHBases===f&&this._renderer.patchSH(v,g),this._gaussians.set(S,g*16),this._fileAABB[b]=Hs(this._fileAABB[b],S,_),g+=_},S=>{i[m]=.03+.97*S,r()}),i[m]=1,r()})),this._animation||this._camera.fitScene(this._gaussians,this._numSplats),this._sceneReady=!0,this._prefetchVariants()}start(){this._running||(this._running=!0,this._tick())}stop(){this._running=!1,this._rafId&&cancelAnimationFrame(this._rafId),this._rafId=null}destroy(){this.stop(),this._camera.detach(),this._renderer.destroy(),this._resizeObs?.disconnect(),clearTimeout(this._resizeTimer)}setBackground(t){this._renderer.setBackground(t)}setSplatScale(t){this._splatScale=t,this._renderer.setSplatScale(t)}setGamma(t){this._renderer.setGamma(t)}async setShDegree(t){t!==this._shDegree&&(this._shDegree=t,this._lastParts?await this.loadParts(this._lastParts):this._lastUrl&&await this.load(this._lastUrl))}setAaDilation(t){this._aaDilation=t,this._renderer.setAaDilation(t),this._sortDirty=!0}setDebugIndex(t){this._renderer.setDebugIndex(t)}async readDebug(){return this._renderer.readDebug()}setAutoRotate(t){this._autoRotate=t}setFlipY(t){!!t!==this._flipY&&(this._flipY=!!t,this._gaussians&&(vt(this._gaussians,this._numSplats),this._renderer.uploadGaussians(this._gaussians,this._numSplats),this._camera.fitScene(this._gaussians,this._numSplats),this._sortDirty=!0,this._lastShData&&(At(this._lastShData,this._lastNumSHBases,this._numSplats),this._renderer.uploadSH(this._lastShData,this._lastNumSHBases))))}getStats(){return{fps:this._frameTimeEMA?1/this._frameTimeEMA:0,numSplats:this._numSplats,activeSplats:this._lastActiveCount||this._numSplats,shDegree:this._shDegree,pixelRatio:this._effectivePixelRatio,sceneName:this._currentSceneName,gpuSort:this._gpuSort&&!this._renderer._gpuSortFailed,gpuSortFailed:this._gpuSort&&this._renderer._gpuSortFailed,tier:this._tier}}getSplatDebug(t){if(!this._gaussians||t<0||t>=this._numSplats)return null;let e=t*16,s=this._gaussians;return{pos:[s[e],s[e+1],s[e+2]],part:s[e+3],color:[s[e+4],s[e+5],s[e+6],s[e+7]],scale:[s[e+8],s[e+9],s[e+10]],quat:[s[e+12],s[e+13],s[e+14],s[e+15]]}}getVariants(t){let e=t==null?[0]:this._partIndex[t];if(!e)return[];let s=this._partVariants[e[0]];return s?s.kind==="spzv"?s.variants.map(i=>i.name):s.names:[]}async setVariant(t,e){let s=t==null?[0]:this._partIndex[t];if(!s)return!1;let i=!1;for(let r of s){let n=this._partVariants[r];if(n){if(n.kind==="spzv"){let o=n.variants.find(u=>u.name===e);if(!o)continue;let[l,c]=this._fileRanges[r];for(let u=l;u<c;u++){let p=u*16,f=(u-l)*4;this._gaussians[p+4]=o.palette[f+0],this._gaussians[p+5]=o.palette[f+1],this._gaussians[p+6]=o.palette[f+2],this._gaussians[p+7]=o.palette[f+3]}this._renderer.patchGaussians(this._gaussians.subarray(l*16,c*16),l),n.active=e,i=!0;continue}if(n.active===e){i=!0;continue}n.names.includes(e)&&(await this._swapPartVariant(r,n,e),i=!0)}}return i}async _swapPartVariant(t,e,s){let i=e.cache[s];i||(i=await jt(`${e.baseUrl}.${s}.${e.ext}`),e.cache[s]=i);let[r,n]=this._fileRanges[t],o=n-r,l=i.count,c=l-o,u=i.data.slice();for(let p=0;p<l;p++)u[p*16+3]=t;if(this._flipY&&vt(u,l),c===0)this._gaussians.set(u,r*16),this._renderer.patchGaussians(u,r);else{let p=this._numSplats+c,f=new Float32Array(p*16);f.set(this._gaussians.subarray(0,r*16),0),f.set(u,r*16),f.set(this._gaussians.subarray(n*16),(r+l)*16);for(let d=0;d<this._fileRanges.length;d++){let[h,m]=this._fileRanges[d];d===t?this._fileRanges[d]=[r,r+l]:h>=n&&(this._fileRanges[d]=[h+c,m+c])}this._gaussians=f,this._numSplats=p,this._depths=new Float32Array(p),this._sort=_t(p),this._activeIdx=new Uint32Array(p),this._renderer.uploadGaussians(this._gaussians,this._numSplats)}this._fileAABB[t]=de(u,l),this._fileNames[t]=qt(`${e.baseUrl}.${s}.${e.ext}`),e.active=s,this._buildPartVolumeMask()}async _prefetchVariants(){if(this._prefetchVariantsEnabled){for(let t of Object.values(this._partVariants))if(t.kind==="file"){for(let e of t.names)if(!(e===t.active||t.cache[e]))try{t.cache[e]=await jt(`${t.baseUrl}.${e}.${t.ext}`)}catch(s){console.warn(`[HoloSplat] variant prefetch failed for "${t.baseUrl}.${e}.${t.ext}": ${s.message}`)}}}}setAnimationPaused(t){this._animPaused=t}setCameraFree(t){let e=this._cameraFree;this._cameraFree=!!t,t?(this._blendBack=null,this._camera.disableZoom(),this.animTickOverride&&(this._camera.allowTouchScroll=!1)):(this._camera.enableZoom(),this.animTickOverride&&(this._camera.allowTouchScroll=!0),e&&this._animation&&(this._blendBack={fromEye:this._camera.eye.slice(),fromTarget:this._camera.target.slice(),t:0,duration:.5}))}_syncCameraMode(){let t=this._animation.markers,e=this._animation.frame,s=null,i=-1;for(let[c,u]of Object.entries(t))u<=e&&u>i&&(i=u,s=c);let r=null;if(s&&(r=window.__hsSceneConfigs?.[s]??null,!r)){let c=Vs(s);if(c&&c!=="hs-locked"){let u=c.match(/zoom-(\d+)/);r={zoom:{enabled:!!u,mode:"limited",range:u?+u[1]:25}}}}let n=r?JSON.stringify(r):null;if(n===this._camMode)return;this._camMode=n;let o=r?`${+!!r.pan?.enabled}:${+!!r.zoom?.enabled}`:null,l=o!==this._camModeType;if(this._camModeType=o,r){let c=r.pan||{},u=r.zoom||{};if(l&&(this._panOffset=[0,0,0],this._zoomFactor=1),u.enabled?(this._camera.zoomEnabled=!0,this._zoomLimit=u.limited?Math.max(0,(u.range??500)/100):null,this._zoomLimit!==null&&(this._zoomFactor=Math.max(1-this._zoomLimit,Math.min(1+this._zoomLimit,this._zoomFactor))),this._camera.zoomDeltaCallback=p=>{let f=this._zoomFactor*p;this._zoomLimit!==null&&(f=Math.max(1-this._zoomLimit,Math.min(1+this._zoomLimit,f))),this._zoomFactor=Math.max(.01,f)}):(this._camera.zoomEnabled=!1,this._camera.zoomDeltaCallback=null,this._zoomFactor=1),c.enabled){if(this._camera.panEnabled=!0,this._camera.panSpeed=1-Math.min(100,Math.max(0,c.damping??0))/100,this._camera.panButton=c.button==="left"?0:2,this._panLimit=c.limited?Math.max(0,(c.radius??500)/100)*this._camera.radius:null,this._panLimit!==null){let p=this._panOffset,f=Math.hypot(p[0],p[1],p[2]);if(f>this._panLimit){let d=this._panLimit/f;p[0]*=d,p[1]*=d,p[2]*=d}}this._camera.panDeltaCallback=(p,f,d)=>{let h=this._panOffset;if(h[0]+=p,h[1]+=f,h[2]+=d,this._panLimit!==null){let m=Math.hypot(h[0],h[1],h[2]);if(m>this._panLimit){let b=this._panLimit/m;h[0]*=b,h[1]*=b,h[2]*=b}}}}else this._camera.panEnabled=!1,this._camera.panDeltaCallback=null,this._panOffset=[0,0,0]}else this._panOffset=[0,0,0],this._zoomFactor=1,this._camera.panDeltaCallback=null,this._camera.zoomDeltaCallback=null,this._camera.panEnabled=!this._animation.focalPoint,this._camera.panSpeed=1,this._camera.panButton=2,this._camera.panRadius=null,this._camera.panOrigin=null,this._camera.enableZoom()}resetCamera(){this._camera.fitScene(this._gaussians,this._numSplats)}focusCamera(){this._camera.focusScene(this._gaussians,this._numSplats)}setGaussians(t,e,s=!1){this._gaussians=t,this._numSplats=e,this._depths=new Float32Array(e),this._sort=_t(e),this._sceneReady=!0,this._renderer.uploadGaussians(t,e),s&&this._camera.fitScene(t,e)}uploadDisplay(t){this._numSplats&&this._renderer.uploadGaussians(t,this._numSplats)}_buildPartVolumeMask(){let t=new Set;for(let n of Object.values(this._clips))for(let o of n.masks??[])t.add(o.name);for(let n of Object.values(this._transitions))for(let o of n.masks??[])t.add(o.name);for(let n of Object.values(this._states))for(let o of n.masks??[])t.add(o.name);let e=[...this._animation?.volumes??[],...[...t].map(n=>({name:n}))],s=this._fileNames??[],i=Math.max(s.length,1),r=new Uint32Array(i);e.forEach((n,o)=>{let l=n.name,c=0;for(let u=0;u<s.length;u++)Is(s[u],l)&&(r[u]|=1<<o,c++);c===0&&s.length>0&&console.warn(`[HoloSplat] mask volume "${l}" matched 0 of ${s.length} file(s) \u2014 check naming convention`)}),this._fileMasks=r,this._renderer.uploadPartVolumeMask(r)}setAnimation(t){if(this._animation=t,!t)return;t.fov!=null&&(this._camera.fov=t.fov*Math.PI/180),t.near!=null&&(this._camera.near=t.near),t.far!=null&&(this._camera.far=t.far),Object.assign(this._properties,t.properties??{}),this._properties.color&&this._applyColorProperty(this._properties.color);let{eye:e,target:s}=t.getCameraFrame();this._camera.setFromLookAt(e,s),this._buildPartVolumeMask()}async loadAnimationUrl(t){let e=await ue(t);return this.setAnimation(e),e}projectCallouts(t){let e=this._camera.viewMatrix,s=this._camera.projMatrix,i=this._canvas.clientWidth,r=this._canvas.clientHeight,n=[];for(let o of t){let[l,c,u]=o.pos,p=e[0]*l+e[4]*c+e[8]*u+e[12],f=e[1]*l+e[5]*c+e[9]*u+e[13],d=e[2]*l+e[6]*c+e[10]*u+e[14];if(d>=0){n.push({id:o.id,visible:!1,x:0,y:0});continue}let h=-d,m=(s[0]*p/h*.5+.5)*i,b=(1-(s[5]*f/h*.5+.5))*r;n.push({id:o.id,visible:!0,x:m,y:b})}return n}get camera(){return this._camera}_tick(){if(!this._running)return;this._rafId=requestAnimationFrame(()=>this._tick());let t=performance.now(),e=this._lastTick?Math.min((t-this._lastTick)/1e3,.1):0;this._lastTick=t,this._sceneReady&&!this._wasSceneReady&&(this._frameTimeEMA=null,this._qualityWarmupUntil=t+Ds),this._wasSceneReady=this._sceneReady;let s=!this._sceneReady||t<this._qualityWarmupUntil;this._updateAdaptiveQuality(e,s);let i=this._canvas.width,r=this._canvas.height,n=[],o=!1;if(Object.keys(this._clipPlaybacks).length>0&&this._tickClips(e),Object.keys(this._transitionPlaybacks).length>0&&this._tickTransitions(e),Object.keys(this._statePlaybacks).length>0&&this._tickStates(e),this._animation){if(!this._animPaused&&this._sceneReady&&(this.animTickOverride?this.animTickOverride(e):this._animation.tick(e)),!this._cameraFree){let{eye:y,target:g}=this._animation.getCameraFrame();if(this._blendBack){this._blendBack.t+=e;let S=qs(Math.min(this._blendBack.t/this._blendBack.duration,1));this._camera.setFromLookAt(Ft(this._blendBack.fromEye,y,S),Ft(this._blendBack.fromTarget,g,S)),this._blendBack.t>=this._blendBack.duration&&(this._blendBack=null)}else{let S=y,_=g;if(this._sceneBlend){let{otherEye:v,otherTarget:B,bf:x}=this._sceneBlend;S=Ft(v,y,x),_=Ft(B,g,x)}this._camera.setFromLookAt(S,_),(this._panOffset[0]!==0||this._panOffset[1]!==0||this._panOffset[2]!==0)&&(this._camera.target[0]+=this._panOffset[0],this._camera.target[1]+=this._panOffset[1],this._camera.target[2]+=this._panOffset[2]),this._zoomFactor!==1&&(this._camera.radius*=this._zoomFactor)}}this._syncCameraMode(),this._syncAssetStates();let h=this._animation.getObjectFrames();if(h.length>0){let y=!1;for(let{id:g,pos:S,quat:_}of h){let v=this._partIndex[g];if(v&&v.length){let B=S,x=_,P=this._sceneBlend?.otherObjects?.[g];if(P){let U=this._sceneBlend.bf;B=Ft(P.pos,S,U),x=yt(P.quat[0],P.quat[1],P.quat[2],P.quat[3],_[0],_[1],_[2],_[3],U)}let k=zt(B,x);this._partLocalPose[g]=k;for(let U of v)this._partTransFlat.set(k,U*16);y=!0}}y&&(this._renderer.updateTransforms(this._partTransFlat),o=!0)}let m=this._animation.getAnchorFrames();if(m.length){let y=!1;for(let{asset:g,pos:S,quat:_}of m){let v=zt(S,_),B=`ctrl.${g}`;for(let x in this._partIndex){if(!x.startsWith(B))continue;let P=this._partLocalPose[x],k=P?Ze(v,P):v;for(let U of this._partIndex[x])this._partTransFlat.set(k,U*16);y=!0}}y&&(this._renderer.updateTransforms(this._partTransFlat),o=!0)}if(n=this._animation.getVolumeFrames(),this._sceneBlend?.otherVolumes){let{otherVolumes:y,bf:g}=this._sceneBlend;for(let S of n){let _=y[S.name];_&&(S.matrix=js(_,S.matrix,g))}}let b=this._animation.frame;n.length>0&&b!==this._lastVolAnimFrame&&(this._sortDirty=!0),this._lastVolAnimFrame=b}else this._autoRotate&&(this._camera.theta+=.005);for(let h in this._clipMaskState)n.push({name:h,matrix:this._clipMaskState[h],softEdge:this._clipMaskSoftEdge[h]});if(n.length>0){for(let h of n){let m=this._maskFeather[h.name];m!=null&&(h.softEdge=m)}this._renderer.updateMaskVolumes(n)}if(!this._numSplats)return;this._camera.update(i,r);let l=this._camera.viewMatrix,c=this._camera.projMatrix,u=!this._lastSortView;if(!u){for(let h=0;h<16;h++)if(l[h]!==this._lastSortView[h]){u=!0;break}}if(!u&&!o&&!this._sortDirty||t-this._lastRenderMs<15)return;this._lastRenderMs=t,this._lastSortView||(this._lastSortView=new Float32Array(16)),this._lastSortView.set(l),this._sortDirty=!1,this.onFrame&&this.onFrame(l,c,i,r);let p=this._camera.focalLength(r);this._renderer.updateUniforms({view:l,proj:c,width:i,height:r,focal:p,near:this._camera.near,radiusCap:this._effectiveRadiusCap});let f=this._computeActiveRanges(n,l,c),d=f===null?this._numSplats:f;if(this._lastActiveCount=d,this._renderer.preprocess(d),this._gpuSort&&f===null&&!this._renderer._gpuSortFailed)this._renderer.runGpuSort(d);else{this._computeDepths(l);let h=this._sort(this._depths,d,f===null?null:this._activeIdx);this._renderer.updateOrder(h,d)}this._renderer.draw(d)}_computeDepths(t){let e=t[2],s=t[6],i=t[10],r=t[14],n=[],o=this._partTransFlat,l=this._partTransforms.length;for(let f=0;f<l;f++){let d=f*16;n.push([e*o[d]+s*o[d+1]+i*o[d+2],e*o[d+4]+s*o[d+5]+i*o[d+6],e*o[d+8]+s*o[d+9]+i*o[d+10],e*o[d+12]+s*o[d+13]+i*o[d+14]+r])}let c=this._gaussians,u=this._depths,p=this._numSplats;for(let f=0;f<p;f++){let d=f*16,h=n[c[d+3]];u[f]=h[0]*c[d]+h[1]*c[d+1]+h[2]*c[d+2]+h[3]}}_computeActiveRanges(t,e,s){return null}_isFileOutsideView(t,e,s,i,r){let n=this._fileAABB[t];if(!n)return!1;let{min:o,max:l}=n,c=this._partTransFlat.subarray(t*16,t*16+16),u=this._camera.near,p=!0,f=!0,d=!0,h=!0,m=!0;for(let b=0;b<8;b++){let y=b&1?l[0]:o[0],g=b&2?l[1]:o[1],S=b&4?l[2]:o[2],_=c[0]*y+c[4]*g+c[8]*S+c[12],v=c[1]*y+c[5]*g+c[9]*S+c[13],B=c[2]*y+c[6]*g+c[10]*S+c[14],x=e[0]*_+e[4]*v+e[8]*B+e[12],P=e[1]*_+e[5]*v+e[9]*B+e[13],k=e[2]*_+e[6]*v+e[10]*B+e[14];if(k<-u){p=!1;let U=-k*i,I=-k*r;x>-U&&(f=!1),x<U&&(d=!1),P>-I&&(h=!1),P<I&&(m=!1)}else f=d=h=m=!1}return p||f||d||h||m}_isFileHidden(t,e,s){let i=this._fileMasks[t];if(!i)return!1;let r=this._fileAABB[t];if(!r)return!1;let n=this._partTransFlat.subarray(t*16,t*16+16);for(let o=0;o<e.length;o++){if(!(i>>o&1))continue;let l=s[o];if(!l)continue;let c=Ze(l,n),u=1/0,p=1/0,f=1/0,d=-1/0,h=-1/0,m=-1/0;for(let y=0;y<8;y++){let g=y&1?r.max[0]:r.min[0],S=y&2?r.max[1]:r.min[1],_=y&4?r.max[2]:r.min[2],v=c[0]*g+c[4]*S+c[8]*_+c[12],B=c[1]*g+c[5]*S+c[9]*_+c[13],x=c[2]*g+c[6]*S+c[10]*_+c[14];v<u&&(u=v),v>d&&(d=v),B<p&&(p=B),B>h&&(h=B),x<f&&(f=x),x>m&&(m=x)}let b=.5+(e[o].softEdge??.05);if(d<-b||u>b||h<-b||p>b||m<-b||f>b)return!0}return!1}_observeResize(){typeof ResizeObserver>"u"||(this._resizeObs=new ResizeObserver(()=>{clearTimeout(this._resizeTimer),this._resizeTimer=setTimeout(()=>this._updateSize(),150)}),this._resizeObs.observe(this._canvas))}_updateSize(){let t=Math.min(window.devicePixelRatio||1,this._effectivePixelRatio),e=Math.round(this._canvas.clientWidth*t),s=Math.round(this._canvas.clientHeight*t);e&&s&&(this._canvas.width!==e||this._canvas.height!==s)&&(this._canvas.width=e,this._canvas.height=s)}_updateAdaptiveQuality(t,e){if(!this._adaptiveQuality||!t)return;let s=1-Math.exp(-t/.5);this._frameTimeEMA=this._frameTimeEMA==null?t:this._frameTimeEMA+(t-this._frameTimeEMA)*s;let i=performance.now();if(i-this._lastQualityCheck<1e3)return;this._lastQualityCheck=i;let r=1/27,n=1/50;!e&&this._frameTimeEMA>r&&this._effectivePixelRatio>this._minPixelRatio?(this._effectivePixelRatio=Math.max(this._minPixelRatio,this._effectivePixelRatio*.85),this._updateSize()):this._frameTimeEMA<n&&this._effectivePixelRatio<this._maxPixelRatio&&(this._effectivePixelRatio=Math.min(this._maxPixelRatio,this._effectivePixelRatio*1.05),this._updateSize()),!e&&this._frameTimeEMA>r&&this._effectiveRadiusCap>this._minRadiusCap?this._effectiveRadiusCap=Math.max(this._minRadiusCap,this._effectiveRadiusCap*.85):this._frameTimeEMA<n&&this._effectiveRadiusCap<this._maxRadiusCap&&(this._effectiveRadiusCap=Math.min(this._maxRadiusCap,this._effectiveRadiusCap*1.05))}},nt=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),Ds=8e3,Rs=4e3,Gs=2;function qt(a){return a.split("?")[0].split("/").pop().replace(/\.(spz|ply|splat)$/i,"").replace(/\.lod\d+$/i,"")}function Xe(a,t){return a.variants.length?"."+(a.variants.find(s=>{let i=s.indexOf("=");return i<0?!1:t?.[s.slice(0,i)]===s.slice(i+1)})??a.variants[0]):""}function Ze(a,t){let e=new Float32Array(16);for(let s=0;s<4;s++)for(let i=0;i<4;i++)e[s*4+i]=a[i]*t[s*4]+a[i+4]*t[s*4+1]+a[i+8]*t[s*4+2]+a[i+12]*t[s*4+3];return e}var Os=3;function de(a,t){if(t===0)return null;let e=1/0,s=1/0,i=1/0,r=-1/0,n=-1/0,o=-1/0;for(let l=0;l<t;l++){let c=l*16,u=a[c],p=a[c+1],f=a[c+2],d=Os*Math.max(a[c+8],a[c+9],a[c+10]);u-d<e&&(e=u-d),u+d>r&&(r=u+d),p-d<s&&(s=p-d),p+d>n&&(n=p+d),f-d<i&&(i=f-d),f+d>o&&(o=f+d)}return{min:[e,s,i],max:[r,n,o]}}function Hs(a,t,e){let s=de(t,e);return s?a?{min:[Math.min(a.min[0],s.min[0]),Math.min(a.min[1],s.min[1]),Math.min(a.min[2],s.min[2])],max:[Math.max(a.max[0],s.max[0]),Math.max(a.max[1],s.max[1]),Math.max(a.max[2],s.max[2])]}:s:a}function Is(a,t){let e=t.indexOf(".."),s=e===-1?t:t.slice(0,e),i=e===-1?"":t.slice(e+2),r=s===""||a===s||a.startsWith(s+"."),n=i===""||a===i||a.endsWith("."+i);return r&&n}function Vs(a){if(a.startsWith("hs-"))return a;let t=a.trim().split(/\s+/).filter(e=>e.startsWith("hs-"));return t.length?t.length===1?t[0]:"hs-"+t.map(e=>e.slice(3)).join("-"):null}function vt(a,t){for(let e=0;e<t;e++){let s=e*16;a[s+1]=-a[s+1],a[s+2]=-a[s+2];let i=a[s+12],r=a[s+13],n=a[s+14],o=a[s+15];a[s+12]=o,a[s+13]=-n,a[s+14]=r,a[s+15]=-i}}var Je=[-1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1];function At(a,t,e){if(!a||!t)return;let s=Math.min(t,Je.length);for(let i=0;i<e;i++){let r=i*t*3;for(let n=0;n<s;n++){if(Je[n]!==-1)continue;let o=r+n*3;a[o]=-a[o],a[o+1]=-a[o+1],a[o+2]=-a[o+2]}}}var fe=["spz","ply","splat"],Ls={};async function jt(a,t,e=0){let s=a.split("?")[0],i=s.lastIndexOf("."),r=i>=0?s.slice(i+1).toLowerCase():"";if(r==="spzv")return Ye(a,t);let n=fe.includes(r),o=n?[r,...fe.filter(u=>u!==r)]:fe,l=n?a.slice(0,a.lastIndexOf(".")):a,c;for(let u of o){let p=`${l}.${u}`,f=u==="ply"?(d,h)=>De(d,h,e):u==="spz"?(d,h)=>Le(d,h,e):Ls[u]??Te;try{return await f(p,t)}catch(d){if(!/^HTTP 4/.test(d.message))throw d;c=d}}throw new Error(`HoloSplat: splat file not found as .spz / .ply / .splat \u2014 "${l}"`)}function qs(a){return a*a*(3-2*a)}function Ft(a,t,e){return[a[0]+(t[0]-a[0])*e,a[1]+(t[1]-a[1])*e,a[2]+(t[2]-a[2])*e]}function js(a,t,e){let s=new Float32Array(16);for(let i=0;i<16;i++)s[i]=a[i]+(t[i]-a[i])*e;return s}function zt(a,t){let[e,s,i,r]=t,[n,o,l]=a,c=e*2,u=s*2,p=i*2,f=e*c,d=e*u,h=e*p,m=s*u,b=s*p,y=i*p,g=r*c,S=r*u,_=r*p;return new Float32Array([1-m-y,d+_,h-S,0,d-_,1-f-y,b+g,0,h+S,b-g,1-f-m,0,n,o,l,1])}function $s(a){if(!a)throw new Error("HoloSplat: canvas option is required");if(typeof a=="string"){let t=document.querySelector(a);if(!t)throw new Error(`HoloSplat: canvas selector "${a}" not found`);return t}return a}var Ns=`
.hs-player{position:relative;overflow:hidden;}
.hs-player canvas{position:absolute;inset:0;width:100%;height:100%;display:block;}
.hs-player .hs-overlay{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:10px;
  pointer-events:none;font-family:system-ui,sans-serif;
}
.hs-player .hs-pct{font-size:1.1rem;font-weight:600;color:rgba(255,255,255,.85);letter-spacing:.02em;}
.hs-player .hs-bar-wrap{width:140px;height:3px;background:rgba(255,255,255,.1);border-radius:2px;overflow:hidden;}
.hs-player .hs-bar{height:100%;background:#3a7aff;width:0%;transition:width .1s;}
.hs-player .hs-msg{font-size:.78rem;color:rgba(255,255,255,.4);text-align:center;max-width:260px;line-height:1.5;padding:0 16px;}
.hs-player .hs-msg.hs-err{color:#f87;}
.hs-player .hs-callouts{position:absolute;inset:0;pointer-events:none;}
.hs-lines{position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none;}
.hs-dot{fill:#3a7aff;stroke:#fff;stroke-width:2;}
.hs-line{stroke:rgba(255,255,255,.55);stroke-width:1.5;}
.hs-callout{position:absolute;pointer-events:auto;}
.hs-callout--hidden{display:none;}
`,ts=!1;function Ys(){if(ts||typeof document>"u")return;ts=!0;let a=document.createElement("style");a.textContent=Ns,document.head.appendChild(a)}function is(a,t={}){Ys();let e=typeof a=="string"?document.querySelector(a):a;if(!e)throw new Error(`HoloSplat: container not found \u2014 "${a}"`);let s=e.getAttribute("data-holosplat")||void 0,i=e.getAttribute("data-holosplat-anim")||void 0,{scene:r,src:n,parts:o,animation:l=i,clips:c,partsDir:u,partsExt:p="",scenes:f,masks:d,sh:h,background:m="transparent",fov:b=60,near:y=.01,far:g=2e3,splatScale:S=1.08,autoRotate:_=!1,flipY:v=!1,maxPixelRatio:B,quality:x="auto",adaptiveQuality:P=!0,prefetchVariants:k,aaDilation:U=.3,gpuSort:I=!1,zIndex:V=5,onLoad:E,onProgress:T,onError:F}=t,Y=r??n??s,rt=x==="auto"?Vt():x,j=Lt(rt),bt=B??j.maxPixelRatio;e.style.zIndex=String(V),e.classList.add("hs-player");let W=document.createElement("canvas"),N=document.createElement("div");N.className="hs-callouts";let L=document.createElement("div");L.className="hs-overlay",L.innerHTML='<div class="hs-pct">0%</div><div class="hs-bar-wrap"><div class="hs-bar"></div></div><div class="hs-msg"></div>',e.appendChild(W),e.appendChild(N),e.appendChild(L);let Q=L.querySelector(".hs-pct"),ot=L.querySelector(".hs-bar-wrap"),Nt=L.querySelector(".hs-bar"),wt=L.querySelector(".hs-msg"),Yt=null,Et=0;function Wt(){Et++,Q.textContent="0%",Q.style.display="",ot.style.display="",wt.textContent="",wt.className="hs-msg",L.style.display="flex",Yt=performance.now()}function Tt(){Et=Math.max(0,Et-1),Et===0&&(L.style.display="none")}function Qt(w){Q.style.display="none",ot.style.display="none",wt.textContent=w,wt.className="hs-msg hs-err",L.style.display="flex"}L.style.display="none";let C=new xt({canvas:W,background:m,fov:b,near:y,far:g,splatScale:S,autoRotate:_,flipY:v,aaDilation:U,maxPixelRatio:bt,adaptiveQuality:P,prefetchVariants:k??j.prefetchVariants,gpuSort:I,tier:rt,onProgress:w=>{let M=Math.round(w*100);if(Q.textContent=M+"%",Nt.style.width=M+"%",Yt&&w>.05){let G=(performance.now()-Yt)/1e3,O=Math.round(G*(1-w)/w);wt.textContent=O>0?"~"+O+"s":""}T&&T(w)}}),St={},Kt="http://www.w3.org/2000/svg";function cs(w){N.innerHTML="";for(let G of Object.keys(St))delete St[G];if(!w.length)return;let M=document.createElementNS(Kt,"svg");M.setAttribute("class","hs-lines"),N.appendChild(M);for(let G of w){let O=document.createElementNS(Kt,"line");O.setAttribute("class","hs-line"),M.appendChild(O);let R=document.createElementNS(Kt,"circle");R.setAttribute("class","hs-dot"),R.setAttribute("r","5"),M.appendChild(R);let $=e.querySelector(`.hs-callout[data-id="${G.id}"]`)??document.querySelector(`.hs-callout[data-id="${G.id}"]`);$||($=document.createElement("div"),$.className="hs-callout",$.dataset.id=G.id),N.appendChild($),St[G.id]={card:$,dot:R,line:O}}}C.onFrame=()=>{if(!C._animation?.callouts.length)return;let w=C.projectCallouts(C._animation.callouts);for(let{id:M,visible:G,x:O,y:R}of w){let $=St[M];if(!$)continue;let{card:X,dot:st,line:J}=$;if(G){let Ut=parseFloat(X.dataset.offsetX??X.dataset.ox??80),Pt=parseFloat(X.dataset.offsetY??X.dataset.oy??-40),pt=O+Ut,Bt=R+Pt;st.setAttribute("cx",O),st.setAttribute("cy",R),J.setAttribute("x1",O),J.setAttribute("y1",R),J.setAttribute("x2",pt),J.setAttribute("y2",Bt),st.style.display="",J.style.display="",X.style.left=pt+"px",X.style.top=Bt+"px",X.classList.remove("hs-callout--hidden")}else st.style.display="none",J.style.display="none",X.classList.add("hs-callout--hidden")}};async function me(w){Wt(),Nt.style.width="0%";try{await C.load(w),Tt()}catch(M){let G=navigator.gpu?M.message:"WebGPU not supported. Use Chrome 113+ or Edge 113+.";throw Qt(G),F&&F(M),M}}async function Xt(w){Wt(),Nt.style.width="0%";try{await C.loadParts(w),Tt()}catch(M){let G=navigator.gpu?M.message:"WebGPU not supported. Use Chrome 113+ or Edge 113+.";throw Qt(G),F&&F(M),M}}function _e(){let w=Object.keys(C._clips),M=0;for(let G of w){let O=document.getElementById(G);!O||O.dataset.hsClipBound||(O.dataset.hsClipBound="1",O.addEventListener("click",()=>C.playClip(G)),M++)}M&&console.log(`[HoloSplat] ${M} clip button(s) bound:`,w)}async function hs(w){let{url:M,splatsDir:G,defaults:O}=typeof w=="string"?{url:w}:w;Wt();try{await C.loadClips(M,{splatsDir:G,defaults:O,lod:j.lod}),_e()}catch(R){console.error("[HoloSplat] clips failed to load:",R),F&&F(R)}finally{Tt()}}async function Zt(w){try{let M=await C.loadAnimationUrl(w);return cs(M.callouts),console.log(`[HoloSplat] animation loaded: ${M.frameCount} frames @ ${M.fps}fps, ${M.callouts.length} callout(s):`,M.callouts.map(G=>G.id),"| markers:",M.markers),M}catch(M){console.error("[HoloSplat] animation failed to load:",M),F&&F(M)}}let ge={load:me,loadParts:Xt,loadAnim:Zt,destroy(){C.destroy(),e.innerHTML="",e.classList.remove("hs-player")},setBackground(w){C.setBackground(w)},setSplatScale(w){C.setSplatScale(w)},setAutoRotate(w){C.setAutoRotate(w)},setFlipY(w){C.setFlipY(w)},setShDegree(w){return C.setShDegree(w)},setAaDilation(w){C.setAaDilation(w)},setAnimationPaused(w){C.setAnimationPaused(w)},setCameraFree(w){C.setCameraFree(w)},setMaskFeather(w,M){C.setMaskFeather(w,M)},playClip(w){C.playClip(w)},playVariant(w,M){C.playVariant(w,M)},playState(w,M){C.playState(w,M)},async loadClips(w,M){let G=await C.loadClips(w,M);return _e(),G},unloadClips(w){C.unloadClips(w)},getVariants(w){return C.getVariants(w)},setVariant(w,M){return C.setVariant(w,M)},resetCamera(){C.resetCamera()},callout(w){return St[w]?.card??null},get camera(){return C.camera},get animation(){return C._animation},get animationPaused(){return C._animPaused},get clips(){return C._clips}};window.__hsPlayers||(window.__hsPlayers=[]),window.__hsPlayers.push({root:e,api:ge,viewer:C});function us(w){if(!f||!w)return null;history.scrollRestoration&&(history.scrollRestoration="manual"),window.scrollTo(0,0),document.documentElement.style.overflowAnchor="none";let M=15,G=.12,O=Object.entries(w.markers).sort((A,D)=>A[1]-D[1]),R=O.map(([A,D],z)=>{let H=f[A]??{};return{name:A,fromFrame:D,toFrame:(O[z+1]?.[1]??w.frameCount)-1,playback:H.playback??"scroll",pingpong:H.pingpong??!1,playOnce:H.playOnce??!1,blendIn:(H.blendIn??0)/100,blendOut:(H.blendOut??0)/100,done:!1,el:H.linkedId?document.getElementById(H.linkedId):null}});if(R.forEach((A,D)=>{A.next=R[D+1]??null}),!R.some(A=>A.playback!=="auto"))return null;C.camera.allowTouchScroll=!0;function X(A){if(!A.el)return null;let D=A.el.getBoundingClientRect();if(D.top>0)return null;let z=window.scrollY,H=z+D.top,tt=H+Math.max(A.el.offsetHeight,1),mt=document.documentElement.scrollHeight-window.innerHeight,Z=Math.min(tt,Math.max(mt,H+1)),Be=Math.max(0,Math.min(1,(z-H)/(Z-H)));return{t:Be,frame:A.fromFrame+Be*(A.toFrame-A.fromFrame)}}let st=null,J=null,Ut=null,Pt=null,pt=null;function Bt(A){let D=Math.max(0,Math.min(1,A));return D*D*(3-2*D)}function ds(){let A=w.getCameraFrame();J=A.eye,Ut=A.target,Pt={};for(let D of w.getObjectFrames())Pt[D.id]={pos:D.pos,quat:D.quat};pt={};for(let D of w.getVolumeFrames())pt[D.name]=D.matrix}function ps(A,D){if(st!==A.name&&(st=A.name,ds()),A.blendIn>0&&D<A.blendIn){let z=Bt(D/A.blendIn);C._sceneBlend={otherEye:J,otherTarget:Ut,otherObjects:Pt,otherVolumes:pt,bf:z};return}if(A.blendOut>0&&A.next&&D>1-A.blendOut){let z=Bt((D-(1-A.blendOut))/A.blendOut),H=w.getCameraFrame(A.next.fromFrame),tt={};for(let Z of w.getObjectFrames(A.next.fromFrame))tt[Z.id]={pos:Z.pos,quat:Z.quat};let mt={};for(let Z of w.getVolumeFrames(A.next.fromFrame))mt[Z.name]=Z.matrix;C._sceneBlend={otherEye:H.eye,otherTarget:H.target,otherObjects:tt,otherVolumes:mt,bf:1-z};return}C._sceneBlend=null}let it=null,ve=!1,xe=null,be=R.findIndex(A=>A.playback==="scroll"),we=be>0?R[be-1]:null,ms=we?we.toFrame:R[0]?.fromFrame??0;function Se(){let A=null;for(let D of R){if(D.playback!=="scroll"||!D.el)continue;let z=X(D);z!==null&&(A={scene:D,t:z.t,frame:z.frame})}if(A){ps(A.scene,A.t),it={scene:A.scene,frame:A.frame};return}it&&(it=null,st=null,C._sceneBlend=null,w.seekFrame(xe??ms))}function Pe(){ve=!0,Se()}return window.addEventListener("scroll",Pe,{passive:!0}),Se(),C.animTickOverride=A=>{let D=w.frame,z=null;for(let H of R)if(!(H.playOnce&&H.done&&D>=H.toFrame-.5)&&D<=H.toFrame+.5){z=H;break}if(z||(z=R[R.length-1]),!z){w.tick(A);return}if(C._currentSceneName=z.name,z.playback==="auto"){if(!z.playOnce&&ve&&it&&it.frame>z.toFrame+.5){xe=D,w.seekFrame(it.frame);return}if(z.playOnce&&z.done)return;let H=A;if(z.pingpong){let mt=Math.min(D-z.fromFrame,z.toFrame-D),Z=Math.max(G,Math.min(1,mt/M));H=A*Z}w.tick(H);let tt=w.frame;z.pingpong?tt>=z.toFrame?(w.seekFrame(z.toFrame),z.playOnce?z.done=!0:w.direction=-1):tt<=z.fromFrame&&(w.seekFrame(z.fromFrame),w.direction=1):tt<z.fromFrame?w.seekFrame(z.fromFrame):z.playOnce&&tt>=z.toFrame&&(w.seekFrame(z.toFrame),z.done=!0);return}it?w.seekFrame(it.frame):z.el||w.seekFrame(z.fromFrame)},()=>{window.removeEventListener("scroll",Pe),C.animTickOverride=null,C._currentSceneName=null,C._sceneBlend=null}}if(f&&typeof f=="object"&&(window.__hsSceneConfigs=window.__hsSceneConfigs||{},Object.assign(window.__hsSceneConfigs,f)),d&&typeof d=="object"){window.__hsMaskConfigs=window.__hsMaskConfigs||{},Object.assign(window.__hsMaskConfigs,d);for(let[w,M]of Object.entries(d))M&&typeof M.feather=="number"&&C.setMaskFeather(w,M.feather)}let fs=h??(f?Object.values(f).reduce((w,M)=>Math.max(w,M.sh??0),C._shDegree):C._shDegree),ye=Math.min(fs,j.shDegreeCap);return ye!==C._shDegree&&(C._shDegree=ye),C.init().then(async()=>{C.start();let w,M=c?Array.isArray(c)?c:[c]:[],G=Promise.all(M.map(hs));if(u&&l){if(w=await Zt(l),w?.objects.length){let O=u.replace(/\/?$/,"/"),R=Object.fromEntries(w.objects.map($=>[$.id,`${O}${le($.id)}${p}`]));R=await dt(R,j.lod),await Xt(R)}}else{let O=[];o?O.push(dt(o,j.lod).then(Xt)):Y&&O.push(et(Y,j.lod).then(me)),l&&O.push(Zt(l).then(R=>{w=R})),await Promise.all(O)}await G,Tt(),E?.(),us(w??C._animation)}).catch(w=>{navigator.gpu||Qt("WebGPU not supported. Use Chrome 113+ or Edge 113+.")}),ge}function es(){document.querySelectorAll("[data-holosplat]").forEach(a=>{if(a._hsPlayer)return;let t=a.getAttribute("data-holosplat")||void 0,e=a.getAttribute("data-holosplat-anim")||void 0,s=a.getAttribute("data-holosplat-parts")||void 0;a._hsPlayer=is(a,{src:t,animation:e,partsDir:s})})}function ss(){if(typeof location>"u"||!new URLSearchParams(location.search).has("hs")||document.getElementById("__hs-script"))return;let a=window.matchMedia("(pointer: coarse)").matches||window.matchMedia("(max-width: 768px)").matches||/Android|iPhone|iPad|iPod/i.test(navigator.userAgent||""),t=document.createElement("script");t.id="__hs-script",t.src=a?"/holosplat/stats.js":"/holosplat/editor.js",document.head.appendChild(t)}typeof document<"u"&&(document.readyState==="loading"?document.addEventListener("DOMContentLoaded",()=>{es(),ss()}):(es(),ss()));var Ws=`
.hs-scene {
  position: relative;
  width: 100%;
}
/* .hs-stage also carries .hs-player (position:relative) from player.js.
   !important ensures sticky wins regardless of injection order.
   100vw fills the full viewport regardless of parent padding/margin. */
.hs-stage {
  position: sticky !important;
  top: 0;
  left: 0;
  width: 100vw !important;
  height: 100vh !important;
  z-index: 1;
  overflow: hidden;
}
.hs-stage canvas {
  position: absolute !important;
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
  display: block;
}
.hs-track {
  /* Overlap the sticky stage so acts start scrolling at the same time the
     canvas appears \u2014 not after it. */
  margin-top: -100vh;
  position: relative;
  /* Clicks pass through to the canvas; restore per-element as needed. */
  pointer-events: none;
  z-index: 2;
}
.hs-act,
.hs-hold {
  position: relative;
}
.hs-caption {
  pointer-events: auto;
}
.hs-caption--hidden {
  opacity: 0;
  pointer-events: none;
}
`,as=!1;function Qs(){if(as||typeof document>"u")return;as=!0;let a=document.createElement("style");a.textContent=Ws,document.head.appendChild(a)}function pe(a,t,e=0){if(a==null||a==="")return e;let s=String(a).trim();if(Object.prototype.hasOwnProperty.call(t,s))return t[s];let i=parseFloat(s);return isNaN(i)?(console.warn(`[HoloSplat] scrollScene: unknown marker "${s}" \u2014 using ${e}`),e):i}function Ks(a,t,e){if(a.classList.contains("hs-hold"))return{el:a,type:"hold",frame:pe(a.dataset.frame,t,0),captions:ns(a)};let i=a.dataset.from??"",r=i==="pingpong-start",n=i==="freecamera-start",o=pe(i,t,0),l=pe(a.dataset.to,t,e),c=0;if(a.dataset.loop!==void 0){let u=a.dataset.loop.trim();c=u===""||u==="true"?1:Math.max(1,parseFloat(u)||1)}return{el:a,type:r?"pingpong":n?"freecamera":"act",from:o,to:l,loop:c,captions:ns(a)}}function ns(a){return[...a.querySelectorAll(".hs-caption")].map(t=>({el:t,at:parseFloat(t.dataset.at??"0")}))}function rs(a,t){if(a.type==="hold")return a.frame;if(a.type==="pingpong"||a.type==="freecamera")return t>=1?a.to:a.from;let{from:e,to:s,loop:i}=a,r=s-e;if(r===0)return e;let n=t;return i>0&&(n=t*i%1),e+n*r}function Xs(a){let t=a.getBoundingClientRect();if(t.top>0)return 0;let e=window.scrollY,s=e+t.top,i=a.offsetHeight||1,r=s+i,n=document.documentElement.scrollHeight-window.innerHeight,o=Math.min(r,Math.max(n,s+1));return Math.max(0,Math.min(1,(e-s)/(o-s)))}function Zs(a,t,e){return a+(t-a)*e}function Js(a,t,e){let s=Math.max(0,Math.min(1,(e-a)/(t-a)));return s*s*(3-2*s)}function ls(a,t,e={}){Qs();let s=a.querySelector(".hs-track");if(!s)return console.warn("[HoloSplat] scrollScene: .hs-track not found inside scene"),{rebuild(){},destroy(){}};let i=[],r=!1,n=1,o=0,l=0,c=0,u=null,p=null,f=e.pingpongTransition??.25,d=!1,h=0,m=!1;function b(x,P,k=!1){r||(d=!1,r=!0,l=x,c=P,n=k?-1:1,o=k?P:x,u=null,p=requestAnimationFrame(g))}function y(){r=!1,p!==null&&(cancelAnimationFrame(p),p=null)}function g(x){if(!r)return;if(p=requestAnimationFrame(g),u===null){u=x;return}let P=(x-u)/1e3;u=x;let k=t.animation?.fps??24;o+=n*k*P,o>=c&&(o=c,n=-1),o<=l&&(o=l,n=1),t.animation.seekFrame(o)}function S(){let x=t.animation,P=x?.markers??{},k=x?x.frameCount-1:0;i=[...s.children].map(U=>Ks(U,P,k)),x&&t.setAnimationPaused(!0)}function _(){if(!t.animation||!i.length)return;let x=rs(i[0],0),P=i[0],k=0;for(let E of i){let T=Xs(E.el);if(T<=0||(x=rs(E,T),P=E,k=T,T<1))break}let U=P.type==="pingpong"&&k>0&&k<1,I=P.type==="freecamera"&&k>0&&k<1;if(U?(d=!1,r||b(P.from,P.to,k>.5)):(r&&(h=o,d=!0),y()),I?m||(m=!0,t.setCameraFree(!0),t.camera.enabled=!0,t.camera.panEnabled=!1):m&&(m=!1,t.setCameraFree(!1),t.camera.enabled=!1,t.camera.panEnabled=!0),!U&&!I)if(d){let E=Js(0,f,k);E>=1&&(d=!1),t.animation.seekFrame(Zs(h,x,E))}else t.animation.seekFrame(x);let V=P?.el?.id??"\u2014";if(_._activeId!==V){_._activeId=V;let E=P,T=E.type==="hold"?`hold @ frame ${E.frame}`:E.type==="pingpong"?`pingpong [${E.from} \u2194 ${E.to}]`:`act [${E.from} \u2192 ${E.to}]`;console.log(`[HoloSplat] active: ${V} (${T}) | current frame: ${x.toFixed(1)}`)}for(let E of i){if(!E.captions.length)continue;let T=E===P;for(let F of E.captions){let Y=T&&k>=F.at;F.el.classList.toggle("hs-caption--hidden",!Y)}}}let v=!1;function B(){v||(v=!0,requestAnimationFrame(()=>{v=!1,_()}))}return window.addEventListener("scroll",B,{passive:!0}),function x(){if(!t.animation){requestAnimationFrame(x);return}t.camera.enabled=!1,e.onReady&&e.onReady(t.animation),S(),_()}(),{rebuild(){y(),d=!1,m&&(m=!1,t.setCameraFree(!1),t.camera.panEnabled=!0),S(),_()},destroy(){y(),d=!1,m&&(m=!1,t.setCameraFree(!1),t.camera.panEnabled=!0),window.removeEventListener("scroll",B),t.camera.enabled=!0,t.setAnimationPaused(!1)}}}function os(){document.querySelectorAll(".hs-scene").forEach(a=>{if(a._hsScroll)return;let t=a.querySelector(".hs-stage");t&&t._hsPlayer&&(a._hsScroll=ls(a,t._hsPlayer))})}typeof document<"u"&&(document.readyState==="loading"?document.addEventListener("DOMContentLoaded",os):os());function $t(a,t){let e=t*16,s=a[e+7],i=a[e+8],r=a[e+9],n=a[e+10];return s*Math.cbrt(Math.max(0,i*r*n))}function ti(a,t,e={}){let{minAlpha:s=0,minScale:i=0,keepFraction:r=1}=e,n=[];for(let l=0;l<t;l++){let c=l*16;a[c+7]<s||Math.max(a[c+8],a[c+9],a[c+10])<i||n.push(l)}r<1&&n.length>0&&(n.sort((l,c)=>$t(a,c)-$t(a,l)),n.length=Math.max(1,Math.round(n.length*r)));let o=new Float32Array(n.length*16);for(let l=0;l<n.length;l++)o.set(a.subarray(n[l]*16,n[l]*16+16),l*16);return{data:o,count:n.length}}function ei(a,t,e={}){let{minAlpha:s=0,minScale:i=0,fractions:r=[1,.8,.6,.4]}=e,n=[];for(let o=0;o<t;o++){let l=o*16;a[l+7]<s||Math.max(a[l+8],a[l+9],a[l+10])<i||n.push(o)}return n.sort((o,l)=>$t(a,l)-$t(a,o)),r.map(o=>{let l=Math.max(1,Math.round(n.length*o)),c=new Float32Array(l*16);for(let u=0;u<l;u++)c.set(a.subarray(n[u]*16,n[u]*16+16),u*16);return{data:c,count:l,fraction:o}})}var si={light:{minAlpha:.004,minScale:1e-4,keepFraction:1},balanced:{minAlpha:.02,minScale:5e-4,keepFraction:.7},aggressive:{minAlpha:.05,minScale:.001,keepFraction:.4},mobile:{minAlpha:.05,minScale:.001,keepFraction:.25}};async function ta(a={}){let{onLoad:t,onError:e,src:s,parts:i,quality:r="auto",...n}=a,o=r==="auto"?Vt():r,l=Lt(o),c=n.maxPixelRatio??l.maxPixelRatio,u=Math.min(n.shDegree??0,l.shDegreeCap),p=n.prefetchVariants??l.prefetchVariants,f=new xt({...n,maxPixelRatio:c,shDegree:u,prefetchVariants:p,tier:o}),d={destroy(){},setBackground(){},setSplatScale(){},setGamma(){},setAaDilation(){},setAutoRotate(){},setFlipY(){},resetCamera(){},focusCamera(){},getVariants(){return[]},async setVariant(){return!1},getStats(){return null},getSplatDebug(){return null},camera:null,setDebugIndex(){},async readDebug(){return null}};try{await f.init(),i?await f.loadParts(await dt(i,l.lod)):s&&await f.load(await et(s,l.lod))}catch(h){if(f.destroy(),e)return e(h),d;throw h}return f.start(),t?.(),{destroy(){f.destroy()},setBackground(h){f.setBackground(h)},setSplatScale(h){f.setSplatScale(h)},setGamma(h){f.setGamma(h)},setAaDilation(h){f.setAaDilation(h)},setAutoRotate(h){f.setAutoRotate(h)},setFlipY(h){f.setFlipY(h)},resetCamera(){f.resetCamera()},focusCamera(){f.focusCamera()},getVariants(h){return f.getVariants(h)},getStats(){return f.getStats()},getSplatDebug(h){return f.getSplatDebug(h)},setDebugIndex(h){f.setDebugIndex(h)},readDebug(){return f.readDebug()},setVariant(h,m){return f.setVariant(h,m)},get camera(){return f.camera}}}export{It as Animation,si as PRUNE_PRESETS,xt as Viewer,As as compressToSpz,Es as compressVariantsToSpzv,ta as create,Vt as detectDeviceTier,je as encodeSpz,$e as encodeSpzv,ei as generateLods,ue as loadAnimation,Re as parsePly,Ue as parseSplat,ae as parseSpz,Bs as parseSpzGzip,oe as parseSpzv,Us as parseSpzvGzip,is as player,ti as pruneGaussians,Lt as qualityForTier,et as resolveLodUrl,dt as resolvePartsLod,ls as scrollScene,le as splatNameFromId};
//# sourceMappingURL=holosplat.esm.js.map
