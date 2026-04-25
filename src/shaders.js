// WGSL shader for Gaussian splatting.
//
// GPU layout (must match CPU Float32Array layout in loaders):
//
//   struct Gaussian {         byte offset
//     pos   : vec3<f32>,   //  0  (12 bytes + 4 implicit WGSL padding = 16)
//     color : vec4<f32>,   // 16  (rgba, premultiplied ready)
//     scale : vec3<f32>,   // 32  (12 bytes + 4 implicit padding = 16)
//     quat  : vec4<f32>,   // 48  (xyzw)
//   }; // stride = 64 bytes = 16 floats/gaussian
//
// Uniform layout (160 bytes = 40 floats):
//   [0-15]  view matrix (col-major)
//   [16-31] proj matrix (col-major)
//   [32-33] viewport (width, height) in pixels
//   [34-35] focal (fx, fy) in pixels
//   [36]    splatScale multiplier
//   [37-39] padding

export const SHADER = /* wgsl */`

struct Uniforms {
  view       : mat4x4<f32>,
  proj       : mat4x4<f32>,
  viewport   : vec2<f32>,
  focal      : vec2<f32>,
  params     : vec4<f32>,  // .x = splatScale  .y = near (view-space units)
};

struct Gaussian {
  pos   : vec3<f32>,
  color : vec4<f32>,
  scale : vec3<f32>,
  quat  : vec4<f32>,
};

@group(0) @binding(0) var<uniform>       uniforms  : Uniforms;
@group(0) @binding(1) var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(2) var<storage, read> order     : array<u32>;

struct VOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) color         : vec4<f32>,
  @location(1) uv            : vec2<f32>,
  @location(2) conic         : vec3<f32>,
};

// Quaternion (xyzw) → column-major rotation matrix
fn quatToMat3(q: vec4<f32>) -> mat3x3<f32> {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0*(y*y+z*z),  2.0*(x*y+w*z),       2.0*(x*z-w*y)),
    vec3<f32>(2.0*(x*y-w*z),         1.0 - 2.0*(x*x+z*z), 2.0*(y*z+w*x)),
    vec3<f32>(2.0*(x*z+w*y),         2.0*(y*z-w*x),       1.0 - 2.0*(x*x+y*y))
  );
}

// Degenerate vertex (moved outside clip space → triangle discarded)
fn degen() -> VOut {
  var o: VOut;
  o.clipPos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
  o.color   = vec4<f32>(0.0);
  o.uv      = vec2<f32>(0.0);
  o.conic   = vec3<f32>(0.0);
  return o;
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

  let gi = order[ii];
  let g  = gaussians[gi];
  let corner = corners[vi];

  // ── Transform to view space ──────────────────────────────────────────────
  let viewPos4 = uniforms.view * vec4<f32>(g.pos, 1.0);
  let t = viewPos4.xyz;

  // Discard if at or behind near plane
  let near = uniforms.params.y;
  if t.z > -near { return degen(); }

  // Near-depth fade: fade out splats within 3× the near distance so that
  // close-up splats don't cover the entire screen (matches Blender's behaviour).
  let depth     = -t.z;
  let nearFade  = clamp((depth - near) / (near * 2.0), 0.0, 1.0);

  // ── 3-D covariance ────────────────────────────────────────────────────────
  let splatScale = uniforms.params.x;
  let s = g.scale * splatScale;
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
  // View rotation (upper-left 3×3 of view matrix, column-major)
  let W = mat3x3<f32>(
    uniforms.view[0].xyz,
    uniforms.view[1].xyz,
    uniforms.view[2].xyz
  );
  let T    = J * W;
  let cov2 = T * cov3 * transpose(T);

  // Extract 2×2 + low-pass filter (anti-aliasing)
  let a = cov2[0][0] + 0.3;   // Σ_xx  (cov2[col][row])
  let b = cov2[0][1];          // Σ_xy
  let c = cov2[1][1] + 0.3;   // Σ_yy

  // ── Inverse covariance (conic) ────────────────────────────────────────────
  let det = a*c - b*b;
  if det < 1e-4 { return degen(); }
  let inv = 1.0 / det;
  // conic = (A, B, C) such that Mahalanobis² = A·dx² + 2B·dx·dy + C·dy²
  let conic = vec3<f32>(c*inv, -b*inv, a*inv);

  // ── Bounding radius (3σ of largest eigenvalue) ────────────────────────────
  let mid      = 0.5 * (a + c);
  let disc     = sqrt(max(0.1, mid*mid - det));
  let rawRadius = ceil(3.0 * sqrt(mid + disc));

  // Clamp screen-space radius: splats larger than half the viewport height
  // are faded out and capped, preventing nearby splats from covering the screen.
  let maxRadius = uniforms.viewport.y * 0.5;
  let sizeFade  = clamp(maxRadius / max(rawRadius, 1.0), 0.0, 1.0);
  let radius    = min(rawRadius, maxRadius);

  // ── Screen-space quad placement ───────────────────────────────────────────
  let clip    = uniforms.proj * viewPos4;
  let ndcXY   = clip.xy / clip.w;
  let pixOff  = corner * radius;
  let ndcOff  = pixOff / (uniforms.viewport * 0.5);

  var o: VOut;
  o.clipPos = vec4<f32>(ndcXY + ndcOff, clip.z / clip.w, 1.0);
  o.color   = vec4<f32>(g.color.rgb, g.color.a * nearFade * sizeFade);
  o.uv      = pixOff;
  o.conic   = conic;
  return o;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let d = in.uv;
  let power = -0.5 * (in.conic.x*d.x*d.x + 2.0*in.conic.y*d.x*d.y + in.conic.z*d.y*d.y);
  if power > 0.0 { discard; }
  let alpha = in.color.a * exp(power);
  if alpha < 1.0/255.0 { discard; }
  // Premultiplied alpha output (blend: src=ONE, dst=ONE_MINUS_SRC_ALPHA)
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
`;
