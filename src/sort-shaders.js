/**
 * GPU compute-shader radix sort kernels for back-to-front depth ordering.
 *
 * Mirrors the CPU algorithm in src/sorter.js: 4 passes of 8-bit-bucket radix
 * sort over depth keys, where view-space z (always negative) is converted to
 * a sortable u32 by flipping all bits of its IEEE-754 representation.
 *
 * Pipeline (see Renderer#runGpuSort):
 *   cs_depth_key   - one thread per splat: computes view-space depth, writes
 *                    the initial key/index arrays (keysA/idxA).
 *   cs_histogram   - per 256-element workgroup: counts the current pass's
 *                    byte into a per-workgroup 256-bin histogram.
 *   cs_scan_reduce - per 256-workgroup chunk: turns per-workgroup histograms
 *                    into chunk-local per-bin exclusive prefixes (blockPfx)
 *                    and per-chunk per-bin totals (chunkTotal).
 *   cs_scan_global - single workgroup: scans chunkTotal into per-chunk
 *                    per-bin offsets (chunkPfx) and global per-bin exclusive
 *                    prefixes (globalPfx).
 *   cs_scan_combine- per 256-element workgroup: adds chunkPfx into blockPfx
 *                    so it becomes globally-relative.
 *   cs_scatter     - per 256-element workgroup: scatters each element to its
 *                    sorted position using globalPfx + blockPfx + a stable
 *                    intra-workgroup rank.
 *
 * cs_scan is split into reduce/global/combine so no single dispatch does
 * O(numWorkgroups) sequential work - each stage is bounded to ~256
 * iterations regardless of scene size, which matters on mobile GPU drivers
 * with aggressive watchdog (TDR) timeouts.
 *
 * Buffers ping-pong between the "A" and "B" key/index arrays - each kernel
 * picks source/destination by sortParams.passIndex parity via the readKey/writeKey
 * helpers below, so no bind-group rebuilds are needed between passes. After 4
 * passes (even), the result lands back in idxA, which is _orderBuf.
 */

export const SORT_SHADER = /* wgsl */`

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

// ── Ping-pong helpers: pass even -> A is source, B is dest; pass odd -> reversed ──

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

// ── cs_depth_key: seed keys/indices from view-space depth ───────────────────

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

// ── cs_histogram: per-workgroup 256-bin histogram of the current byte ───────

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

// ── cs_scan_reduce: per-chunk per-bin exclusive prefixes + chunk totals ─────
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

// ── cs_scan_global: scan chunk totals + compute global per-bin offsets ─────
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

// ── cs_scan_combine: fold chunk offsets into blockPfx ───────────────────────
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

// ── cs_scatter: stable scatter to sorted positions ───────────────────────────

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
`;

// Elements processed by each 256-thread workgroup (1 element/thread) for
// cs_depth_key, cs_histogram and cs_scatter dispatches.
export const ELEMS_PER_WG = 256;

/** Number of 256-element workgroups needed to cover `count` elements. */
export function numWorkgroups(count) {
  return Math.ceil(count / ELEMS_PER_WG);
}

/** Number of 256-workgroup chunks cs_scan_reduce/cs_scan_global operate over. */
export function numChunks(count) {
  return Math.ceil(numWorkgroups(count) / 256);
}
