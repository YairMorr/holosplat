/**
 * WebGPU renderer for Gaussian splatting.
 *
 * Manages device, pipeline, bind groups, and GPU buffers.
 * Upload happens once (gaussians); order buffer is updated every frame.
 *
 * Buffer layout (must match src/shaders.js):
 *   Uniforms  : 160 bytes (40 × f32)
 *   Gaussians : N × 64 bytes (16 × f32 each)
 *   Order     : N × 4 bytes (u32 indices, sorted back-to-front)
 */
import { SHADER } from './shaders.js';
import { SORT_SHADER, numWorkgroups, numChunks } from './sort-shaders.js';

// Fullscreen blit: copies the HDR (rgba16float) composited image to the
// canvas's swapchain format. Splats are blended directly into an 8-bit
// canvas in most simple renderers — but a single pixel can be covered by
// dozens of overlapping translucent splats, and 8-bit blending accumulates
// rounding error on every single blend. Compositing in float and quantizing
// only once, at the very end, avoids that.
const BLIT_SHADER = /* wgsl */`
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
`;

// Uniform buffer offsets (in float32 index units)
const U_VIEW      = 0;   // mat4 (16 floats)
const U_PROJ      = 16;  // mat4 (16 floats)
const U_VIEWPORT  = 32;  // vec2 (2 floats)
const U_FOCAL     = 34;  // vec2 (2 floats)
const U_PARAMS    = 36;  // vec4 (4 floats), .x = splatScale  .w = radiusCap
const U_SH_PARAMS   = 40;  // vec4 (4 floats), .x = shDegree  .y = numSHBases  .z = aaDilation
const U_AA_DILATION = 42;  // within shParams.z
const U_SIZE        = 44;  // total floats → 176 bytes

export class Renderer {
  constructor(canvas, background) {
    this.canvas     = canvas;
    this.background = parseBackground(background);
    this.device     = null;
    this.context    = null;
    this.pipeline   = null;
    this.bindGroup  = null;
    this._mainModule = null; // shared SHADER module — see _createPipeline/_createPreprocessPipeline

    // HDR (rgba16float) intermediate target + blit pipeline — see BLIT_SHADER.
    this._hdrFormat   = 'rgba16float';
    this._hdrTexture  = null;
    this._hdrView     = null;
    this._blitPipeline  = null;
    this._blitSampler   = null;
    this._blitBindGroup = null;

    // GPU buffers
    this._uniformBuf     = null;
    this._gaussianBuf    = null;
    this._orderBuf       = null;  // also bound as idxA for the GPU sort
    this._transformBuf   = null;
    this._maskVolBuf     = null;  // MaskUniforms (uniform, 656 bytes)
    this._partVolMaskBuf = null;  // u32 per part (storage)

    // Per-splat preprocess (see shaders.js cs_preprocess) — covariance/eigen/
    // SH math computed once per splat instead of once per vertex. _splatGeomBuf
    // holds the SplatGeom struct (48 bytes/splat) the compute pass writes and
    // vs_main reads; _preprocessParamsBuf carries the live splat count.
    this._splatGeomBuf        = null;
    this._preprocessParamsBuf = null;
    this._preprocessPipeline  = null;
    this._preprocessBindGroup = null;
    this._preprocessCountData = new Uint32Array(4); // scratch for preprocess()'s writeBuffer

    // SH coefficients buffer — null when shDegree=0; rebuilt by uploadSH/allocateSH.
    // _shDummyBuf is a single vec3<f32> bound in place of _shBuf when no SH data exists.
    this._shBuf      = null;
    this._shDummyBuf = null;
    this._shNumBases = 0;

    // GPU radix sort buffers/pipelines (see src/sort-shaders.js) — sized for
    // the current splat count in uploadGaussians; _globalPfxBuf is fixed-size.
    this._keysBufA      = null;
    this._keysBufB      = null;
    this._idxBufB       = null;
    this._histoBuf      = null;
    this._blockPfxBuf   = null;
    this._globalPfxBuf  = null;
    this._chunkTotalBuf = null;
    this._chunkPfxBuf   = null;
    this._sortParamsBuf = null;
    this._sortParamsStaging = null;
    // 5 slots x {numElements, passIndex, numWorkgroups, numChunks}: slot 0 for
    // cs_depth_key (passIndex unused), slots 1-4 for the 4 radix passes — see
    // runGpuSort.
    this._sortParamsData = new Uint32Array(20);

    this._depthKeyPipeline    = null;
    this._histogramPipeline   = null;
    this._scanReducePipeline  = null;
    this._scanGlobalPipeline  = null;
    this._scanCombinePipeline = null;
    this._scatterPipeline     = null;
    this._depthKeyBindGroup    = null;
    this._histogramBindGroup   = null;
    this._scanReduceBindGroup  = null;
    this._scanGlobalBindGroup  = null;
    this._scanCombineBindGroup = null;
    this._scatterBindGroup     = null;

    // Set true by the uncapturederror handler (see init) if a GPU validation
    // error occurs — Viewer#_tick stops calling runGpuSort once this is set,
    // falling back to the CPU sorter without crashing.
    this._gpuSortFailed = false;

    // CPU-side uniform data — shared ArrayBuffer so float and u32 views alias.
    this._uniforms    = new Float32Array(U_SIZE);
    this._uniformsU32 = new Uint32Array(this._uniforms.buffer);
    this._uniforms[U_PARAMS]     = 1.08; // splatScale default — slightly >1 to close coverage gaps on sparse/specular regions (overwritten by Viewer#setSplatScale on init)
    this._uniforms[U_PARAMS + 2] = 1.0;  // gamma = 1.0 (linear)
    this._uniforms[U_PARAMS + 3] = 1.0;  // radiusCap = 1.0 × viewport.y (safety net, not a routine constraint — see Viewer)
    // shParams: degree=0, numBases=0 (no SH), aaDilation=0.3 (matches PlayCanvas/SuperSplat's gsplat shader)
    this._uniforms[U_AA_DILATION] = 0.3;

    this._numSplats = 0;
    this._maskWarned = new Set();
  }

  // ── Initialise WebGPU ──────────────────────────────────────────────────────

  async init() {
    if (!navigator.gpu) throw new Error('WebGPU is not supported in this browser.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter found.');
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize:               adapter.limits.maxBufferSize,
      },
    });

    // Defensive diagnostics: a GPU validation error (e.g. from a future
    // sort-shader bug) disables GPU sort for subsequent frames instead of
    // spamming errors or relying on the page to crash. device.lost can't be
    // recovered from here (would need to recreate the device + all GPU
    // resources) — just log it clearly for diagnosis.
    this.device.addEventListener('uncapturederror', (event) => {
      console.error('[HoloSplat] GPU error:', event.error);
      this._gpuSortFailed = true;
    });
    this.device.lost.then((info) => {
      console.error('[HoloSplat] WebGPU device lost:', info.reason, info.message);
    });

    this.context = this.canvas.getContext('webgpu');
    this._format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this._format,
      alphaMode: 'premultiplied',
    });

    this._createPipeline();
    this._createBlitPipeline();
    this._createSortPipelines();
    this._createPreprocessPipeline();
    this._uniformBuf = this._createBuffer(U_SIZE * 4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this._preprocessParamsBuf = this._createBuffer(16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    // Dummy SH buffer — one vec3<f32> — bound when shDegree=0 so the bind
    // group is always valid even before SH data is uploaded.
    this._shDummyBuf = this._createBuffer(12, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    // TEMP debug instrumentation — see shaders.js binding(7) / setDebugIndex.
    this._debugBuf     = this._createBuffer(16 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this._debugReadBuf = this._createBuffer(16 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    this._uniforms[U_SH_PARAMS + 3] = -1; // disabled by default

    // GPU sort: fixed-size buffers (256 bins) — created once.
    this._globalPfxBuf  = this._createBuffer(256 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this._sortParamsBuf = this._createBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    // Staging buffer for all 5 SortParams slots — copied into _sortParamsBuf
    // slot-by-slot inside the command encoder (see runGpuSort). Avoids the
    // need for separate queue.submit() calls or per-pass bind groups, since
    // queue.writeBuffer() calls all land before any encoded dispatch runs.
    this._sortParamsStaging = this._createBuffer(16 * 5, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    // Default transforms buffer: one identity mat4 (single-part scenes)
    const identity = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
    this._transformBuf = this._createBuffer(64, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(this._transformBuf, 0, identity);

    // Mask volume buffers — zeroed by default (count=0 → no masking)
    // Layout: 16-byte header (vec4<u32>, .x = count) + 8 × MaskVolume (80 bytes each) = 656 bytes
    this._maskVolBuf = this._createBuffer(656, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    // One u32 per part; zero means no volumes affect this part
    this._partVolMaskBuf = this._createBuffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  }

  // ── Upload scene data (call once after load) ───────────────────────────────

  uploadGaussians(data, count) {
    this._numSplats = count;

    // Gaussian storage buffer (read-once upload). Deliberately NOT calling
    // .destroy() on the outgoing buffer here: uploadGaussians can run
    // mid-session (setFlipY, variant swaps that change splat count) while
    // the render loop's runGpuSort still has commands for the *previous*
    // buffers in flight on the queue. Destroying a buffer still referenced
    // by unfinished submitted work raises a GPU validation error, which
    // permanently disables GPU sort for the rest of the session (see
    // _gpuSortFailed in init()/uncapturederror). Dropping the reference and
    // letting it get GC'd once the GPU is done with it is the safe pattern.
    this._gaussianBuf = this.device.createBuffer({
      size:  data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this._gaussianBuf, 0, data);

    // Order buffer (rewritten every frame). Also bound as idxA for the GPU
    // sort (COPY_SRC so debugReadOrder can read it back for verification).
    this._orderBuf = this.device.createBuffer({
      size:  count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Per-splat preprocess output (see shaders.js SplatGeom) — 48 bytes/splat
    // (vec4 ndc + vec2 v1 + vec2 v2 + vec4 color). Written by cs_preprocess,
    // read by vs_main. Same in-flight-GPU-work hazard as above — no .destroy().
    this._splatGeomBuf = this.device.createBuffer({
      size:  Math.max(count * 48, 48),
      usage: GPUBufferUsage.STORAGE,
    });

    // GPU sort scratch buffers, sized for this splat count. See the
    // .destroy() note above — same in-flight-GPU-work hazard applies here.
    const u32Buf = (n) => this.device.createBuffer({
      size: Math.max(n * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._keysBufA    = u32Buf(count);
    this._keysBufB    = u32Buf(count);
    this._idxBufB     = u32Buf(count);
    const nwg = numWorkgroups(count);
    const nch = numChunks(count);
    this._histoBuf      = u32Buf(nwg * 256);
    this._blockPfxBuf   = u32Buf(nwg * 256);
    this._chunkTotalBuf = u32Buf(nch * 256);
    this._chunkPfxBuf   = u32Buf(nch * 256);

    this._rebuildBindGroup();
    this._rebuildPreprocessBindGroup();
    this._rebuildSortBindGroups();
  }

  // ── Per-frame updates ──────────────────────────────────────────────────────

  updateUniforms({ view, proj, width, height, focal, near = 0.01, radiusCap }) {
    const u = this._uniforms;
    u.set(view,   U_VIEW);
    u.set(proj,   U_PROJ);
    u[U_VIEWPORT]     = width;
    u[U_VIEWPORT + 1] = height;
    u[U_FOCAL]        = focal;
    u[U_FOCAL + 1]    = focal;
    u[U_PARAMS + 1]   = near;
    if (radiusCap != null) u[U_PARAMS + 3] = radiusCap;
    this.device.queue.writeBuffer(this._uniformBuf, 0, u);
  }

  updateOrder(sortedIndices, count) {
    // Defensive clamp against two independent size mismatches, either of
    // which makes the requested write exceed a buffer's actual bytes:
    //   - _orderBuf (destination) is sized as of the last uploadGaussians().
    //   - sortedIndices (source) comes from the async CPU sorter
    //     (sorter.js's createSorter), which can return a *stale* result from
    //     a previous, smaller scene's sort while the worker is still busy
    //     computing the first sort for a newly-loaded, larger scene.
    // Either way, writeBuffer throws (crashing the render loop on every
    // subsequent frame, since _tick doesn't catch) rather than just
    // producing a wrong-for-one-frame sort order. Skipping a frame's sort
    // update is harmless; crashing the whole render loop is not.
    const n = Math.min(count, this._orderBuf.size / 4, sortedIndices.length);
    if (n < count) {
      console.warn(`[HoloSplat] updateOrder: requested count (${count}) exceeds available buffer/source size (${n}) — clamping this frame's sort update`);
    }
    if (n <= 0) return;
    this.device.queue.writeBuffer(
      this._orderBuf, 0,
      sortedIndices.buffer, 0, n * 4
    );
  }

  /** Upload a new transforms buffer (array of Float32Array(16), one mat4 per part).
   *  Rebuilds the bind group — call once after uploadGaussians. */
  uploadTransforms(transforms) {
    const flat = new Float32Array(transforms.length * 16);
    for (let i = 0; i < transforms.length; i++) flat.set(transforms[i], i * 16);
    // No .destroy() on the outgoing buffer — see uploadGaussians for why.
    this._transformBuf = this._createBuffer(
      flat.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(this._transformBuf, 0, flat);
    if (this._gaussianBuf) {
      this._rebuildPreprocessBindGroup();
      // _depthKeyBindGroup also references _transformBuf — rebuild GPU sort
      // bind groups too, or they'd point at the just-destroyed buffer.
      this._rebuildSortBindGroups();
    }
  }

  /** Write updated transform data each frame without reallocating (transforms array must
   *  be the same length as the last uploadTransforms call). */
  updateTransforms(flat) {
    if (this._transformBuf) this.device.queue.writeBuffer(this._transformBuf, 0, flat);
  }

  /** Upload per-part volume bitmask (one u32 per part, bit i = affected by volume i).
   *  Call after loading parts or attaching a new animation with volumes. */
  uploadPartVolumeMask(masks) {
    // No .destroy() on the outgoing buffer — see uploadGaussians for why.
    this._partVolMaskBuf = this._createBuffer(
      Math.max(masks.byteLength, 4),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(this._partVolMaskBuf, 0, masks);
    if (this._gaussianBuf) this._rebuildPreprocessBindGroup();
  }

  /** Write current-frame inverse matrices for all active volumes.
   *  @param {Array<{matrix: Float32Array(16), softEdge: number}>} volumes */
  updateMaskVolumes(volumes) {
    const count = Math.min(volumes.length, 8);
    const arrBuf = new ArrayBuffer(656);
    const u32v   = new Uint32Array(arrBuf);
    const f32v   = new Float32Array(arrBuf);

    u32v[0] = count;  // header.x = count (bytes 0-3); bytes 4-15 = 0 (pad)

    for (let i = 0; i < count; i++) {
      const byteBase = 16 + i * 80;
      const f32Base  = byteBase >> 2;  // / 4
      const vol = volumes[i];
      const inv = invertMat4(vol.matrix);
      if (inv) {
        f32v.set(inv, f32Base);       // invMatrix (16 floats)
      } else if (!this._maskWarned.has(vol.name)) {
        this._maskWarned.add(vol.name);
        console.warn(`[HoloSplat] mask volume "${vol.name}" has a non-invertible (degenerate) transform — it will have no effect. Check that its Blender object has non-zero scale on all 3 axes.`);
      }
      f32v[f32Base + 16] = vol.softEdge ?? 0.05;  // params.x
      // params.yzw = 0 (padding, already zero)
    }

    this.device.queue.writeBuffer(this._maskVolBuf, 0, arrBuf);
  }

  /** Patch a region of the Gaussian buffer in-place (no reallocation).
   *  Used for progressive streaming: call after uploadGaussians to fill in chunks.
   *  @param {Float32Array} data          chunk of decoded splats (n × 16 floats)
   *  @param {number}       vertexOffset  first vertex index to overwrite */
  patchGaussians(data, vertexOffset) {
    if (!this._gaussianBuf) return;
    this.device.queue.writeBuffer(this._gaussianBuf, vertexOffset * 64, data);
  }

  setSplatScale(s) {
    this._uniforms[U_PARAMS] = s;
  }

  setGamma(g) {
    this._uniforms[U_PARAMS + 2] = g;
  }

  /** Upload a full SH coefficients buffer (non-streaming scenes).
   *  shData: Float32Array(count × numSHBases × 3), or null to clear SH.
   *  Rebuilds the bind group. */
  uploadSH(shData, numSHBases) {
    // No .destroy() on the outgoing buffer — see uploadGaussians for why.
    this._shBuf = null;
    this._shNumBases = numSHBases || 0;
    this._uniforms[U_SH_PARAMS + 1] = this._shNumBases;
    if (shData && this._shNumBases > 0) {
      this._shBuf = this._createBuffer(shData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(this._shBuf, 0, shData);
    }
    if (this._gaussianBuf) this._rebuildPreprocessBindGroup();
  }

  /** Pre-allocate the SH buffer for a streaming scene (zeros, no data yet).
   *  Follow up with patchSH() as chunks arrive. */
  allocateSH(count, numSHBases) {
    // No .destroy() on the outgoing buffer — see uploadGaussians for why.
    this._shBuf = null;
    this._shNumBases = numSHBases || 0;
    this._uniforms[U_SH_PARAMS + 1] = this._shNumBases;
    if (this._shNumBases > 0 && count > 0) {
      this._shBuf = this._createBuffer(count * this._shNumBases * 12, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    }
    if (this._gaussianBuf) this._rebuildPreprocessBindGroup();
  }

  /** Write a chunk of decoded SH coefficients into the SH buffer at gaussianOffset.
   *  shChunk: Float32Array(nVerts × numSHBases × 3). */
  patchSH(shChunk, gaussianOffset) {
    if (this._shBuf && shChunk) {
      this.device.queue.writeBuffer(this._shBuf, gaussianOffset * this._shNumBases * 12, shChunk);
    }
  }

  /** Update the active SH degree written into the per-frame uniform. */
  setShDegree(deg) {
    this._uniforms[U_SH_PARAMS] = deg;
  }

  setAaDilation(v) {
    this._uniforms[U_AA_DILATION] = v;
  }

  /** TEMP debug instrumentation — see shaders.js binding(7). Set the
   *  gaussian index to dump intermediate per-splat values for (-1 disables),
   *  then call readDebug() after a frame has rendered. */
  setDebugIndex(i) {
    this._uniforms[U_SH_PARAMS + 3] = i;
  }

  /** Reads back the debug buffer written by the shader for the splat index
   *  set via setDebugIndex(). Returns a plain object of named fields. */
  async readDebug() {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this._debugBuf, 0, this._debugReadBuf, 0, 16 * 4);
    this.device.queue.submit([encoder.finish()]);
    await this._debugReadBuf.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this._debugReadBuf.getMappedRange().slice(0));
    this._debugReadBuf.unmap();
    const [tz, a, b, c, det, lambda1, lambda2, extent, sizeFade, nearFade, aaFactor, finalAlpha, r, g, bl, worldX] = data;
    return { tz, a, b, c, det, lambda1, lambda2, extent, sizeFade, nearFade, aaFactor, finalAlpha, rgb: [r, g, bl], worldX };
  }

  setBackground(bg) {
    this.background = parseBackground(bg);
  }

  /** Runs cs_preprocess over the first `count` gaussians — must be called
   *  every frame before runGpuSort/the CPU sort path and before draw(), since
   *  it depends on this frame's view/proj/transforms. See shaders.js. */
  preprocess(count) {
    if (!count || !this._preprocessBindGroup) return;
    this._preprocessCountData[0] = count;
    this.device.queue.writeBuffer(this._preprocessParamsBuf, 0, this._preprocessCountData);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this._preprocessPipeline);
    pass.setBindGroup(0, this._preprocessBindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  // ── Draw ───────────────────────────────────────────────────────────────────

  draw(count = this._numSplats) {
    if (!count || !this.bindGroup) return;
    this._ensureHdrTexture();

    const encoder = this.device.createCommandEncoder();
    const splatPass = encoder.beginRenderPass({
      colorAttachments: [{
        view:       this._hdrView,
        clearValue: this.background,
        loadOp:     'clear',
        storeOp:    'store',
      }],
    });
    splatPass.setPipeline(this.pipeline);
    splatPass.setBindGroup(0, this.bindGroup);
    splatPass.draw(6, count, 0, 0); // 6 verts/instance, count instances
    splatPass.end();

    const blitPass = encoder.beginRenderPass({
      colorAttachments: [{
        view:    this.context.getCurrentTexture().createView(),
        loadOp:  'load', // fullscreen triangle overwrites every pixel — no clear needed
        storeOp: 'store',
      }],
    });
    blitPass.setPipeline(this._blitPipeline);
    blitPass.setBindGroup(0, this._blitBindGroup);
    blitPass.draw(3, 1, 0, 0); // fullscreen triangle
    blitPass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  /** (Re)creates the HDR intermediate texture + its blit bind group to match
   *  the canvas's current backing-buffer size. Cheap to call every frame —
   *  no-ops once sized. */
  _ensureHdrTexture() {
    const w = this.canvas.width, h = this.canvas.height;
    if (this._hdrTexture && this._hdrTexture.width === w && this._hdrTexture.height === h) return;
    this._hdrTexture?.destroy();
    this._hdrTexture = this.device.createTexture({
      size: [w, h],
      format: this._hdrFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this._hdrView = this._hdrTexture.createView();
    this._blitBindGroup = this.device.createBindGroup({
      layout: this._blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._hdrView },
        { binding: 1, resource: this._blitSampler },
      ],
    });
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────

  destroy() {
    this._hdrTexture?.destroy();
    this._debugBuf?.destroy();
    this._debugReadBuf?.destroy();
    this._uniformBuf?.destroy();
    this._gaussianBuf?.destroy();
    this._orderBuf?.destroy();
    this._transformBuf?.destroy();
    this._maskVolBuf?.destroy();
    this._partVolMaskBuf?.destroy();
    this._keysBufA?.destroy();
    this._keysBufB?.destroy();
    this._idxBufB?.destroy();
    this._histoBuf?.destroy();
    this._blockPfxBuf?.destroy();
    this._globalPfxBuf?.destroy();
    this._chunkTotalBuf?.destroy();
    this._chunkPfxBuf?.destroy();
    this._sortParamsBuf?.destroy();
    this._sortParamsStaging?.destroy();
    this._shBuf?.destroy();
    this._shDummyBuf?.destroy();
    this._splatGeomBuf?.destroy();
    this._preprocessParamsBuf?.destroy();
    this.context?.unconfigure();
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  _createPipeline() {
    // Shared with _createPreprocessPipeline() — one compiled module for the
    // whole SHADER source (vs_main/fs_main/cs_preprocess).
    this._mainModule = this._mainModule ?? this.device.createShaderModule({ code: SHADER });
    const module = this._mainModule;

    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module, entryPoint: 'vs_main' },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{
          // Splats blend into the HDR intermediate target, not the canvas
          // directly — see BLIT_SHADER.
          format: this._hdrFormat,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      // No depth/stencil — Gaussians are manually sorted back-to-front
    });
  }

  _createBlitPipeline() {
    const module = this.device.createShaderModule({ code: BLIT_SHADER });
    this._blitPipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: this._format }] },
      primitive: { topology: 'triangle-list' },
    });
    this._blitSampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  }

  _createBuffer(size, usage) {
    return this.device.createBuffer({ size, usage });
  }

  /** Creates the GPU radix sort compute pipelines (see src/sort-shaders.js).
   *  Each entry point only references the bindings it needs, so layout:'auto'
   *  derives a distinct bind group layout per pipeline — see _rebuildSortBindGroups. */
  /** Creates the cs_preprocess compute pipeline (see shaders.js) — shares the
   *  SHADER module with vs_main/fs_main so it can reuse the Gaussian/Uniforms
   *  structs and evalSH/quatToMat3 helpers without duplicating them. */
  _createPreprocessPipeline() {
    this._mainModule = this._mainModule ?? this.device.createShaderModule({ code: SHADER });
    const module = this._mainModule;
    this._preprocessPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'cs_preprocess' },
    });
  }

  _createSortPipelines() {
    const module = this.device.createShaderModule({ code: SORT_SHADER });
    const make = (entryPoint) => this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint },
    });
    this._depthKeyPipeline    = make('cs_depth_key');
    this._histogramPipeline   = make('cs_histogram');
    this._scanReducePipeline  = make('cs_scan_reduce');
    this._scanGlobalPipeline  = make('cs_scan_global');
    this._scanCombinePipeline = make('cs_scan_combine');
    this._scatterPipeline     = make('cs_scatter');
  }

  /** vs_main/fs_main now only touch uniforms, order, and the precomputed
   *  splatGeom buffer — all the per-gaussian source data (gaussians,
   *  transforms, masks, SH, debug) moved into cs_preprocess. See
   *  _rebuildPreprocessBindGroup for that pipeline's (much larger) bind group. */
  _rebuildBindGroup() {
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._uniformBuf  } },
        { binding: 2, resource: { buffer: this._orderBuf    } },
        { binding: 9, resource: { buffer: this._splatGeomBuf } },
      ],
    });
  }

  /** Rebuilds cs_preprocess's bind group. Called after uploadGaussians, and
   *  whenever transforms/masks/SH change (those don't affect vs_main/fs_main
   *  anymore — see _rebuildBindGroup — only the preprocess compute pass). */
  _rebuildPreprocessBindGroup() {
    this._preprocessBindGroup = this.device.createBindGroup({
      layout: this._preprocessPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0,  resource: { buffer: this._uniformBuf                } },
        { binding: 1,  resource: { buffer: this._gaussianBuf               } },
        { binding: 3,  resource: { buffer: this._transformBuf              } },
        { binding: 4,  resource: { buffer: this._maskVolBuf                } },
        { binding: 5,  resource: { buffer: this._partVolMaskBuf            } },
        { binding: 6,  resource: { buffer: this._shBuf ?? this._shDummyBuf } },
        { binding: 7,  resource: { buffer: this._debugBuf                  } },
        { binding: 8,  resource: { buffer: this._splatGeomBuf              } },
        { binding: 10, resource: { buffer: this._preprocessParamsBuf       } },
      ],
    });
  }

  /** Rebuilds the GPU sort bind groups against the current per-frame
   *  buffers (gaussians, transforms, order/idxA, keys/idx/histo/prefix scratch).
   *  Called after uploadGaussians, since buffer sizes depend on splat count.
   *  Each bind group is built from its own pipeline's derived layout — only
   *  the bindings that pipeline's entry point actually references. */
  _rebuildSortBindGroups() {
    const entries = {
      0:  { buffer: this._uniformBuf },
      1:  { buffer: this._gaussianBuf },
      2:  { buffer: this._transformBuf },
      3:  { buffer: this._sortParamsBuf },
      4:  { buffer: this._keysBufA },
      5:  { buffer: this._keysBufB },
      6:  { buffer: this._orderBuf }, // idxA
      7:  { buffer: this._idxBufB },
      8:  { buffer: this._histoBuf },
      9:  { buffer: this._blockPfxBuf },
      10: { buffer: this._globalPfxBuf },
      11: { buffer: this._chunkTotalBuf },
      12: { buffer: this._chunkPfxBuf },
    };
    const buildGroup = (pipeline, bindings) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map(binding => ({ binding, resource: entries[binding] })),
    });
    this._depthKeyBindGroup    = buildGroup(this._depthKeyPipeline,    [0, 1, 2, 3, 4, 6]);
    this._histogramBindGroup   = buildGroup(this._histogramPipeline,   [3, 4, 5, 8]);
    this._scanReduceBindGroup  = buildGroup(this._scanReducePipeline,  [3, 8, 9, 11]);
    this._scanGlobalBindGroup  = buildGroup(this._scanGlobalPipeline,  [3, 10, 11, 12]);
    this._scanCombineBindGroup = buildGroup(this._scanCombinePipeline, [9, 12]);
    this._scatterBindGroup     = buildGroup(this._scatterPipeline,     [3, 4, 5, 6, 7, 9, 10]);
  }

  /** Runs the GPU radix sort over the first `count` gaussians (4 passes of
   *  depth_key -> histogram -> scan_reduce -> scan_global -> scan_combine ->
   *  scatter). Result permutation lands back in _orderBuf (idxA) — ready for
   *  vs_main's next draw, no CPU readback needed. See src/sort-shaders.js.
   *
   *  Submitted as 5 separate queue.submit() calls (depth_key, then one per
   *  radix pass) rather than one big command buffer — mobile driver TDR
   *  watchdogs key off a single submission's GPU time, and bundling all ~13
   *  array-sized dispatches into one submission risks tripping it on large
   *  scenes. Each per-pass submission now carries at most ~3 array-sized
   *  dispatches (histogram, scan_combine, scatter). */
  runGpuSort(count) {
    if (!count) return;
    const nwg = numWorkgroups(count);
    const nch = numChunks(count);

    // Fill all 5 SortParams slots up front and upload in one go. Per-pass
    // values are applied inside each submission's encoder via
    // copyBufferToBuffer — a queue.writeBuffer() per pass would all land
    // before any dispatch in these command buffers runs, since encoded
    // commands only execute at submit().
    const p = this._sortParamsData;
    for (let slot = 0; slot < 5; slot++) {
      const o = slot * 4;
      p[o]     = count;
      p[o + 1] = slot === 0 ? 0 : slot - 1; // slot 0 = cs_depth_key (passIndex unused)
      p[o + 2] = nwg;
      p[o + 3] = nch;
    }
    this.device.queue.writeBuffer(this._sortParamsStaging, 0, p);

    {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this._sortParamsStaging, 0, this._sortParamsBuf, 0, 16);
      const pass = encoder.beginComputePass();
      pass.setPipeline(this._depthKeyPipeline);
      pass.setBindGroup(0, this._depthKeyBindGroup);
      pass.dispatchWorkgroups(nwg);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    }

    for (let i = 0; i < 4; i++) {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this._sortParamsStaging, (i + 1) * 16, this._sortParamsBuf, 0, 16);

      const histPass = encoder.beginComputePass();
      histPass.setPipeline(this._histogramPipeline);
      histPass.setBindGroup(0, this._histogramBindGroup);
      histPass.dispatchWorkgroups(nwg);
      histPass.end();

      const scanReducePass = encoder.beginComputePass();
      scanReducePass.setPipeline(this._scanReducePipeline);
      scanReducePass.setBindGroup(0, this._scanReduceBindGroup);
      scanReducePass.dispatchWorkgroups(nch);
      scanReducePass.end();

      const scanGlobalPass = encoder.beginComputePass();
      scanGlobalPass.setPipeline(this._scanGlobalPipeline);
      scanGlobalPass.setBindGroup(0, this._scanGlobalBindGroup);
      scanGlobalPass.dispatchWorkgroups(1);
      scanGlobalPass.end();

      const scanCombinePass = encoder.beginComputePass();
      scanCombinePass.setPipeline(this._scanCombinePipeline);
      scanCombinePass.setBindGroup(0, this._scanCombineBindGroup);
      scanCombinePass.dispatchWorkgroups(nwg);
      scanCombinePass.end();

      const scatterPass = encoder.beginComputePass();
      scatterPass.setPipeline(this._scatterPipeline);
      scatterPass.setBindGroup(0, this._scatterBindGroup);
      scatterPass.dispatchWorkgroups(nwg);
      scatterPass.end();

      this.device.queue.submit([encoder.finish()]);
    }
  }

  /** Debug/test helper: reads back _orderBuf (idxA) as a Uint32Array(count).
   *  Not used in the render hot path — see examples/gpu-sort-test.html. */
  async debugReadOrder(count) {
    const byteSize = count * 4;
    const staging = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this._orderBuf, 0, staging, 0, byteSize);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();
    return result;
  }
}

// ── 4×4 matrix inverse (column-major Float32Array, gl-matrix convention) ──────

export function invertMat4(a) {
  const a00=a[0], a01=a[1], a02=a[2], a03=a[3];
  const a10=a[4], a11=a[5], a12=a[6], a13=a[7];
  const a20=a[8], a21=a[9], a22=a[10],a23=a[11];
  const a30=a[12],a31=a[13],a32=a[14],a33=a[15];

  const b00=a00*a11-a01*a10, b01=a00*a12-a02*a10;
  const b02=a00*a13-a03*a10, b03=a01*a12-a02*a11;
  const b04=a01*a13-a03*a11, b05=a02*a13-a03*a12;
  const b06=a20*a31-a21*a30, b07=a20*a32-a22*a30;
  const b08=a20*a33-a23*a30, b09=a21*a32-a22*a31;
  const b10=a21*a33-a23*a31, b11=a22*a33-a23*a32;

  const det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
  if (!det) return null;
  const d = 1.0 / det;
  const out = new Float32Array(16);
  out[0]  = (a11*b11 - a12*b10 + a13*b09) * d;
  out[1]  = (a02*b10 - a01*b11 - a03*b09) * d;
  out[2]  = (a31*b05 - a32*b04 + a33*b03) * d;
  out[3]  = (a22*b04 - a21*b05 - a23*b03) * d;
  out[4]  = (a12*b08 - a10*b11 - a13*b07) * d;
  out[5]  = (a00*b11 - a02*b08 + a03*b07) * d;
  out[6]  = (a32*b02 - a30*b05 - a33*b01) * d;
  out[7]  = (a20*b05 - a22*b02 + a23*b01) * d;
  out[8]  = (a10*b10 - a11*b08 + a13*b06) * d;
  out[9]  = (a01*b08 - a00*b10 - a03*b06) * d;
  out[10] = (a30*b04 - a31*b02 + a33*b00) * d;
  out[11] = (a21*b02 - a20*b04 - a23*b00) * d;
  out[12] = (a11*b07 - a10*b09 - a12*b06) * d;
  out[13] = (a00*b09 - a01*b07 + a02*b06) * d;
  out[14] = (a31*b01 - a30*b03 - a32*b00) * d;
  out[15] = (a20*b03 - a21*b01 + a22*b00) * d;
  return out;
}

// ── Background colour parsing ──────────────────────────────────────────────

function parseBackground(bg) {
  if (!bg || bg === 'transparent') return { r: 0, g: 0, b: 0, a: 0 };
  if (Array.isArray(bg))           return { r: bg[0], g: bg[1], b: bg[2], a: bg[3] ?? 1 };
  if (typeof bg === 'string') {
    const hex = bg.replace('#', '');
    if (hex.length === 6) {
      return {
        r: parseInt(hex.slice(0, 2), 16) / 255,
        g: parseInt(hex.slice(2, 4), 16) / 255,
        b: parseInt(hex.slice(4, 6), 16) / 255,
        a: 1,
      };
    }
    if (hex.length === 8) {
      return {
        r: parseInt(hex.slice(0, 2), 16) / 255,
        g: parseInt(hex.slice(2, 4), 16) / 255,
        b: parseInt(hex.slice(4, 6), 16) / 255,
        a: parseInt(hex.slice(6, 8), 16) / 255,
      };
    }
  }
  return { r: 0, g: 0, b: 0, a: 1 };
}
