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

// Uniform buffer offsets (in float32 index units)
const U_VIEW     = 0;   // mat4 (16 floats)
const U_PROJ     = 16;  // mat4 (16 floats)
const U_VIEWPORT = 32;  // vec2 (2 floats)
const U_FOCAL    = 34;  // vec2 (2 floats)
const U_PARAMS   = 36;  // vec4 (4 floats), .x = splatScale
const U_SIZE     = 40;  // total floats → 160 bytes

export class Renderer {
  constructor(canvas, background) {
    this.canvas     = canvas;
    this.background = parseBackground(background);
    this.device     = null;
    this.context    = null;
    this.pipeline   = null;
    this.bindGroup  = null;

    // GPU buffers
    this._uniformBuf  = null;
    this._gaussianBuf = null;
    this._orderBuf    = null;

    // CPU-side uniform data
    this._uniforms = new Float32Array(U_SIZE);
    this._uniforms[U_PARAMS] = 1.0; // splatScale = 1

    this._numSplats = 0;
  }

  // ── Initialise WebGPU ──────────────────────────────────────────────────────

  async init() {
    if (!navigator.gpu) throw new Error('WebGPU is not supported in this browser.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter found.');
    this.device = await adapter.requestDevice();

    this.context = this.canvas.getContext('webgpu');
    this._format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this._format,
      alphaMode: 'premultiplied',
    });

    this._createPipeline();
    this._uniformBuf = this._createBuffer(U_SIZE * 4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  }

  // ── Upload scene data (call once after load) ───────────────────────────────

  uploadGaussians(data, count) {
    this._numSplats = count;

    // Gaussian storage buffer (read-once upload)
    if (this._gaussianBuf) this._gaussianBuf.destroy();
    this._gaussianBuf = this.device.createBuffer({
      size:  data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this._gaussianBuf, 0, data);

    // Order buffer (rewritten every frame)
    if (this._orderBuf) this._orderBuf.destroy();
    this._orderBuf = this.device.createBuffer({
      size:  count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this._rebuildBindGroup();
  }

  // ── Per-frame updates ──────────────────────────────────────────────────────

  updateUniforms({ view, proj, width, height, focal }) {
    const u = this._uniforms;
    u.set(view,   U_VIEW);
    u.set(proj,   U_PROJ);
    u[U_VIEWPORT]     = width;
    u[U_VIEWPORT + 1] = height;
    u[U_FOCAL]        = focal;
    u[U_FOCAL + 1]    = focal;
    this.device.queue.writeBuffer(this._uniformBuf, 0, u);
  }

  updateOrder(sortedIndices, count) {
    this.device.queue.writeBuffer(
      this._orderBuf, 0,
      sortedIndices.buffer, 0, count * 4
    );
  }

  setSplatScale(s) {
    this._uniforms[U_PARAMS] = s;
  }

  setBackground(bg) {
    this.background = parseBackground(bg);
  }

  // ── Draw ───────────────────────────────────────────────────────────────────

  draw() {
    if (!this._numSplats || !this.bindGroup) return;

    const encoder = this.device.createCommandEncoder();
    const pass    = encoder.beginRenderPass({
      colorAttachments: [{
        view:       this.context.getCurrentTexture().createView(),
        clearValue: this.background,
        loadOp:     'clear',
        storeOp:    'store',
      }],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(6, this._numSplats, 0, 0); // 6 verts/instance, N instances
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────

  destroy() {
    this._uniformBuf?.destroy();
    this._gaussianBuf?.destroy();
    this._orderBuf?.destroy();
    this.context?.unconfigure();
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  _createPipeline() {
    const module = this.device.createShaderModule({ code: SHADER });

    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module, entryPoint: 'vs_main' },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{
          format: this._format,
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

  _createBuffer(size, usage) {
    return this.device.createBuffer({ size, usage });
  }

  _rebuildBindGroup() {
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._uniformBuf  } },
        { binding: 1, resource: { buffer: this._gaussianBuf } },
        { binding: 2, resource: { buffer: this._orderBuf    } },
      ],
    });
  }
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
