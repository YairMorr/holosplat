/**
 * Loader for 3D Gaussian Splatting .ply files (standard 3DGS training output).
 *
 * Supports two loading modes:
 *   loadPly(url, onProgress)      — full download, then decode (all formats)
 *   openPlyStream(url, onProgress) — streams header immediately, returns a reader
 *                                    for iterative chunk decoding via decodePlyVertices
 *
 * Transforms applied:
 *   f_dc_0/1/2 → RGB:  0.5 + SH_C0 * f_dc  (SH_C0 = 0.28209479177387814)
 *   opacity    → sigmoid: 1 / (1 + exp(−x))
 *   scale_i    → exp(scale_i)
 *   rot_0..3   → quaternion (w,x,y,z) → (x,y,z,w), normalised
 *
 * Output: canonical Float32Array layout, 16 floats per Gaussian.
 */

import { fetchWithProgress } from './fetch-utils.js';

const SH_C0 = 0.28209479177387814;

// ── Public API ─────────────────────────────────────────────────────────────

export async function loadPly(url, onProgress) {
  const buffer = await fetchWithProgress(url, onProgress);
  return parsePly(buffer);
}

export function parsePly(buffer) {
  const bytes = new Uint8Array(buffer);
  const eoh   = findEndHeader(bytes);
  if (eoh < 0) throw new Error('Invalid PLY: end_header not found');
  const { numVertices, propMap, stride, hasColor } = parseHeaderText(
    new TextDecoder().decode(bytes.subarray(0, eoh))
  );
  if (numVertices === 0) throw new Error('PLY file contains no vertices');
  const src = new DataView(buffer, eoh);
  return { data: decodePlyVertices(src, propMap, stride, hasColor, numVertices), count: numVertices };
}

/**
 * Open a PLY stream: reads the header bytes (typically < 1 KB), then
 * returns `{ numVertices, consume }`.
 *
 * `consume(onChunk, onProgress?)` streams and decodes the vertex data,
 * calling `onChunk(Float32Array, vertexCount)` for each decoded batch.
 * It always processes bytes already buffered during header reading before
 * pulling more from the network, so the caller never misses data even if
 * the entire file arrived in the first network read.
 *
 * @returns {{ numVertices: number,
 *             consume: (onChunk, onProgress?) => Promise<void> }}
 */
export async function openPlyStream(url, onHeaderProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);

  const contentLength = parseInt(res.headers.get('content-length') || '0', 10);
  const reader = res.body.getReader();
  let buf    = new Uint8Array(0);
  let loaded = 0;

  // Read until end_header is found.
  for (;;) {
    const { done, value } = await reader.read();
    if (done) throw new Error('PLY stream ended before end_header');

    loaded += value.byteLength;
    if (onHeaderProgress && contentLength > 0) onHeaderProgress(loaded / contentLength);

    const next = new Uint8Array(buf.length + value.length);
    next.set(buf); next.set(value, buf.length);
    buf = next;

    const eoh = findEndHeader(buf);
    if (eoh >= 0) {
      const info    = parseHeaderText(new TextDecoder().decode(buf.subarray(0, eoh)));
      const pending = buf.slice(eoh);  // vertex bytes already in memory
      const { numVertices, propMap, stride, hasColor } = info;

      const consume = async (onChunk, onProgress) => {
        let p = pending;

        for (;;) {
          const { done: d, value: v } = await reader.read();

          if (!d) {
            loaded += v.byteLength;
            if (onProgress && contentLength > 0) onProgress(loaded / contentLength);
            const next2 = new Uint8Array(p.length + v.length);
            next2.set(p); next2.set(v, p.length);
            p = next2;
          }

          // Decode all complete vertices now in p (including any leftover pending)
          const nVerts = Math.floor(p.length / stride);
          if (nVerts > 0) {
            const usedBytes = nVerts * stride;
            onChunk(
              decodePlyVertices(
                new DataView(p.buffer, p.byteOffset, usedBytes),
                propMap, stride, hasColor, nVerts
              ),
              nVerts
            );
            p = p.slice(usedBytes);
          }

          if (d) break;
        }
      };

      return { numVertices, consume };
    }
    if (buf.length > 65536) throw new Error('PLY header exceeds 64 KB');
  }
}

/**
 * Decode `count` Gaussian splats from a DataView positioned at the start
 * of vertex data. Used by both parsePly (full load) and streaming.
 *
 * @param {DataView}  src
 * @param {object}    propMap    { name: { offset, type } }
 * @param {number}    stride     bytes per vertex
 * @param {boolean}   hasColor
 * @param {number}    count
 * @returns {Float32Array}  count × 16 floats
 */
export function decodePlyVertices(src, propMap, stride, hasColor, count) {
  const data = new Float32Array(count * 16);
  for (let i = 0; i < count; i++) {
    const base = i * stride;
    const d    = i * 16;

    data[d + 0] = readProp(src, base, propMap['x']);
    data[d + 1] = readProp(src, base, propMap['y']);
    data[d + 2] = readProp(src, base, propMap['z']);

    if (hasColor) {
      data[d + 4] = clamp01(0.5 + SH_C0 * readProp(src, base, propMap['f_dc_0']));
      data[d + 5] = clamp01(0.5 + SH_C0 * readProp(src, base, propMap['f_dc_1']));
      data[d + 6] = clamp01(0.5 + SH_C0 * readProp(src, base, propMap['f_dc_2']));
    } else {
      data[d + 4] = 1; data[d + 5] = 1; data[d + 6] = 1;
    }

    data[d + 7]  = sigmoid(readProp(src, base, propMap['opacity']));
    data[d + 8]  = Math.exp(readProp(src, base, propMap['scale_0']));
    data[d + 9]  = Math.exp(readProp(src, base, propMap['scale_1']));
    data[d + 10] = Math.exp(readProp(src, base, propMap['scale_2']));

    const rw = readProp(src, base, propMap['rot_0']);
    const rx = readProp(src, base, propMap['rot_1']);
    const ry = readProp(src, base, propMap['rot_2']);
    const rz = readProp(src, base, propMap['rot_3']);
    const rl = Math.hypot(rx, ry, rz, rw) || 1;
    data[d + 12] = rx / rl;
    data[d + 13] = ry / rl;
    data[d + 14] = rz / rl;
    data[d + 15] = rw / rl;
  }
  return data;
}

// ── Header parsing ─────────────────────────────────────────────────────────

/** Return byte offset of first byte after "end_header\n", or -1 if not found. */
function findEndHeader(bytes) {
  // "end_header" as bytes [101,110,100,95,104,101,97,100,101,114]
  const tag = [101, 110, 100, 95, 104, 101, 97, 100, 101, 114];
  outer: for (let i = 0; i <= bytes.length - tag.length; i++) {
    for (let j = 0; j < tag.length; j++) {
      if (bytes[i + j] !== tag[j]) continue outer;
    }
    let end = i + tag.length;
    if (bytes[end] === 13) end++; // \r
    if (bytes[end] === 10) end++; // \n
    return end;
  }
  return -1;
}

function parseHeaderText(text) {
  const lines = text.split('\n');
  let numVertices = 0;
  let inVertex    = false;
  const properties = [];

  for (const raw of lines) {
    const parts = raw.trim().split(/\s+/);
    if (parts[0] === 'element') {
      inVertex = parts[1] === 'vertex';
      if (inVertex) numVertices = parseInt(parts[2], 10);
    } else if (parts[0] === 'property' && inVertex) {
      properties.push({ type: parts[1], name: parts[2] });
    }
  }

  const propMap = {};
  let stride = 0;
  for (const p of properties) {
    propMap[p.name] = { offset: stride, type: p.type };
    stride += sizeOf(p.type);
  }

  const required = ['x','y','z','scale_0','scale_1','scale_2',
                    'rot_0','rot_1','rot_2','rot_3','opacity'];
  for (const r of required) {
    if (!propMap[r]) throw new Error(`PLY missing required property: ${r}`);
  }

  return {
    numVertices,
    propMap,
    stride,
    hasColor: !!(propMap['f_dc_0'] && propMap['f_dc_1'] && propMap['f_dc_2']),
  };
}

// ── Binary read helpers ────────────────────────────────────────────────────

function sizeOf(type) {
  switch (type) {
    case 'float': case 'float32': case 'int':   case 'uint':   return 4;
    case 'double': case 'int64': case 'uint64': return 8;
    case 'short': case 'ushort': case 'int16': case 'uint16': return 2;
    case 'char':  case 'uchar':  case 'int8':  case 'uint8':  return 1;
    default: return 4;
  }
}

function readProp(view, base, prop) {
  const off = base + prop.offset;
  switch (prop.type) {
    case 'float': case 'float32': return view.getFloat32(off, true);
    case 'double':                return view.getFloat64(off, true);
    case 'int': case 'int32':     return view.getInt32(off, true);
    case 'uint': case 'uint32':   return view.getUint32(off, true);
    case 'short': case 'int16':   return view.getInt16(off, true);
    case 'ushort': case 'uint16': return view.getUint16(off, true);
    case 'char': case 'int8':     return view.getInt8(off);
    case 'uchar': case 'uint8':   return view.getUint8(off);
    default:                      return view.getFloat32(off, true);
  }
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
