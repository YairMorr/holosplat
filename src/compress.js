/**
 * HoloSplat compressor — encodes canonical Gaussian data to the .spz format
 * and gzip-compresses it for storage or transfer.
 *
 * The .spz format (Niantic Labs, v3) stores each Gaussian as:
 *   Position  9 bytes  (int24 × 3, fixed-point with auto-selected precision)
 *   Alpha     1 byte   (uint8)
 *   Color     3 bytes  (uint8 RGB)
 *   Scale     3 bytes  (uint8 log-space quantized)
 *   Rotation  4 bytes  (smallest-3 quaternion encoding)
 *   ─────────────────
 *   Total    20 bytes/Gaussian (before gzip) vs 32 bytes for .splat
 *
 * Input: canonical Float32Array (16 floats/Gaussian):
 *   [0-2]  pos.xyz
 *   [3]    pad
 *   [4-7]  color.rgba (0-1)
 *   [8-10] scale.xyz (linear)
 *   [11]   pad
 *   [12-15] quat.xyzw
 *
 * Usage:
 *   import { compressToSpz } from 'holosplat';
 *   const buffer = await compressToSpz(data, count);
 *   const blob = new Blob([buffer], { type: 'application/octet-stream' });
 */

const SPZ_MAGIC   = 0x5053474E;
const SPZ_VERSION = 3;

/**
 * Encode canonical Gaussian data and gzip-compress to a .spz ArrayBuffer.
 *
 * @param {Float32Array} data   – 16 floats/Gaussian, canonical layout
 * @param {number}       count  – number of Gaussians
 * @param {object}       [opts]
 * @param {number}       [opts.fractionalBits]  – position fixed-point precision
 *                                                (auto-detected from data range if omitted)
 * @returns {Promise<ArrayBuffer>}
 */
export async function compressToSpz(data, count, opts = {}) {
  const raw = encodeSpz(data, count, opts);
  return compressGzip(raw);
}

/**
 * Encode canonical Gaussian data to a raw (uncompressed) .spz Uint8Array.
 * Useful if you want to compress with a different algorithm or inspect the bytes.
 *
 * @param {Float32Array} data
 * @param {number}       count
 * @param {object}       [opts]
 * @param {number}       [opts.fractionalBits]
 * @returns {Uint8Array}
 */
export function encodeSpz(data, count, opts = {}) {
  // ── Choose position fixed-point precision ─────────────────────────────────
  // int24 range: -8,388,608 … 8,388,607  (2^23 - 1)
  // max representable position = INT24_MAX / 2^fractionalBits
  let { fractionalBits } = opts;
  if (fractionalBits == null) {
    let maxAbsPos = 0;
    for (let i = 0; i < count; i++) {
      const b = i * 16;
      if (Math.abs(data[b])     > maxAbsPos) maxAbsPos = Math.abs(data[b]);
      if (Math.abs(data[b + 1]) > maxAbsPos) maxAbsPos = Math.abs(data[b + 1]);
      if (Math.abs(data[b + 2]) > maxAbsPos) maxAbsPos = Math.abs(data[b + 2]);
    }
    const INT24_MAX = (1 << 23) - 1;
    fractionalBits = maxAbsPos > 0
      ? Math.min(20, Math.max(0, Math.floor(Math.log2(INT24_MAX / maxAbsPos))))
      : 12;
  }

  // ── Allocate buffer ───────────────────────────────────────────────────────
  const HEADER = 16;
  const total  = HEADER + count * (9 + 1 + 3 + 3 + 4);
  const buf    = new ArrayBuffer(total);
  const view   = new DataView(buf);
  const u8     = new Uint8Array(buf);

  // ── Header ────────────────────────────────────────────────────────────────
  view.setUint32( 0, SPZ_MAGIC,       true);
  view.setUint32( 4, SPZ_VERSION,     true);
  view.setUint32( 8, count,           true);
  view.setUint8 (12, 0);                    // shDegree
  view.setUint8 (13, fractionalBits);
  view.setUint8 (14, 0);                    // flags
  view.setUint8 (15, 0);                    // reserved

  // ── Section offsets (column-organised, matching the SPZ spec) ────────────
  const offPos   = HEADER;
  const offAlpha = offPos   + count * 9;
  const offColor = offAlpha + count * 1;
  const offScale = offColor + count * 3;
  const offRot   = offScale + count * 3;

  const posScale = 1 << fractionalBits;

  for (let i = 0; i < count; i++) {
    const d = i * 16;

    // Position (float → fixed-point int24 LE)
    writeInt24(view, offPos + i * 9 + 0, data[d + 0] * posScale);
    writeInt24(view, offPos + i * 9 + 3, data[d + 1] * posScale);
    writeInt24(view, offPos + i * 9 + 6, data[d + 2] * posScale);

    // Alpha
    u8[offAlpha + i] = clampU8(data[d + 7] * 255);

    // Color RGB
    u8[offColor + i * 3 + 0] = clampU8(data[d + 4] * 255);
    u8[offColor + i * 3 + 1] = clampU8(data[d + 5] * 255);
    u8[offColor + i * 3 + 2] = clampU8(data[d + 6] * 255);

    // Scale: linear → log-space uint8
    // Inverse of: Math.exp((b - 128) / 16)  →  b = log(scale) * 16 + 128
    u8[offScale + i * 3 + 0] = encodeScale(data[d +  8]);
    u8[offScale + i * 3 + 1] = encodeScale(data[d +  9]);
    u8[offScale + i * 3 + 2] = encodeScale(data[d + 10]);

    // Rotation: smallest-3 encoding (4 bytes, uint32 LE)
    view.setUint32(offRot + i * 4,
      encodeQuat(data[d + 12], data[d + 13], data[d + 14], data[d + 15]),
      true);
  }

  return u8;
}

// ── Encoding helpers ──────────────────────────────────────────────────────────

function writeInt24(view, offset, value) {
  const v = Math.max(-8388608, Math.min(8388607, Math.round(value)));
  view.setUint8(offset,     v & 0xFF);
  view.setUint8(offset + 1, (v >>  8) & 0xFF);
  view.setUint8(offset + 2, (v >> 16) & 0xFF);
}

function clampU8(v) {
  return Math.max(0, Math.min(255, Math.round(v)));
}

function encodeScale(linear) {
  return clampU8(Math.log(Math.max(1e-9, linear)) * 16 + 128);
}

/**
 * Encode a unit quaternion (x,y,z,w) using smallest-3 / "omit largest" encoding.
 *
 * The largest component is stored implicitly as sqrt(1 - a² - b² - c²).
 * Canonically it must be positive, so we flip the sign of all components when
 * the largest is negative (the rotation is identical either way).
 *
 * Packed uint32:
 *   bits [ 1: 0]  index of largest component (0=x,1=y,2=z,3=w)
 *   bits [11: 2]  a  (int10, scaled by 512*√2 to fill [-1/√2, 1/√2])
 *   bits [21:12]  b
 *   bits [31:22]  c
 */
function encodeQuat(qx, qy, qz, qw) {
  // Normalize
  const len = Math.hypot(qx, qy, qz, qw) || 1;
  const q = [qx / len, qy / len, qz / len, qw / len];

  // Find largest-magnitude component
  let maxIdx = 0;
  for (let j = 1; j < 4; j++) {
    if (Math.abs(q[j]) > Math.abs(q[maxIdx])) maxIdx = j;
  }

  // Ensure largest is positive (flip sign of whole quaternion if needed)
  const sign = q[maxIdx] < 0 ? -1 : 1;

  // The three "other" components, in the order the decoder expects:
  //   maxIdx=0 → [qy,qz,qw]  idx [1,2,3]
  //   maxIdx=1 → [qx,qz,qw]  idx [0,2,3]
  //   maxIdx=2 → [qx,qy,qw]  idx [0,1,3]
  //   maxIdx=3 → [qx,qy,qz]  idx [0,1,2]
  // [0,1,2,3].filter(j => j !== maxIdx) produces exactly these in ascending order.
  const others = [0, 1, 2, 3].filter(j => j !== maxIdx);

  // Scale factor: decoder uses s = (1/√2) / 512,  encoder uses 1/s = 512*√2
  const S = 512 * Math.SQRT2;
  const encode10 = j => Math.max(-512, Math.min(511, Math.round(q[j] * sign * S)));

  const a = encode10(others[0]);
  const b = encode10(others[1]);
  const c = encode10(others[2]);

  // Pack and return as unsigned 32-bit (>>> 0 ensures correct bit pattern)
  return ((maxIdx & 3) | ((a & 0x3FF) << 2) | ((b & 0x3FF) << 12) | ((c & 0x3FF) << 22)) >>> 0;
}

// ── Gzip compression ──────────────────────────────────────────────────────────

async function compressGzip(data) {
  if (typeof CompressionStream === 'undefined') {
    throw new Error('CompressionStream API is not available in this environment');
  }
  const stream = new CompressionStream('gzip');
  const writer = stream.writable.getWriter();
  writer.write(data);
  writer.close();
  return new Response(stream.readable).arrayBuffer();
}
