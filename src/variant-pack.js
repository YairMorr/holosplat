/**
 * HoloSplat variant packer — merges N color/material variants of the same
 * geometry into a single .spzv file: geometry (position/scale/rotation) is
 * stored once, plus one small color+alpha palette per variant. The viewer
 * can swap the active palette at runtime (Viewer#setVariant) with no reload
 * and without rendering every variant simultaneously.
 *
 * .spzv layout (gzip-compressed), reusing the .spz per-Gaussian encodings
 * (see compress.js):
 *
 *   Header (16 bytes, little-endian):
 *     [0-3]  magic        uint32  = 0x56505348 ("HSPV")
 *     [4-7]  version      uint32  = 1
 *     [8-11] numPoints    uint32
 *     [12]   numVariants  uint8
 *     [13]   fractionalBits uint8 (position precision, as in .spz)
 *     [14]   flags        uint8  (reserved)
 *     [15]   reserved     uint8
 *
 *   Variant names: numVariants × 16 bytes (UTF-8, NUL-padded)
 *
 *   Geometry (shared, numPoints entries):
 *     Position  numPoints × 9 bytes (int24 × 3, ÷ 2^fractionalBits)
 *     Scale     numPoints × 3 bytes (log-space uint8)
 *     Rotation  numPoints × 4 bytes (smallest-3 encoding)
 *
 *   Palettes (one block per variant, in declaration order, numPoints entries):
 *     Alpha     numPoints × 1 byte
 *     Color     numPoints × 3 bytes (uint8 RGB)
 *
 * Usage:
 *   import { compressVariantsToSpzv } from 'holosplat';
 *   const buffer = await compressVariantsToSpzv([
 *     { name: 'blue',   data: blueData   },
 *     { name: 'yellow', data: yellowData },
 *   ], count);
 */

import { writeInt24, clampU8, encodeScale, encodeQuat, compressGzip } from './compress.js';

export const SPZV_MAGIC   = 0x56505348; // "HSPV"
export const SPZV_VERSION = 1;
export const SPZV_NAME_BYTES = 16;

/**
 * Encode N color/material variants of the same geometry into a raw
 * (uncompressed) .spzv Uint8Array. Geometry (position/scale/rotation) is
 * taken from `variants[0].data`; every variant contributes its own
 * color+alpha palette.
 *
 * @param {{name: string, data: Float32Array}[]} variants  – each `data` is
 *   16 floats/Gaussian canonical layout, `count` Gaussians, same geometry
 * @param {number} count
 * @param {object} [opts]
 * @param {number} [opts.fractionalBits]  – position fixed-point precision
 * @returns {Uint8Array}
 */
export function encodeSpzv(variants, count, opts = {}) {
  if (!variants.length) throw new Error('HoloSplat: encodeSpzv requires at least one variant');
  if (variants.length > 255) throw new Error('HoloSplat: encodeSpzv supports at most 255 variants');

  const geom = variants[0].data;

  let { fractionalBits } = opts;
  if (fractionalBits == null) {
    let maxAbsPos = 0;
    for (let i = 0; i < count; i++) {
      const b = i * 16;
      maxAbsPos = Math.max(maxAbsPos, Math.abs(geom[b]), Math.abs(geom[b + 1]), Math.abs(geom[b + 2]));
    }
    const INT24_MAX = (1 << 23) - 1;
    fractionalBits = maxAbsPos > 0
      ? Math.min(20, Math.max(0, Math.floor(Math.log2(INT24_MAX / maxAbsPos))))
      : 12;
  }

  const numVariants = variants.length;
  const HEADER  = 16;
  const NAMES   = numVariants * SPZV_NAME_BYTES;
  const GEOM    = count * (9 + 3 + 4);
  const PALETTE = count * (1 + 3);
  const total   = HEADER + NAMES + GEOM + numVariants * PALETTE;

  const buf  = new ArrayBuffer(total);
  const view = new DataView(buf);
  const u8   = new Uint8Array(buf);

  view.setUint32( 0, SPZV_MAGIC,   true);
  view.setUint32( 4, SPZV_VERSION, true);
  view.setUint32( 8, count,        true);
  view.setUint8 (12, numVariants);
  view.setUint8 (13, fractionalBits);
  view.setUint8 (14, 0); // flags
  view.setUint8 (15, 0); // reserved

  let off = HEADER;
  const encoder = new TextEncoder();
  for (const v of variants) {
    const bytes = encoder.encode(v.name).slice(0, SPZV_NAME_BYTES - 1);
    u8.set(bytes, off);
    off += SPZV_NAME_BYTES;
  }

  const offPos   = off;
  const offScale = offPos   + count * 9;
  const offRot   = offScale + count * 3;
  off = offRot + count * 4;

  const posScale = 1 << fractionalBits;
  for (let i = 0; i < count; i++) {
    const d = i * 16;

    writeInt24(view, offPos + i * 9 + 0, geom[d + 0] * posScale);
    writeInt24(view, offPos + i * 9 + 3, geom[d + 1] * posScale);
    writeInt24(view, offPos + i * 9 + 6, geom[d + 2] * posScale);

    u8[offScale + i * 3 + 0] = encodeScale(geom[d +  8]);
    u8[offScale + i * 3 + 1] = encodeScale(geom[d +  9]);
    u8[offScale + i * 3 + 2] = encodeScale(geom[d + 10]);

    view.setUint32(offRot + i * 4,
      encodeQuat(geom[d + 12], geom[d + 13], geom[d + 14], geom[d + 15]),
      true);
  }

  for (const v of variants) {
    const offAlpha = off;
    const offColor = offAlpha + count;
    for (let i = 0; i < count; i++) {
      const d = i * 16;
      u8[offAlpha + i]         = clampU8(v.data[d + 7] * 255);
      u8[offColor + i * 3 + 0] = clampU8(v.data[d + 4] * 255);
      u8[offColor + i * 3 + 1] = clampU8(v.data[d + 5] * 255);
      u8[offColor + i * 3 + 2] = clampU8(v.data[d + 6] * 255);
    }
    off = offColor + count * 3;
  }

  return u8;
}

/**
 * Encode and gzip-compress N color/material variants to a .spzv ArrayBuffer.
 * @see encodeSpzv
 * @returns {Promise<ArrayBuffer>}
 */
export async function compressVariantsToSpzv(variants, count, opts = {}) {
  const raw = encodeSpzv(variants, count, opts);
  return compressGzip(raw);
}
