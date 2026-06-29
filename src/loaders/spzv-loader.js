/**
 * Loader for the .spzv packed-variant format — see variant-pack.js for the
 * on-disk layout. Decodes to canonical Float32Array geometry (16 floats/
 * Gaussian, color/alpha taken from the first variant) plus the full set of
 * per-variant color+alpha palettes for runtime swapping.
 */

import { fetchWithProgress } from './fetch-utils.js';
import { readInt24, decodeScale, decodeQuat, decompressGzip } from './spz-loader.js';
import { SPZV_MAGIC, SPZV_NAME_BYTES } from '../variant-pack.js';

// Yield to main thread every PARSE_CHUNK Gaussians while decoding — keeps the
// page responsive for large files (see spz-loader.js).
const PARSE_CHUNK = 50000;

function yieldToMain() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

export async function loadSpzv(url, onProgress) {
  const compressed = await fetchWithProgress(url, onProgress);
  const buffer     = await decompressGzip(compressed);
  return parseSpzv(buffer);
}

/** Decompress and parse an in-memory gzip-compressed .spzv buffer (e.g. from a File). */
export async function parseSpzvGzip(compressed) {
  const buffer = await decompressGzip(compressed);
  return parseSpzv(buffer);
}

/**
 * @param {ArrayBuffer} buffer
 * @returns {Promise<{
 *   data: Float32Array,
 *   count: number,
 *   variants: {name: string, palette: Float32Array}[],
 * }>}
 *   `data` is canonical 16-floats/Gaussian layout using the first variant's
 *   palette. Each `variants[i].palette` is `count * 4` floats: r,g,b,a.
 */
export async function parseSpzv(buffer) {
  const view = new DataView(buffer);
  const u8   = new Uint8Array(buffer);

  const magic          = view.getUint32(0, true);
  const version        = view.getUint32(4, true);
  const count          = view.getUint32(8, true);
  const numVariants    = view.getUint8(12);
  const fractionalBits = view.getUint8(13);

  if (magic !== SPZV_MAGIC) {
    throw new Error(
      `Invalid .spzv magic: 0x${magic.toString(16).toUpperCase()} ` +
      `(expected 0x${SPZV_MAGIC.toString(16).toUpperCase()})`
    );
  }
  if (version !== 1) {
    console.warn(`HoloSplat: .spzv version ${version} is untested; attempting load anyway`);
  }

  // ── Variant names ──────────────────────────────────────────────────────────
  let off = 16;
  const decoder = new TextDecoder();
  const names = [];
  for (let v = 0; v < numVariants; v++) {
    const bytes = u8.subarray(off, off + SPZV_NAME_BYTES);
    const nul   = bytes.indexOf(0);
    names.push(decoder.decode(bytes.subarray(0, nul < 0 ? SPZV_NAME_BYTES : nul)));
    off += SPZV_NAME_BYTES;
  }

  // ── Geometry (shared) ─────────────────────────────────────────────────────
  const offPos   = off;
  const offScale = offPos   + count * 9;
  const offRot   = offScale + count * 3;
  off = offRot + count * 4;

  const posDiv = 1 << fractionalBits;
  const data   = new Float32Array(count * 16);

  for (let i = 0; i < count; i++) {
    const d  = i * 16;
    const pb = offPos + i * 9;
    data[d + 0] = readInt24(view, pb + 0) / posDiv;
    data[d + 1] = readInt24(view, pb + 3) / posDiv;
    data[d + 2] = readInt24(view, pb + 6) / posDiv;

    const sb = offScale + i * 3;
    data[d +  8] = decodeScale(u8[sb + 0]);
    data[d +  9] = decodeScale(u8[sb + 1]);
    data[d + 10] = decodeScale(u8[sb + 2]);

    const [qx, qy, qz, qw] = decodeQuat(view.getUint32(offRot + i * 4, true));
    data[d + 12] = qx;
    data[d + 13] = qy;
    data[d + 14] = qz;
    data[d + 15] = qw;

    if ((i + 1) % PARSE_CHUNK === 0) await yieldToMain();
  }

  // ── Palettes (one per variant) ───────────────────────────────────────────────
  const variants = [];
  for (let v = 0; v < numVariants; v++) {
    const offAlpha = off;
    const offColor = offAlpha + count;
    const palette  = new Float32Array(count * 4);

    for (let i = 0; i < count; i++) {
      const p = i * 4;
      palette[p + 0] = u8[offColor + i * 3 + 0] / 255;
      palette[p + 1] = u8[offColor + i * 3 + 1] / 255;
      palette[p + 2] = u8[offColor + i * 3 + 2] / 255;
      palette[p + 3] = u8[offAlpha + i] / 255;

      if (v === 0) {
        const d = i * 16;
        data[d + 4] = palette[p + 0];
        data[d + 5] = palette[p + 1];
        data[d + 6] = palette[p + 2];
        data[d + 7] = palette[p + 3];
      }
    }

    off = offColor + count * 3;
    variants.push({ name: names[v], palette });
    if ((v + 1) % 1 === 0) await yieldToMain();
  }

  return { data, count, variants, shData: null, numSHBases: 0 };
}
