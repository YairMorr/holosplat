/**
 * Loader for the .spz Gaussian splat format (Niantic Labs).
 *
 * The file is gzip-compressed. After decompression:
 *
 * Header (16 bytes, little-endian):
 *   [0-3]  magic    uint32  = 0x5053474E  ('N','G','S','P' → LE 0x5053474E)
 *   [4-7]  version  uint32  (2 = xyz-int8 rot, 3-4 = smallest-3 rot)
 *   [8-11] numPoints uint32
 *   [12]   shDegree  uint8
 *   [13]   fractionalBits uint8  (position precision; typically 12)
 *   [14]   flags     uint8
 *   [15]   reserved  uint8
 *
 * Data sections (column-organized, all for numPoints Gaussians):
 *   Positions  : numPoints × 9 bytes  (3 × int24 LE signed, ÷ 2^fractionalBits)
 *   Alphas     : numPoints × 1 byte   (uint8 → opacity 0–1)
 *   Colors     : numPoints × 3 bytes  (uint8 RGB, 0–255 → 0–1)
 *   Scales     : numPoints × 3 bytes  (uint8, log-space, exp((b–128)/16))
 *   Rotations  : v2:   numPoints × 3 bytes  (int8 x,y,z; w = sqrt(1-x²-y²-z²))
 *                v3-4: numPoints × 4 bytes  (2-bit largestIdx + three 10-bit signed)
 *
 * Output: canonical Float32Array layout (16 floats/Gaussian) compatible with
 *         the HoloSplat renderer — see shaders.js for the GPU-side layout.
 */

const MAGIC    = 0x5053474E;
const SQRT2INV = 1 / Math.SQRT2;

export async function loadSpz(url, onProgress) {
  const compressed = await fetchWithProgress(url, onProgress);
  const buffer     = await decompressGzip(compressed);
  return parseSpz(buffer);
}

export function parseSpz(buffer) {
  const view = new DataView(buffer);

  // ── Header ───────────────────────────────────────────────────────────────
  const magic          = view.getUint32(0, true);
  const version        = view.getUint32(4, true);
  const numPoints      = view.getUint32(8, true);
  const shDegree       = view.getUint8(12);
  const fractionalBits = view.getUint8(13);
  // flags / reserved at 14-15 (ignored for now)

  if (magic !== MAGIC) {
    throw new Error(
      `Invalid .spz magic: 0x${magic.toString(16).toUpperCase()} ` +
      `(expected 0x${MAGIC.toString(16).toUpperCase()})`
    );
  }
  if (version < 2 || version > 4) {
    console.warn(`HoloSplat: .spz version ${version} is untested; attempting load anyway`);
  }

  const posDiv  = 1 << fractionalBits;  // position divisor
  const rotSize = version >= 3 ? 4 : 3; // bytes per rotation

  // ── Section offsets ───────────────────────────────────────────────────────
  const offPos   = 16;
  const offAlpha = offPos   + numPoints * 9;
  const offColor = offAlpha + numPoints * 1;
  const offScale = offColor + numPoints * 3;
  const offRot   = offScale + numPoints * 3;
  // SH rest (if shDegree > 0): offRot + numPoints * rotSize — not used here

  const data = new Float32Array(numPoints * 16);

  for (let i = 0; i < numPoints; i++) {
    const d = i * 16;

    // ── Position (int24 signed LE, ÷ 2^fractionalBits) ───────────────────
    const pb = offPos + i * 9;
    data[d + 0] = readInt24(view, pb + 0) / posDiv;
    data[d + 1] = readInt24(view, pb + 3) / posDiv;
    data[d + 2] = readInt24(view, pb + 6) / posDiv;
    // d[3] = 0 (padding)

    // ── Color (uint8 RGB → 0-1) ───────────────────────────────────────────
    const cb = offColor + i * 3;
    data[d + 4] = view.getUint8(cb + 0) / 255;
    data[d + 5] = view.getUint8(cb + 1) / 255;
    data[d + 6] = view.getUint8(cb + 2) / 255;

    // ── Alpha (uint8 → 0-1) ───────────────────────────────────────────────
    data[d + 7] = view.getUint8(offAlpha + i) / 255;

    // ── Scale (uint8 log-space → linear) ─────────────────────────────────
    const sb = offScale + i * 3;
    data[d +  8] = Math.exp((view.getUint8(sb + 0) - 128) / 16);
    data[d +  9] = Math.exp((view.getUint8(sb + 1) - 128) / 16);
    data[d + 10] = Math.exp((view.getUint8(sb + 2) - 128) / 16);
    // d[11] = 0 (padding)

    // ── Rotation ─────────────────────────────────────────────────────────
    const rb = offRot + i * rotSize;
    let qx, qy, qz, qw;

    if (version >= 3) {
      // Smallest-3 encoding: 4 bytes
      //   bits [ 1: 0]  largestIdx (0=x, 1=y, 2=z, 3=w)
      //   bits [11: 2]  component a (int10 signed)
      //   bits [21:12]  component b (int10 signed)
      //   bits [31:22]  component c (int10 signed)
      const u32  = view.getUint32(rb, true);
      const idx  = u32 & 3;
      const s    = SQRT2INV / 512;              // int10 → [-1/√2, 1/√2]
      const a    = signExtend10((u32 >>  2) & 0x3FF) * s;
      const b_   = signExtend10((u32 >> 12) & 0x3FF) * s;
      const c    = signExtend10((u32 >> 22) & 0x3FF) * s;
      const d_   = Math.sqrt(Math.max(0, 1 - a*a - b_*b_ - c*c));

      // Re-insert the reconstructed largest component
      switch (idx) {
        case 0: qx = d_; qy = a;  qz = b_; qw = c;  break;
        case 1: qx = a;  qy = d_; qz = b_; qw = c;  break;
        case 2: qx = a;  qy = b_; qz = d_; qw = c;  break;
        default: qx = a; qy = b_; qz = c;  qw = d_;
      }
    } else {
      // v2: int8 x,y,z; reconstruct w = sqrt(1 - x²-y²-z²)
      qx = view.getInt8(rb + 0) / 128;
      qy = view.getInt8(rb + 1) / 128;
      qz = view.getInt8(rb + 2) / 128;
      qw = Math.sqrt(Math.max(0, 1 - qx*qx - qy*qy - qz*qz));
    }

    const ql = Math.hypot(qx, qy, qz, qw) || 1;
    data[d + 12] = qx / ql;
    data[d + 13] = qy / ql;
    data[d + 14] = qz / ql;
    data[d + 15] = qw / ql;
  }

  return { data, count: numPoints };
}

// ── Binary helpers ─────────────────────────────────────────────────────────

function readInt24(view, offset) {
  const lo = view.getUint8(offset);
  const mi = view.getUint8(offset + 1);
  const hi = view.getInt8(offset + 2);  // signed → sign extension
  return lo | (mi << 8) | (hi << 16);
}

function signExtend10(v) {
  // Sign-extend 10-bit integer to JS number
  return v & 0x200 ? v | 0xFFFFFC00 : v;
}

// ── Gzip decompression (uses browser DecompressionStream API) ──────────────

async function decompressGzip(buffer) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('DecompressionStream is not available in this environment');
  }
  const stream = new DecompressionStream('gzip');
  const writer = stream.writable.getWriter();
  writer.write(buffer);
  writer.close();
  return new Response(stream.readable).arrayBuffer();
}

// fetchWithProgress is defined in fetch-utils.js (shared)
