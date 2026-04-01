/**
 * Loader for the .splat binary format.
 *
 * Each Gaussian is 32 bytes:
 *   bytes  0–11 : position     (3 × float32, little-endian)
 *   bytes 12–23 : scale        (3 × float32, already linear – no exp needed)
 *   bytes 24–27 : color RGBA   (4 × uint8, 0–255)
 *   bytes 28–31 : quaternion   (4 × uint8, w/x/y/z order, decoded via (b−128)/128)
 *
 * Output: Float32Array with 16 floats per Gaussian in the canonical layout:
 *   [0-2]  pos.xyz
 *   [3]    0 (padding)
 *   [4-7]  color.rgba (0–1)
 *   [8-10] scale.xyz
 *   [11]   0 (padding)
 *   [12-15] quat.xyzw
 */
export async function loadSplat(url, onProgress) {
  const buffer = await fetchWithProgress(url, onProgress);
  return parseSplat(buffer);
}

export function parseSplat(buffer) {
  const STRIDE = 32;
  const N      = Math.floor(buffer.byteLength / STRIDE);
  if (N === 0) throw new Error('Empty or invalid .splat file');

  const src  = new DataView(buffer);
  const data = new Float32Array(N * 16);

  for (let i = 0; i < N; i++) {
    const s  = i * STRIDE;
    const d  = i * 16;

    // Position
    data[d + 0] = src.getFloat32(s +  0, true);
    data[d + 1] = src.getFloat32(s +  4, true);
    data[d + 2] = src.getFloat32(s +  8, true);
    // d[3] = 0 (padding)

    // Color (uint8 → 0..1)
    data[d + 4] = src.getUint8(s + 24) / 255;
    data[d + 5] = src.getUint8(s + 25) / 255;
    data[d + 6] = src.getUint8(s + 26) / 255;
    data[d + 7] = src.getUint8(s + 27) / 255;

    // Scale (already linear in .splat format)
    data[d +  8] = src.getFloat32(s + 12, true);
    data[d +  9] = src.getFloat32(s + 16, true);
    data[d + 10] = src.getFloat32(s + 20, true);
    // d[11] = 0 (padding)

    // Quaternion: bytes 28–31 = [w, x, y, z], decoded (b−128)/128, then normalize
    const qw = (src.getUint8(s + 28) - 128) / 128;
    const qx = (src.getUint8(s + 29) - 128) / 128;
    const qy = (src.getUint8(s + 30) - 128) / 128;
    const qz = (src.getUint8(s + 31) - 128) / 128;
    const ql = Math.hypot(qx, qy, qz, qw) || 1;
    data[d + 12] = qx / ql;
    data[d + 13] = qy / ql;
    data[d + 14] = qz / ql;
    data[d + 15] = qw / ql;
  }

  return { data, count: N };
}

// fetchWithProgress is defined in fetch-utils.js (shared)
