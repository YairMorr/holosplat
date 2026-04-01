/**
 * 16-bit two-pass radix sort for depth ordering.
 *
 * Sorts indices ascending by depth value (smallest depth = farthest = rendered first).
 * All buffers are pre-allocated at construction time — zero allocations per frame.
 */
export function createSorter(maxN) {
  const keys0   = new Uint16Array(maxN);
  const keys1   = new Uint16Array(maxN);
  const idx0    = new Uint32Array(maxN);
  const idx1    = new Uint32Array(maxN);
  const counts  = new Int32Array(256);
  const pfx     = new Int32Array(256);

  /**
   * @param {Float32Array} depths  – one depth value per Gaussian
   * @param {number}       N       – number of Gaussians
   * @returns {Uint32Array}        – sorted indices (view into pre-alloc buffer)
   */
  return function sort(depths, N) {
    // ── Quantize depths to uint16 ──────────────────────────────────────────
    let minD = depths[0], maxD = depths[0];
    for (let i = 1; i < N; i++) {
      if (depths[i] < minD) minD = depths[i];
      if (depths[i] > maxD) maxD = depths[i];
    }
    const range = maxD - minD;
    const scale = range > 0 ? 65535 / range : 0;

    for (let i = 0; i < N; i++) {
      keys0[i] = Math.round((depths[i] - minD) * scale) | 0;
      idx0[i]  = i;
    }

    // ── Pass 1: sort by low byte ───────────────────────────────────────────
    counts.fill(0);
    for (let i = 0; i < N; i++) counts[keys0[i] & 0xff]++;

    pfx[0] = 0;
    for (let i = 1; i < 256; i++) pfx[i] = pfx[i - 1] + counts[i - 1];

    for (let i = 0; i < N; i++) {
      const k   = keys0[i] & 0xff;
      const pos = pfx[k]++;
      keys1[pos] = keys0[i];
      idx1[pos]  = idx0[i];
    }

    // ── Pass 2: sort by high byte (stable → preserves low-byte order) ─────
    counts.fill(0);
    for (let i = 0; i < N; i++) counts[(keys1[i] >> 8) & 0xff]++;

    pfx[0] = 0;
    for (let i = 1; i < 256; i++) pfx[i] = pfx[i - 1] + counts[i - 1];

    for (let i = 0; i < N; i++) {
      const k   = (keys1[i] >> 8) & 0xff;
      const pos = pfx[k]++;
      idx0[pos] = idx1[i];
    }

    return idx0; // ascending depth order = back-to-front
  };
}
