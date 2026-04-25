/**
 * Async 32-bit radix sort for depth ordering.
 *
 * Converts IEEE-754 view-space depths (always negative) to sortable uint32
 * keys by flipping all bits — gives full float precision with zero per-frame
 * rescaling, eliminating the shimmer caused by the previous 16-bit approach.
 *
 * 4 passes × 8-bit buckets.  Runs in a Web Worker so the main thread never
 * blocks on sorting.  Falls back to synchronous if Workers are unavailable.
 */

// ── Shared sort logic (used by both worker and sync fallback) ─────────────────

const SORT_FN = `
function radixSort32(keys, idx, tmp_k, tmp_i, counts, pfx, N) {
  for (let pass = 0; pass < 4; pass++) {
    const shift = pass * 8;
    counts.fill(0);
    for (let i = 0; i < N; i++) counts[(keys[i] >>> shift) & 0xff]++;
    pfx[0] = 0;
    for (let i = 1; i < 256; i++) pfx[i] = pfx[i-1] + counts[i-1];
    for (let i = 0; i < N; i++) {
      const b = (keys[i] >>> shift) & 0xff;
      tmp_k[pfx[b]]   = keys[i];
      tmp_i[pfx[b]++] = idx[i];
    }
    // swap
    const k = keys; keys = tmp_k; tmp_k = k;
    const x = idx;  idx  = tmp_i; tmp_i = x;
  }
  // after 4 passes (even) idx holds the result
  return idx;
}
`;

// ── Worker source ─────────────────────────────────────────────────────────────

const WORKER_SRC = SORT_FN + `
const u32 = new Uint32Array(1);
const f32 = new Float32Array(u32.buffer);

let keys0, keys1, idx0, idx1, counts, pfx, allocN = 0;
function alloc(N) {
  if (N <= allocN) return;
  allocN = N;
  keys0  = new Uint32Array(N);
  keys1  = new Uint32Array(N);
  idx0   = new Uint32Array(N);
  idx1   = new Uint32Array(N);
  counts = new Uint32Array(256);
  pfx    = new Uint32Array(256);
}

self.onmessage = function(e) {
  const depths = e.data.depths;
  const N      = e.data.N;
  alloc(N);

  // Convert negative floats → sortable uint32 by flipping all bits
  const dv = new DataView(depths.buffer);
  for (let i = 0; i < N; i++) {
    keys0[i] = dv.getUint32(i * 4, true) ^ 0xffffffff;
    idx0[i]  = i;
  }

  const sorted = radixSort32(keys0, idx0, keys1, idx1, counts, pfx, N);
  const result = sorted.slice(0, N);
  self.postMessage({ order: result }, [result.buffer]);
};
`;

// ── Synchronous fallback ──────────────────────────────────────────────────────

function createSyncSorter(maxN) {
  let keys0  = new Uint32Array(maxN);
  let keys1  = new Uint32Array(maxN);
  let idx0   = new Uint32Array(maxN);
  let idx1   = new Uint32Array(maxN);
  const counts = new Uint32Array(256);
  const pfx    = new Uint32Array(256);

  // inline radixSort32
  function sort(depths, N) {
    const dv = new DataView(depths.buffer);
    for (let i = 0; i < N; i++) {
      keys0[i] = dv.getUint32(i * 4, true) ^ 0xffffffff;
      idx0[i]  = i;
    }
    for (let pass = 0; pass < 4; pass++) {
      const shift = pass * 8;
      counts.fill(0);
      for (let i = 0; i < N; i++) counts[(keys0[i] >>> shift) & 0xff]++;
      pfx[0] = 0;
      for (let i = 1; i < 256; i++) pfx[i] = pfx[i-1] + counts[i-1];
      for (let i = 0; i < N; i++) {
        const b = (keys0[i] >>> shift) & 0xff;
        keys1[pfx[b]]   = keys0[i];
        idx1[pfx[b]++]  = idx0[i];
      }
      [keys0, keys1] = [keys1, keys0];
      [idx0,  idx1 ] = [idx1,  idx0 ];
    }
    return idx0;
  }

  return sort;
}

// ── Async worker sorter ───────────────────────────────────────────────────────

export function createSorter(maxN) {
  if (typeof Worker === 'undefined') return createSyncSorter(maxN);

  let worker;
  try {
    const blob = new Blob([WORKER_SRC], { type: 'application/javascript' });
    const url  = URL.createObjectURL(blob);
    worker = new Worker(url);
    URL.revokeObjectURL(url);
  } catch (_) {
    return createSyncSorter(maxN);
  }

  // Identity order until first worker result arrives
  let lastOrder = new Uint32Array(maxN);
  for (let i = 0; i < maxN; i++) lastOrder[i] = i;

  let busy   = false;
  let queued = null;   // latest pending request while worker is busy

  worker.onmessage = (e) => {
    lastOrder = e.data.order;
    busy = false;
    if (queued) {
      const { depths, N } = queued;
      queued = null;
      busy = true;
      worker.postMessage({ depths, N }, [depths.buffer]);
    }
  };

  return function sort(depths, N) {
    const copy = new Float32Array(N);
    copy.set(depths.subarray(0, N));

    if (!busy) {
      busy = true;
      worker.postMessage({ depths: copy, N }, [copy.buffer]);
    } else {
      queued = { depths: copy, N };   // drop stale request, keep latest
    }

    return lastOrder;
  };
}
