/**
 * Splat pruning — removes low-impact Gaussians to cut file size and render
 * cost (sort + rasterization both scale with splat count).
 *
 * Input/output: canonical Float32Array (16 floats/Gaussian), see compress.js
 * for the layout.
 *
 * Two independent stages:
 *   1. Hard filter — drop splats that are nearly invisible regardless of
 *      viewing distance: `alpha < minAlpha` or every scale axis < `minScale`.
 *   2. Importance subsample — of what's left, keep the top `keepFraction`
 *      ranked by `alpha * cbrt(volume)` (small/transparent splats go first).
 *
 * Usage:
 *   import { pruneGaussians, generateLods, PRUNE_PRESETS } from 'holosplat';
 *   const { data, count } = pruneGaussians(srcData, srcCount, PRUNE_PRESETS.balanced);
 */

/** Relative visual contribution of a splat — bigger & more opaque = more important. */
function importance(data, i) {
  const j = i * 16;
  const alpha = data[j + 7];
  const sx = data[j + 8], sy = data[j + 9], sz = data[j + 10];
  return alpha * Math.cbrt(Math.max(0, sx * sy * sz));
}

/**
 * @param {Float32Array} data
 * @param {number}       count
 * @param {object}       [opts]
 * @param {number}       [opts.minAlpha=0]       drop splats with alpha below this (0-1)
 * @param {number}       [opts.minScale=0]       drop splats whose largest scale axis is below this (world units)
 * @param {number}       [opts.keepFraction=1]   keep this fraction of the remaining splats, ranked by importance
 * @returns {{data: Float32Array, count: number}}
 */
export function pruneGaussians(data, count, opts = {}) {
  const { minAlpha = 0, minScale = 0, keepFraction = 1 } = opts;

  let kept = [];
  for (let i = 0; i < count; i++) {
    const j = i * 16;
    if (data[j + 7] < minAlpha) continue;
    if (Math.max(data[j + 8], data[j + 9], data[j + 10]) < minScale) continue;
    kept.push(i);
  }

  if (keepFraction < 1 && kept.length > 0) {
    kept.sort((a, b) => importance(data, b) - importance(data, a));
    kept.length = Math.max(1, Math.round(kept.length * keepFraction));
  }

  const out = new Float32Array(kept.length * 16);
  for (let k = 0; k < kept.length; k++) out.set(data.subarray(kept[k] * 16, kept[k] * 16 + 16), k * 16);
  return { data: out, count: kept.length };
}

/**
 * Generates a series of LOD tiers by progressively keeping fewer, more
 * important splats — each tier's kept set is a prefix of the previous
 * tier's (after the hard filter), so tiers are strictly nested.
 *
 * @param {Float32Array} data
 * @param {number}       count
 * @param {object}       [opts]
 * @param {number}       [opts.minAlpha=0]   hard filter, applied once before ranking
 * @param {number}       [opts.minScale=0]
 * @param {number[]}     [opts.fractions=[1, 0.8, 0.6, 0.4]]  keepFraction per tier
 * @returns {{data: Float32Array, count: number, fraction: number}[]}
 */
export function generateLods(data, count, opts = {}) {
  const { minAlpha = 0, minScale = 0, fractions = [1, 0.8, 0.6, 0.4] } = opts;

  let kept = [];
  for (let i = 0; i < count; i++) {
    const j = i * 16;
    if (data[j + 7] < minAlpha) continue;
    if (Math.max(data[j + 8], data[j + 9], data[j + 10]) < minScale) continue;
    kept.push(i);
  }
  kept.sort((a, b) => importance(data, b) - importance(data, a));

  return fractions.map(fraction => {
    const n = Math.max(1, Math.round(kept.length * fraction));
    const out = new Float32Array(n * 16);
    for (let k = 0; k < n; k++) out.set(data.subarray(kept[k] * 16, kept[k] * 16 + 16), k * 16);
    return { data: out, count: n, fraction };
  });
}

/** Named pruning presets for the prune/LOD UI tools. */
export const PRUNE_PRESETS = {
  light:      { minAlpha: 0.004, minScale: 0.0001, keepFraction: 1 },
  balanced:   { minAlpha: 0.02,  minScale: 0.0005, keepFraction: 0.7 },
  aggressive: { minAlpha: 0.05,  minScale: 0.001,  keepFraction: 0.4 },
  mobile:     { minAlpha: 0.05,  minScale: 0.001,  keepFraction: 0.25 },
};
