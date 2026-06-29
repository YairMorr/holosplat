/**
 * Lightweight device-capability heuristics, used by `create()`/`player()` to
 * pick sane defaults with zero configuration ("quality: 'auto'", the default).
 *
 * Tiers and what they cap:
 *   'low'    – maxPixelRatio 1, SH capped at degree 1, prefers a
 *              `<name>.lods/<name>.lod3.spz` sibling of the scene file if
 *              one exists (40% of splats kept — the most aggressive tier
 *              `generateLods` produces), no background variant prefetch.
 *   'medium' – maxPixelRatio 1.5, SH capped at degree 1, prefers `lod2`
 *              (60% of splats kept), no background variant prefetch.
 *   'high'   – no caps, loads the file as given (`lod0`, 100% of splats),
 *              background variant prefetch enabled.
 *
 * 'low' previously disabled SH entirely (cap 0) — going from full SH to zero
 * is such a visually drastic, "looks broken" regression (flat shading reads
 * as fundamentally lower quality than competing viewers, not just slightly
 * softer) that it's a bad trade even on weak devices, and it silently made
 * any SH-degree control (e.g. the /holosplat editor's) look like it was
 * doing nothing on auto-detected 'low'-tier machines. Capping at 1 still
 * saves most of the cost without that.
 *
 * LOD tiers are produced by examples/prune.html ("Generate LOD tiers"), in a
 * `<name>.lods/` folder next to the source file.
 */

const TIER_SETTINGS = {
  low:    { maxPixelRatio: 1,   shDegreeCap: 1,        lod: 3, prefetchVariants: false },
  medium: { maxPixelRatio: 1.5, shDegreeCap: 1,        lod: 2, prefetchVariants: false },
  high:   { maxPixelRatio: 2,   shDegreeCap: Infinity, lod: 0, prefetchVariants: true },
};

/**
 * Classifies the current device into a quality tier using `navigator`
 * heuristics (memory, core count, save-data, mobile UA). Returns 'high' in
 * non-browser environments or when nothing suggests a weaker device.
 *
 * Mobile GPUs (even on phones with 8GB RAM / 8 cores, e.g. Pixel 7) can't
 * sustain even the 'medium' tier — measured ~10fps on a Pixel 7 at
 * maxPixelRatio 1.5 / lod 1. Memory/core-count alone overestimates their
 * fill-rate budget, so any mobile UA is capped at 'low' regardless of specs.
 *
 * @returns {'low'|'medium'|'high'}
 */
export function detectDeviceTier() {
  if (typeof navigator === 'undefined') return 'high';
  if (navigator.connection?.saveData) return 'low';

  const mem    = navigator.deviceMemory; // GiB — Chrome/Edge/Android only
  const cores  = navigator.hardwareConcurrency || 4;
  const mobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent || '');

  if ((mem != null && mem <= 2) || cores <= 2 || mobile) return 'low';
  if ((mem != null && mem <= 4) || cores <= 4) return 'medium';
  return 'high';
}

/** @param {'low'|'medium'|'high'} tier */
export function qualityForTier(tier) {
  return TIER_SETTINGS[tier] || TIER_SETTINGS.high;
}

/**
 * Probes for a `<name>.lods/<name>.lod{N}.spz` sibling folder of `url` and
 * returns it if found; otherwise returns `url` unchanged. `lod` of 0 always
 * returns `url` without a network request. LOD tiers are always `.spz`
 * regardless of the source file's own format (see examples/prune.html,
 * which generates them via compressToSpz).
 *
 * @param {string} url
 * @param {number} lod
 * @returns {Promise<string>}
 */
export async function resolveLodUrl(url, lod) {
  if (!lod) return url;
  const m = url.match(/\.(spz|ply|splat)$/i);
  if (!m) return url;
  const base  = url.slice(0, -m[0].length);
  const slash = base.lastIndexOf('/');
  const dir   = base.slice(0, slash + 1);
  const name  = base.slice(slash + 1);
  const candidate = `${dir}${name}.lods/${name}.lod${lod}.spz`;
  try {
    const res = await fetch(candidate, { method: 'HEAD', cache: 'no-store' });
    if (res.ok) return candidate;
  } catch { /* fall through to original url */ }
  return url;
}

/**
 * Applies {@link resolveLodUrl} across a `parts` map. Values may be a single
 * URL, an array of URLs, or `{ url, variants }` (lazy per-variant parts —
 * only `url` is LOD-resolved; `variants` passes through unchanged).
 */
export async function resolvePartsLod(parts, lod) {
  if (!lod) return parts;
  const out = {};
  for (const [id, value] of Object.entries(parts)) {
    if (Array.isArray(value)) {
      out[id] = await Promise.all(value.map(u => resolveLodUrl(u, lod)));
    } else if (value && typeof value === 'object') {
      out[id] = { ...value, url: await resolveLodUrl(value.url, lod) };
    } else {
      out[id] = await resolveLodUrl(value, lod);
    }
  }
  return out;
}
