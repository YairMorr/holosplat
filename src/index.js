/**
 * HoloSplat – WebGPU Gaussian Splat viewer library
 *
 * ── Quick start ──────────────────────────────────────────────────────────────
 *
 *   ESM (bundler / native module):
 *     import { create } from 'holosplat';
 *
 *   IIFE (Webflow / plain script tag):
 *     <script src="holosplat.iife.js"></script>
 *     HoloSplat.create({ ... });
 *
 * ── create(options) ──────────────────────────────────────────────────────────
 *
 *   canvas     {string|HTMLCanvasElement}  – required; CSS selector or element
 *   src        {string}                   – URL of .splat or .ply file
 *   background {string|number[]}          – '#rrggbb', '#rrggbbaa', 'transparent',
 *                                           or [r,g,b,a] (0–1). Default '#000000'
 *   fov        {number}                   – vertical field of view, degrees. Default 60
 *   near       {number}                   – near clip plane. Default 0.1
 *   far        {number}                   – far clip plane. Default 2000
 *   splatScale {number}                   – global scale multiplier. Default 1
 *   autoRotate {boolean}                  – slow continuous orbit. Default false
 *   quality    {'auto'|'low'|'medium'|'high'} – device-tier presets for
 *                                           maxPixelRatio/shDegree caps and LOD
 *                                           file selection (see device-tier.js).
 *                                           Default 'auto' (detects the device).
 *   onLoad     {function}                 – called after scene is fully loaded
 *   onProgress {function(0..1)}           – called during fetch with progress 0–1
 *   onError    {function(Error)}          – called on any error
 *
 * ── Returns ───────────────────────────────────────────────────────────────────
 *
 *   A controller object with:
 *     destroy()            – stop rendering, release GPU resources
 *     setBackground(bg)    – change background colour
 *     setSplatScale(s)     – change global splat scale
 *     setAutoRotate(bool)  – toggle auto-rotate
 *     resetCamera()        – fit camera back to scene
 *     getVariants(id?)     – list color/material variant names for a part (or the
 *                            whole scene if `id` is omitted); [] if it has none
 *     setVariant(id?, name) – switch the active color/material variant (async —
 *                            instant for packed .spzv parts, fetches on first use
 *                            for per-file variants; see Viewer#setVariant)
 *     camera               – OrbitCamera instance (for direct manipulation)
 */
import { Viewer } from './viewer.js';
import { player } from './player.js';
import { Animation, loadAnimation, splatNameFromId } from './animation.js';
import { scrollScene } from './scroll-scene.js';
import { compressToSpz, encodeSpz } from './compress.js';
import { parseSplat } from './loaders/splat-loader.js';
import { parsePly } from './loaders/ply-loader.js';
import { parseSpz, parseSpzGzip } from './loaders/spz-loader.js';
import { parseSpzv, parseSpzvGzip } from './loaders/spzv-loader.js';
import { encodeSpzv, compressVariantsToSpzv } from './variant-pack.js';
import { pruneGaussians, generateLods, PRUNE_PRESETS } from './optimize.js';
import { detectDeviceTier, qualityForTier, resolveLodUrl, resolvePartsLod } from './device-tier.js';

export async function create(options = {}) {
  const { onLoad, onError, src, parts, quality = 'auto', ...viewerOpts } = options;

  const tier = quality === 'auto' ? detectDeviceTier() : quality;
  const caps = qualityForTier(tier);

  const maxPixelRatio   = viewerOpts.maxPixelRatio ?? caps.maxPixelRatio;
  const shDegree        = Math.min(viewerOpts.shDegree ?? 0, caps.shDegreeCap);
  const prefetchVariants = viewerOpts.prefetchVariants ?? caps.prefetchVariants;

  const viewer = new Viewer({ ...viewerOpts, maxPixelRatio, shDegree, prefetchVariants, tier });

  const noop = {
    destroy() {}, setBackground() {}, setSplatScale() {}, setGamma() {}, setAaDilation() {},
    setAutoRotate() {}, setFlipY() {}, resetCamera() {}, focusCamera() {},
    getVariants() { return []; }, async setVariant() { return false; },
    getStats() { return null; }, getSplatDebug() { return null; }, camera: null,
    setDebugIndex() {}, async readDebug() { return null; },
  };

  try {
    await viewer.init();
    if (parts)     await viewer.loadParts(await resolvePartsLod(parts, caps.lod));
    else if (src)  await viewer.load(await resolveLodUrl(src, caps.lod));
  } catch (err) {
    viewer.destroy();
    if (onError) { onError(err); return noop; }
    throw err;
  }

  viewer.start();
  onLoad?.();

  return {
    destroy()           { viewer.destroy(); },
    setBackground(bg)   { viewer.setBackground(bg); },
    setSplatScale(s)    { viewer.setSplatScale(s); },
    setGamma(g)         { viewer.setGamma(g); },
    setAaDilation(v)    { viewer.setAaDilation(v); },
    setAutoRotate(v)    { viewer.setAutoRotate(v); },
    setFlipY(v)         { viewer.setFlipY(v); },
    resetCamera()       { viewer.resetCamera(); },
    focusCamera()       { viewer.focusCamera(); },
    getVariants(id)     { return viewer.getVariants(id); },
    getStats()          { return viewer.getStats(); },
    getSplatDebug(i)    { return viewer.getSplatDebug(i); },
    setDebugIndex(i)    { viewer.setDebugIndex(i); },
    readDebug()         { return viewer.readDebug(); },
    setVariant(id, name) { return viewer.setVariant(id, name); },  // async
    get camera()        { return viewer.camera; },
  };
}

// Also expose Viewer class, animation, scroll scene, parsers, compression/pruning,
// and device-tier utilities
export {
  Viewer, player, scrollScene, Animation, loadAnimation, splatNameFromId,
  compressToSpz, encodeSpz, parseSplat, parsePly, parseSpz, parseSpzGzip,
  parseSpzv, parseSpzvGzip, encodeSpzv, compressVariantsToSpzv,
  pruneGaussians, generateLods, PRUNE_PRESETS,
  detectDeviceTier, qualityForTier, resolveLodUrl, resolvePartsLod,
};
