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
 *     camera               – OrbitCamera instance (for direct manipulation)
 */
import { Viewer } from './viewer.js';
import { player } from './player.js';
import { Animation, loadAnimation } from './animation.js';
import { compressToSpz, encodeSpz } from './compress.js';
import { parseSplat } from './loaders/splat-loader.js';
import { parsePly } from './loaders/ply-loader.js';

export async function create(options = {}) {
  const { onLoad, onError, src, ...viewerOpts } = options;

  const viewer = new Viewer({ ...viewerOpts });

  const noop = {
    destroy() {}, setBackground() {}, setSplatScale() {},
    setAutoRotate() {}, resetCamera() {}, camera: null,
  };

  try {
    await viewer.init();
    if (src) await viewer.load(src);
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
    setAutoRotate(v)    { viewer.setAutoRotate(v); },
    resetCamera()       { viewer.resetCamera(); },
    get camera()        { return viewer.camera; },
  };
}

// Also expose Viewer class, animation, parsers, and compression utilities
export { Viewer, player, Animation, loadAnimation, compressToSpz, encodeSpz, parseSplat, parsePly };
