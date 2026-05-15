/**
 * HoloSplat API middleware for Node.js servers.
 *
 * Mounts three routes that the /holosplat editor uses to read and write
 * project files. Plug it into any Connect-compatible server at /hs-api.
 *
 * ── Express / Connect ────────────────────────────────────────────────────────
 *   import { createHsApiHandler } from 'holosplat/server';
 *   app.use('/hs-api', createHsApiHandler());
 *
 * ── Vite (vite.config.js) ────────────────────────────────────────────────────
 *   import { createHsApiHandler } from 'holosplat/server';
 *   export default defineConfig({
 *     plugins: [{
 *       name: 'holosplat',
 *       configureServer(server) {
 *         server.middlewares.use('/hs-api', createHsApiHandler());
 *       },
 *     }],
 *   });
 *
 * ── Next.js pages router  (pages/api/hs-api/[...route].js) ───────────────────
 *   import { createHsApiHandler } from 'holosplat/server';
 *   const handler = createHsApiHandler();
 *   export default function hsApi(req, res) {
 *     req.url = '/' + (req.query.route ?? []).join('/') +
 *               (req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '');
 *     handler(req, res);
 *   }
 *   export const config = { api: { bodyParser: false } };
 *
 * ── Routes served ─────────────────────────────────────────────────────────────
 *   GET  /hs-api/ls              List .spz/.ply/.splat and .json files
 *   GET  /hs-api/file?path=<rel> Read a file (relative to project root)
 *   PUT  /hs-api/file?path=<rel> Write a file
 */

import fs   from 'fs';
import path from 'path';

const SCENE_EXTS  = new Set(['.spz', '.splat', '.ply']);
const SKIP_JSON   = new Set(['package.json', 'package-lock.json', 'tsconfig.json',
                              'tsconfig.node.json', 'jsconfig.json']);

/**
 * @param {string} [root=process.cwd()]  Project root — all paths resolved relative to this.
 * @returns {(req, res, next?) => void}  Connect/Express-compatible middleware.
 */
export function createHsApiHandler(root = process.cwd()) {

  // ── Path safety ─────────────────────────────────────────────────────────────

  function safePath(rel) {
    if (!rel) return null;
    const full    = path.resolve(root, rel);
    const relBack = path.relative(root, full);
    // Reject traversal outside root or attempts to read root itself
    if (relBack.startsWith('..') || path.isAbsolute(relBack)) return null;
    if (full === root) return null;
    return full;
  }

  // ── Helpers ──────────────────────────────────────────────────────────────────

  function sendJson(res, status, data) {
    const body = JSON.stringify(data);
    res.writeHead(status, {
      'Content-Type':   'application/json',
      'Content-Length': Buffer.byteLength(body),
    });
    res.end(body);
  }

  function scanDir(dir, prefix, out) {
    if (!fs.existsSync(dir)) return;
    for (const f of fs.readdirSync(dir).sort()) {
      const full = path.join(dir, f);
      if (!fs.statSync(full).isFile()) continue;
      const ext = path.extname(f).toLowerCase();
      const rel = prefix ? `${prefix}/${f}` : f;
      if (SCENE_EXTS.has(ext))  out.spz.push(rel);
      else if (ext === '.json') out.json.push(rel);
    }
  }

  // ── Handler ──────────────────────────────────────────────────────────────────

  return function hsApiHandler(req, res, next) {
    const url    = new URL(req.url, 'http://x');
    const route  = url.pathname;
    const params = Object.fromEntries(url.searchParams);

    res.setHeader('Access-Control-Allow-Origin',  '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, PUT, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    res.setHeader('Cache-Control', 'no-cache');

    if (req.method === 'OPTIONS') {
      res.writeHead(204); return res.end();
    }

    // GET /ls
    if (route === '/ls' && req.method === 'GET') {
      const result = { spz: [], json: [] };
      // Common static asset locations across different project setups
      scanDir(path.join(root, 'public', 'scenes'), 'public/scenes', result);
      scanDir(path.join(root, 'public'),            'public',        result);
      scanDir(path.join(root, 'scenes'),            'scenes',        result);
      scanDir(path.join(root, 'blender'),           'blender',       result);
      // Root-level JSON files (hs-config, blender exports, etc.)
      if (fs.existsSync(root)) {
        for (const f of fs.readdirSync(root).sort()) {
          if (!f.endsWith('.json') || SKIP_JSON.has(f)) continue;
          if (fs.statSync(path.join(root, f)).isFile()) result.json.push(f);
        }
      }
      return sendJson(res, 200, result);
    }

    // GET /file?path=...
    if (route === '/file' && req.method === 'GET') {
      const full = safePath(params.path);
      if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
      const body = fs.readFileSync(full);
      res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Length': body.length });
      return res.end(body);
    }

    // PUT /file?path=...
    if (route === '/file' && req.method === 'PUT') {
      const full = safePath(params.path);
      if (!full) return sendJson(res, 403, { error: 'forbidden' });
      const chunks = [];
      req.on('data', c => chunks.push(c));
      req.on('end', () => {
        try {
          const buf = Buffer.concat(chunks);
          const dir = path.dirname(full);
          if (dir && dir !== root) fs.mkdirSync(dir, { recursive: true });
          fs.writeFileSync(full, buf);
          sendJson(res, 200, { ok: true });
        } catch (e) {
          sendJson(res, 500, { error: e.message });
        }
      });
      return;
    }

    if (typeof next === 'function') next();
    else { res.writeHead(404); res.end('Not found'); }
  };
}
