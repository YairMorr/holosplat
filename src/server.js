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
 *   GET  /hs-api/page-source?page=<url> Read the source file mapped to a page URL
 *   POST /hs-api/html-attr       Patch attributes on a tag in an HTML source file
 *   POST /hs-api/init-player     Scaffold a blank player() call into a page with none
 *   POST /hs-api/js-sh           Patch the `sh:` property in a player({...}) call
 *   POST /hs-api/js-anim         Patch the `animation:` property
 *   POST /hs-api/js-parts        Patch the `parts:` property
 *   POST /hs-api/js-partsDir     Patch the `partsDir:` property
 *   POST /hs-api/js-zIndex       Patch the `zIndex:` property
 *   POST /hs-api/js-aaDilation   Patch the `aaDilation:` property
 *   POST /hs-api/js-scenes       Patch the `scenes:` property
 *   POST /hs-api/js-masks        Patch the `masks:` property
 *   POST /hs-api/js-clips        Patch the `clips:` property
 *
 * These mirror server.py's routes of the same name — see that file for the
 * canonical implementation/comments; this is a straight Node.js port so both
 * server.py and this Node middleware support the full ?hs overlay editor.
 */

import fs   from 'fs';
import path from 'path';

const SCENE_EXTS  = new Set(['.spz', '.splat', '.ply', '.spzv']);
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

  function serveFile(res, full) {
    if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
    const body = fs.readFileSync(full);
    const ext  = path.extname(full).toLowerCase();
    const mime = ext === '.html' || ext === '.htm' ? 'text/html'
               : ext === '.json'                   ? 'application/json'
               : 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': mime, 'Content-Length': body.length });
    res.end(body);
  }

  // hs-pages.json (project root) maps a URL path to the source file that
  // holds its player() call — e.g. {"/colors": "src/components/
  // ColorsPlayer.tsx"}. Needed for framework projects (Next.js, etc.)
  // where a URL has no file of the same name; an AI coding assistant (or a
  // human) creates the file and adds the mapping — see the "ask your AI
  // assistant" prompt the editor shows when Init page can't find a literal
  // file (holosplat/editor.js's askClaudePrompt). Optional: if absent,
  // every route below falls back to the legacy literal-path convention.
  function loadPagesMap() {
    const file = path.join(root, 'hs-pages.json');
    if (!fs.existsSync(file)) return {};
    try {
      const data = JSON.parse(fs.readFileSync(file, 'utf8'));
      return data && typeof data === 'object' ? data : {};
    } catch {
      return {};
    }
  }

  // Resolve an hs-api request's `page` field (the requesting page's URL,
  // e.g. the browser's location.pathname) to a safe file path, or null.
  // Checks hs-pages.json first, then falls back to treating the URL path
  // itself as a project-relative file path ('/' -> index.html, matching
  // plain static-HTML sites where the URL already is the filename). Shared
  // by every js-* / html-attr / init-player / page-source route below.
  function pagePath(page) {
    const clean  = (page || '').split('?')[0];
    const mapped = loadPagesMap()[clean];
    if (mapped) return safePath(mapped);
    return safePath(clean.replace(/^\//, '') || 'index.html');
  }

  // Detect the indentation used by the `animation:` line in a player({...})
  // call (or fall back to 6 spaces) — every js-* route lines new properties
  // up with the rest of the call.
  function detectIndent(html) {
    const m = /^([ \t]*)animation\s*:/m.exec(html);
    return m ? m[1] : '      ';
  }

  // Replace a `player({...})` config property with `newLine`.
  //
  // Removes every existing line matching `removeRe` first (heals duplicate
  // lines left behind by earlier saves), then inserts `newLine` once at the
  // first matching location in `insertRes` — regexes for "insert after this
  // line", or starting with `\n` to mean "insert before this match" (used
  // for the closing `});`). Mirrors server.py's _set_js_config_line.
  function setJsConfigLine(html, removeRe, newLine, insertRes) {
    const updated = html.replace(removeRe, '');
    for (const re of insertRes) {
      const m = re.exec(updated);
      if (m) {
        const insertBefore = re.source.startsWith('\\n');
        const ins = insertBefore ? m.index + 1 : m.index + m[0].length;
        return updated.slice(0, ins) + newLine + '\n' + updated.slice(ins);
      }
    }
    return updated;
  }

  // Shared insert-anchor patterns, in priority order, for properties that
  // can follow animation:/flipY:/scenes: or otherwise land right before the
  // call's closing `});`.
  const ANIM_FLIPY_SCENES_CLOSE = [
    /[ \t]*animation\s*:[^\n]*\n/m,
    /[ \t]*flipY\s*:[^\n]*\n/m,
    /[ \t]*scenes\s*:[^\n]*\n/m,
    /\n[ \t]*\}\s*\)\s*;/m,
  ];
  const ANIM_CLOSE = [
    /[ \t]*animation\s*:[^\n]*\n/m,
    /\n[ \t]*\}\s*\)\s*;/m,
  ];

  // Read a request body to completion and parse it as JSON; calls
  // `handler(body, req, res)` with the parsed object, or sends a 400 on
  // invalid JSON.
  function withJsonBody(req, res, handler) {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => {
      let body;
      try {
        body = JSON.parse(Buffer.concat(chunks).toString());
      } catch {
        return sendJson(res, 400, { error: 'invalid JSON' });
      }
      try {
        handler(body);
      } catch (e) {
        sendJson(res, 500, { error: e.message });
      }
    });
  }

  // Recursive — scene assets live in subfolders (e.g. scenes/headphones/).
  function scanDir(dir, prefix, out) {
    if (!fs.existsSync(dir)) return;
    for (const f of fs.readdirSync(dir).sort()) {
      const full = path.join(dir, f);
      const stat = fs.statSync(full);
      const rel = prefix ? `${prefix}/${f}` : f;
      if (stat.isDirectory()) { scanDir(full, rel, out); continue; }
      if (!stat.isFile()) continue;
      const ext = path.extname(f).toLowerCase();
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
    res.setHeader('Access-Control-Allow-Methods', 'GET, PUT, POST, OPTIONS');
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
      result.spz.sort();
      result.json.sort();
      return sendJson(res, 200, result);
    }

    // GET /file?path=...
    if (route === '/file' && req.method === 'GET') {
      return serveFile(res, safePath(params.path));
    }

    // GET /page-source?page=<url> — like /file but resolves `page` (a URL,
    // e.g. location.pathname) via hs-pages.json / the legacy literal-path
    // convention instead of taking a literal repo path. Used by the editor
    // to read the current page's source for displaying already-saved
    // config (see editor.js's loadPageState).
    if (route === '/page-source' && req.method === 'GET') {
      return serveFile(res, pagePath(params.page));
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

    // POST /html-attr  — patch data-hs-scene / data-hs attributes in an HTML source file
    if (route === '/html-attr' && req.method === 'POST') {
      const chunks = [];
      req.on('data', c => chunks.push(c));
      req.on('end', () => {
        try {
          const { page, id, attrs } = JSON.parse(Buffer.concat(chunks).toString());
          if (!page || !id || !attrs) return sendJson(res, 400, { error: 'missing fields' });

          // Resolve page path — strip leading slash and query string
          const rel = page.replace(/^\//, '').split('?')[0];
          const full = safePath(rel);
          if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'file not found' });

          let html = fs.readFileSync(full, 'utf8');
          // Find the opening tag that contains id="<id>"
          const idPat  = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
          const tagRx  = new RegExp(`(<[a-zA-Z][^>]*?\\s(?:id="${idPat}")[^>]*?)(?=\\s*>)`, 's');
          const match  = html.match(tagRx);
          if (!match) return sendJson(res, 404, { error: `element #${id} not found in ${rel}` });

          let tag = match[1];
          for (const [name, value] of Object.entries(attrs)) {
            const n = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            tag = tag.replace(new RegExp(`\\s+${n}(?:="[^"]*"|='[^']*')?`), '');
            if (value != null) tag += ` ${name}="${String(value).replace(/"/g, '&quot;')}"`;
          }
          html = html.slice(0, match.index) + tag + html.slice(match.index + match[0].length);
          fs.writeFileSync(full, html, 'utf8');
          sendJson(res, 200, { ok: true });
        } catch (e) {
          sendJson(res, 500, { error: e.message });
        }
      });
      return;
    }

    // POST /init-player — scaffold a blank player() call into a page with
    // none yet (see server.py's _api_init_player for the canonical docs).
    if (route === '/init-player' && req.method === 'POST') {
      withJsonBody(req, res, ({ page }) => {
        const full = pagePath(page);
        if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
        let html = fs.readFileSync(full, 'utf8');
        if (/\bplayer\s*\(/.test(html)) return sendJson(res, 409, { error: 'page already has a player() call' });
        const bodyClose = /<\/body\s*>/i.exec(html);
        if (!bodyClose) return sendJson(res, 400, { error: 'no </body> tag found' });
        const snippet =
          '\n  <div id="hs-main" style="position:fixed;inset:0;z-index:0"></div>\n' +
          '  <script type="module">\n' +
          "    import { player } from '/dist/holosplat.esm.js';\n" +
          "    const api = player('#hs-main', {\n" +
          '    });\n' +
          '  </script>\n';
        html = html.slice(0, bodyClose.index) + snippet + html.slice(bodyClose.index);
        fs.writeFileSync(full, html, 'utf8');
        sendJson(res, 200, { ok: true });
      });
      return;
    }

    // POST /js-sh — patch the `sh:` (global SH degree) property
    if (route === '/js-sh' && req.method === 'POST') {
      withJsonBody(req, res, ({ page, sh }) => {
        const full = pagePath(page);
        if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
        let html = fs.readFileSync(full, 'utf8');
        const indent = detectIndent(html);
        const newLine = `${indent}sh: ${parseInt(sh, 10) || 0}, // hs-sh`;
        html = setJsConfigLine(
          html,
          /^[ \t]*sh\s*:\s*\d+[^\n]*\/\/\s*hs-sh[^\n]*\n?/gm,
          newLine,
          ANIM_CLOSE
        );
        fs.writeFileSync(full, html, 'utf8');
        sendJson(res, 200, { ok: true });
      });
      return;
    }

    // POST /js-anim — patch the `animation:` (animation JSON URL) property
    if (route === '/js-anim' && req.method === 'POST') {
      withJsonBody(req, res, ({ page, url }) => {
        const full = pagePath(page);
        if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
        let html = fs.readFileSync(full, 'utf8');
        const indent  = detectIndent(html);
        const escaped = (url || '').replace(/'/g, "\\'");
        const newLine = `${indent}animation: '${escaped}',`;
        html = setJsConfigLine(
          html,
          /^[ \t]*animation\s*:\s*(['"])[^'"]*\1[^\n]*\n?/gm,
          newLine,
          ANIM_FLIPY_SCENES_CLOSE
        );
        fs.writeFileSync(full, html, 'utf8');
        sendJson(res, 200, { ok: true });
      });
      return;
    }

    // POST /js-parts — patch the `parts:` (multi-file scene map) property
    if (route === '/js-parts' && req.method === 'POST') {
      withJsonBody(req, res, ({ page, parts }) => {
        const full = pagePath(page);
        if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
        let html = fs.readFileSync(full, 'utf8');
        const indent  = detectIndent(html);
        const newLine = `${indent}parts: ${JSON.stringify(parts ?? {})}, // hs-parts`;
        html = setJsConfigLine(
          html,
          /^[ \t]*parts\s*:\s*\{[^}]*\}[^\n]*\n?/gm,
          newLine,
          ANIM_FLIPY_SCENES_CLOSE
        );
        fs.writeFileSync(full, html, 'utf8');
        sendJson(res, 200, { ok: true });
      });
      return;
    }

    // POST /js-partsDir — patch the `partsDir:` property
    if (route === '/js-partsDir' && req.method === 'POST') {
      withJsonBody(req, res, ({ page, partsDir }) => {
        const full = pagePath(page);
        if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
        let html = fs.readFileSync(full, 'utf8');
        const indent  = detectIndent(html);
        const escaped = (partsDir || '').replace(/'/g, "\\'");
        const newLine = `${indent}partsDir: '${escaped}',`;
        html = setJsConfigLine(
          html,
          /^[ \t]*partsDir\s*:\s*(['"])[^'"]*\1[^\n]*\n?/gm,
          newLine,
          ANIM_FLIPY_SCENES_CLOSE
        );
        fs.writeFileSync(full, html, 'utf8');
        sendJson(res, 200, { ok: true });
      });
      return;
    }

    // POST /js-zIndex — patch the `zIndex:` (player stacking order) property
    if (route === '/js-zIndex' && req.method === 'POST') {
      withJsonBody(req, res, ({ page, zIndex }) => {
        const full = pagePath(page);
        if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
        let html = fs.readFileSync(full, 'utf8');
        const indent  = detectIndent(html);
        const zi      = parseInt(zIndex, 10);
        const newLine = `${indent}zIndex: ${Number.isFinite(zi) ? zi : 5}, // hs-zi`;
        html = setJsConfigLine(
          html,
          /^[ \t]*zIndex\s*:\s*-?\d+[^\n]*\/\/\s*hs-zi[^\n]*\n?/gm,
          newLine,
          ANIM_FLIPY_SCENES_CLOSE
        );
        fs.writeFileSync(full, html, 'utf8');
        sendJson(res, 200, { ok: true });
      });
      return;
    }

    // POST /js-aaDilation — patch the `aaDilation:` (AA covariance dilation) property
    if (route === '/js-aaDilation' && req.method === 'POST') {
      withJsonBody(req, res, ({ page, aaDilation }) => {
        const full = pagePath(page);
        if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
        let html = fs.readFileSync(full, 'utf8');
        const indent  = detectIndent(html);
        const aa      = Math.round((parseFloat(aaDilation) || 0.15) * 10000) / 10000;
        const newLine = `${indent}aaDilation: ${aa}, // hs-aa`;
        html = setJsConfigLine(
          html,
          /^[ \t]*aaDilation\s*:\s*[0-9.]+[^\n]*\/\/\s*hs-aa[^\n]*\n?/gm,
          newLine,
          ANIM_CLOSE
        );
        fs.writeFileSync(full, html, 'utf8');
        sendJson(res, 200, { ok: true });
      });
      return;
    }

    // POST /js-scenes — patch the `scenes:` (per-marker scroll config) property
    if (route === '/js-scenes' && req.method === 'POST') {
      withJsonBody(req, res, ({ page, scenes }) => {
        const full = pagePath(page);
        if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
        let html = fs.readFileSync(full, 'utf8');
        const indent  = detectIndent(html);
        const newLine = `${indent}scenes: ${JSON.stringify(scenes ?? {})}, // hs-scenes`;
        html = setJsConfigLine(
          html,
          /^[ \t]*scenes\s*:[^\n]*\/\/\s*hs-scenes[^\n]*\n?/gm,
          newLine,
          ANIM_CLOSE
        );
        fs.writeFileSync(full, html, 'utf8');
        sendJson(res, 200, { ok: true });
      });
      return;
    }

    // POST /js-masks — patch the `masks:` (mask volume feather overrides) property
    if (route === '/js-masks' && req.method === 'POST') {
      withJsonBody(req, res, ({ page, masks }) => {
        const full = pagePath(page);
        if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
        let html = fs.readFileSync(full, 'utf8');
        const indent  = detectIndent(html);
        const newLine = `${indent}masks: ${JSON.stringify(masks ?? {})}, // hs-masks`;
        html = setJsConfigLine(
          html,
          /^[ \t]*masks\s*:[^\n]*\/\/\s*hs-masks[^\n]*\n?/gm,
          newLine,
          ANIM_CLOSE
        );
        fs.writeFileSync(full, html, 'utf8');
        sendJson(res, 200, { ok: true });
      });
      return;
    }

    // POST /js-clips — patch the `clips:` (asset clip file list) property
    if (route === '/js-clips' && req.method === 'POST') {
      withJsonBody(req, res, ({ page, clips }) => {
        const full = pagePath(page);
        if (!full || !fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
        let html = fs.readFileSync(full, 'utf8');
        const indent  = detectIndent(html);
        const newLine = `${indent}clips: ${JSON.stringify(clips ?? [])}, // hs-clips`;
        html = setJsConfigLine(
          html,
          /^[ \t]*clips\s*:[^\n]*\/\/\s*hs-clips[^\n]*\n?/gm,
          newLine,
          ANIM_CLOSE
        );
        fs.writeFileSync(full, html, 'utf8');
        sendJson(res, 200, { ok: true });
      });
      return;
    }

    if (typeof next === 'function') next();
    else { res.writeHead(404); res.end('Not found'); }
  };
}
