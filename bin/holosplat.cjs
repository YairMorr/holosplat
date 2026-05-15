#!/usr/bin/env node
'use strict';

const fs   = require('fs');
const path = require('path');

const pkg  = path.join(__dirname, '..');  // root of the installed package
const cwd  = process.cwd();              // target project root
const args = process.argv.slice(2);
const cmd  = args[0];
const flags = new Set(args.slice(1));

// ── init ──────────────────────────────────────────────────────────────────────
if (cmd === 'init') {
  const withServer = flags.has('--with-server');
  const noServer   = flags.has('--no-server');

  // Detect whether the project already has a Node.js server
  // (has package.json with a dev/start script → assume it has one)
  let hasOwnServer = noServer;
  if (!withServer && !noServer && fs.existsSync(path.join(cwd, 'package.json'))) {
    try {
      const p = JSON.parse(fs.readFileSync(path.join(cwd, 'package.json'), 'utf8'));
      hasOwnServer = !!(p.scripts?.dev || p.scripts?.start);
    } catch {}
  }

  // ── holosplat/ editor ──────────────────────────────────────────────────────
  const editorDir = path.join(cwd, 'holosplat');
  if (!fs.existsSync(editorDir)) fs.mkdirSync(editorDir, { recursive: true });

  // index.html — rewrite script src to be self-contained
  const html = fs.readFileSync(path.join(pkg, 'holosplat', 'index.html'), 'utf8')
    .replace('../dist/holosplat.iife.js', './holosplat.iife.js');
  fs.writeFileSync(path.join(editorDir, 'index.html'), html);

  // Copy the IIFE build so the editor works without a bundler
  fs.copyFileSync(
    path.join(pkg, 'dist', 'holosplat.iife.js'),
    path.join(editorDir, 'holosplat.iife.js')
  );

  // ── scenes/ ───────────────────────────────────────────────────────────────
  const scenesDir = path.join(cwd, 'scenes');
  if (!fs.existsSync(scenesDir)) {
    fs.mkdirSync(scenesDir, { recursive: true });
    fs.writeFileSync(path.join(scenesDir, '.gitkeep'), '');
  }

  // ── .gitignore ────────────────────────────────────────────────────────────
  const giPath  = path.join(cwd, '.gitignore');
  const giLines = ['holosplat/holosplat.iife.js'];
  if (fs.existsSync(giPath)) {
    const existing = fs.readFileSync(giPath, 'utf8');
    const missing  = giLines.filter(l => !existing.includes(l));
    if (missing.length)
      fs.appendFileSync(giPath, '\n# HoloSplat editor runtime (copied from node_modules)\n' + missing.join('\n') + '\n');
  }

  // ── Server setup ──────────────────────────────────────────────────────────
  if (!hasOwnServer) {
    // No existing server detected — copy server.py
    const dst = path.join(cwd, 'server.py');
    if (!fs.existsSync(dst)) {
      fs.copyFileSync(path.join(pkg, 'server.py'), dst);
      console.log('  server.py                   ← local dev server (python server.py)');
    } else {
      console.log('  (server.py already exists — skipped)');
    }
  } else {
    // Existing server — write a ready-to-paste snippet
    writeServerSnippet(editorDir);
    console.log('  holosplat/server-snippet.js ← paste these routes into your server');
  }

  // ── Done ──────────────────────────────────────────────────────────────────
  console.log('\n  HoloSplat initialised\n');
  console.log('  holosplat/index.html        ← art direction editor');
  console.log('  holosplat/holosplat.iife.js ← editor runtime (gitignored)');
  console.log('  scenes/                     ← drop .spz / .ply / .splat files here');

  if (!hasOwnServer) {
    console.log('\n  Start editing:');
    console.log('    python server.py');
    console.log('    open http://localhost:8080/holosplat/\n');
  } else {
    console.log('\n  Next steps:');
    console.log('    1. Add the routes from holosplat/server-snippet.js to your server');
    console.log('    2. Start your dev server');
    console.log('    3. Open http://localhost:<port>/holosplat/\n');
  }

// ── upgrade ───────────────────────────────────────────────────────────────────
} else if (cmd === 'upgrade') {
  const editorDir = path.join(cwd, 'holosplat');
  if (!fs.existsSync(editorDir)) {
    console.error('\n  Run `npx holosplat init` first.\n');
    process.exit(1);
  }

  const html = fs.readFileSync(path.join(pkg, 'holosplat', 'index.html'), 'utf8')
    .replace('../dist/holosplat.iife.js', './holosplat.iife.js');
  fs.writeFileSync(path.join(editorDir, 'index.html'), html);
  fs.copyFileSync(
    path.join(pkg, 'dist', 'holosplat.iife.js'),
    path.join(editorDir, 'holosplat.iife.js')
  );

  // Update server.py if it exists
  const serverDst = path.join(cwd, 'server.py');
  if (fs.existsSync(serverDst)) {
    fs.copyFileSync(path.join(pkg, 'server.py'), serverDst);
    console.log('  Updated server.py');
  }

  // Update snippet if it exists
  const snippetDst = path.join(cwd, 'holosplat', 'server-snippet.js');
  if (fs.existsSync(snippetDst)) writeServerSnippet(path.join(cwd, 'holosplat'));

  const version = require(path.join(pkg, 'package.json')).version;
  console.log(`\n  HoloSplat upgraded to v${version}\n`);

// ── help ──────────────────────────────────────────────────────────────────────
} else {
  console.log(`
  HoloSplat CLI  v${require(path.join(pkg, 'package.json')).version}

  npx holosplat init               Set up the editor in this project
                                   Auto-detects whether a server exists.
  npx holosplat init --with-server Also copy server.py (Python dev server)
  npx holosplat init --no-server   Skip server setup entirely

  npx holosplat upgrade            Refresh editor files after updating the package
`);
}

// ── helpers ───────────────────────────────────────────────────────────────────
function writeServerSnippet(dir) {
  const snippet = `// HoloSplat API routes — add to your existing server
// Generated by: npx holosplat init
//
// The /holosplat editor needs these three routes to read and write
// project files (hs-config.json, scene listings, etc.)
//
// ─────────────────────────────────────────────────────────────────
// Express / Connect
// ─────────────────────────────────────────────────────────────────
import { createHsApiHandler } from 'holosplat/server';
app.use('/hs-api', createHsApiHandler());

// ─────────────────────────────────────────────────────────────────
// Vite  (vite.config.js)
// ─────────────────────────────────────────────────────────────────
import { createHsApiHandler } from 'holosplat/server';
export default defineConfig({
  plugins: [{
    name: 'holosplat',
    configureServer(server) {
      server.middlewares.use('/hs-api', createHsApiHandler());
    },
  }],
});

// ─────────────────────────────────────────────────────────────────
// Next.js — pages router  (pages/api/hs-api/[...route].js)
// ─────────────────────────────────────────────────────────────────
import { createHsApiHandler } from 'holosplat/server';
const handler = createHsApiHandler();
export default function hsApi(req, res) {
  // Re-join the dynamic route segments Next.js parsed from [...route]
  const sub = '/' + (req.query.route ?? []).join('/');
  const qs  = req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '';
  req.url   = sub + qs;
  handler(req, res);
}
export const config = { api: { bodyParser: false } };
`;
  fs.writeFileSync(path.join(dir, 'server-snippet.js'), snippet);
}
