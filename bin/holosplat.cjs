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

  // Detect project type from package.json
  const projectPkg    = readPkgJson(cwd);
  const framework     = detectFramework(projectPkg);
  const hasOwnServer  = noServer ? false
    : withServer ? false
    : (framework !== 'none');

  // ── holosplat/ editor ──────────────────────────────────────────────────────
  const editorDir = path.join(cwd, 'holosplat');
  if (!fs.existsSync(editorDir)) fs.mkdirSync(editorDir, { recursive: true });

  const html = fs.readFileSync(path.join(pkg, 'holosplat', 'index.html'), 'utf8')
    .replace('../dist/holosplat.iife.js', './holosplat.iife.js');
  fs.writeFileSync(path.join(editorDir, 'index.html'), html);

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
  const giPath = path.join(cwd, '.gitignore');
  const giLine = 'holosplat/holosplat.iife.js';
  if (fs.existsSync(giPath)) {
    const gi = fs.readFileSync(giPath, 'utf8');
    if (!gi.includes(giLine))
      fs.appendFileSync(giPath, '\n# HoloSplat editor runtime\n' + giLine + '\n');
  }

  // ── Server setup ──────────────────────────────────────────────────────────
  if (!hasOwnServer) {
    const dst = path.join(cwd, 'server.py');
    if (!fs.existsSync(dst)) {
      fs.copyFileSync(path.join(pkg, 'server.py'), dst);
      console.log('  server.py                   ← python server.py to start');
    } else {
      console.log('  (server.py already exists — skipped)');
    }
  } else {
    writeServerSnippet(editorDir, framework);
    console.log('  holosplat/server-snippet.js ← add these routes to your server');
  }

  // ── CLAUDE.md ─────────────────────────────────────────────────────────────
  writeClaudeMd(cwd, framework, hasOwnServer);
  console.log('  CLAUDE.md                   ← HoloSplat section added/created');

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
    console.log('    1. Add routes from holosplat/server-snippet.js to your server');
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

  const serverDst = path.join(cwd, 'server.py');
  if (fs.existsSync(serverDst))
    fs.copyFileSync(path.join(pkg, 'server.py'), serverDst);

  const snippetDst = path.join(cwd, 'holosplat', 'server-snippet.js');
  if (fs.existsSync(snippetDst))
    writeServerSnippet(path.join(cwd, 'holosplat'), detectFramework(readPkgJson(cwd)));

  const version = require(path.join(pkg, 'package.json')).version;
  console.log(`\n  HoloSplat upgraded to v${version}\n`);

// ── help ──────────────────────────────────────────────────────────────────────
} else {
  const version = require(path.join(pkg, 'package.json')).version;
  console.log(`
  HoloSplat CLI  v${version}

  npx holosplat init               Set up the editor in this project
  npx holosplat init --with-server Also copy server.py (Python dev server)
  npx holosplat init --no-server   Skip server setup entirely
  npx holosplat upgrade            Refresh editor files after updating the package
`);
}

// ── helpers ───────────────────────────────────────────────────────────────────

function readPkgJson(dir) {
  try { return JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8')); }
  catch { return {}; }
}

function detectFramework(pkg) {
  const deps = { ...pkg.dependencies, ...pkg.devDependencies };
  if (deps.next)    return 'nextjs';
  if (deps.vite)    return 'vite';
  if (deps.express) return 'express';
  if (pkg.scripts?.dev || pkg.scripts?.start) return 'generic';
  return 'none';
}

function writeServerSnippet(dir, framework) {
  const snippets = {
    express: `// Express / Connect
import { createHsApiHandler } from 'holosplat/server';
app.use('/hs-api', createHsApiHandler());
`,
    vite: `// Vite (vite.config.js)
import { createHsApiHandler } from 'holosplat/server';
export default defineConfig({
  plugins: [{
    name: 'holosplat',
    configureServer(server) {
      server.middlewares.use('/hs-api', createHsApiHandler());
    },
  }],
});
`,
    nextjs: `// Next.js — pages router  (pages/api/hs-api/[...route].js)
import { createHsApiHandler } from 'holosplat/server';
const handler = createHsApiHandler();
export default function hsApi(req, res) {
  const sub = '/' + (req.query.route ?? []).join('/');
  const qs  = req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '';
  req.url   = sub + qs;
  handler(req, res);
}
export const config = { api: { bodyParser: false } };
`,
    generic: `// Generic Node.js (mount at /hs-api in your server)
import { createHsApiHandler } from 'holosplat/server';
const hsApi = createHsApiHandler();
// Express:    app.use('/hs-api', hsApi);
// Vite:       server.middlewares.use('/hs-api', hsApi);
// Raw http:   if (req.url.startsWith('/hs-api')) { req.url = req.url.slice(7); hsApi(req, res); }
`,
  };

  const body = `// HoloSplat API routes — add to your existing server
// Generated by: npx holosplat init
//
// The /holosplat editor needs GET /hs-api/ls, GET /hs-api/file, PUT /hs-api/file
// to read and write project files (hs-config.json, scene listings).

${snippets[framework] || snippets.generic}
// ── All frameworks ──────────────────────────────────────────────────────────
// Express / Connect:  app.use('/hs-api', createHsApiHandler())
// Vite:               server.middlewares.use('/hs-api', createHsApiHandler())
// Next.js pages:      see pages/api/hs-api/[...route].js above
`;
  fs.writeFileSync(path.join(dir, 'server-snippet.js'), body);
}

function writeClaudeMd(dir, framework, hasOwnServer) {
  const claudePath = path.join(dir, 'CLAUDE.md');
  const marker     = '<!-- holosplat -->';

  const serverSection = hasOwnServer
    ? serverInstructions(framework)
    : `### Dev server
Run \`python server.py\` — serves the project and mounts \`/hs-api\` automatically.
The editor is at \`http://localhost:8080/holosplat/\`.`;

  const section = `
${marker}
## HoloSplat

Installed as \`holosplat\` npm package. WebGPU Gaussian Splat viewer with scroll-driven animation.

### Import (ESM / bundler — no script tag needed)
\`\`\`js
import { Viewer, player, scrollScene } from 'holosplat';
\`\`\`

${serverSection}

### Single embed (auto-init via data attribute)
\`\`\`html
<div data-holosplat="/scenes/scene.spz"
     data-holosplat-anim="/scenes/anim.json"
     style="width:100%;height:500px"></div>
\`\`\`

### Scroll-driven scene (HTML structure)
\`\`\`html
<div class="hs-scene">
  <!-- sticky canvas -->
  <div class="hs-stage"
       data-holosplat="/scenes/scene.spz"
       data-holosplat-anim="/scenes/anim.json"></div>

  <!-- scroll track — each child maps to a Blender marker range -->
  <div class="hs-track">
    <div class="hs-act"  data-from="intro" data-to="next" style="height:200vh"></div>
    <div class="hs-hold" data-frame="next"                style="height:100vh"></div>
    <div class="hs-act"  data-from="next"  data-to="end"  style="height:300vh"></div>
  </div>
</div>
\`\`\`
- \`data-from\` / \`data-to\` / \`data-frame\` accept Blender marker names or frame numbers.
- Height (vh) controls how many scroll pixels the animation takes — taller = slower.
- Special act types: \`data-from="pingpong-start"\` (auto-loop) and \`data-from="freecamera-start"\` (free orbit).

### hs-config.json (managed by the /holosplat editor, do not hand-edit)
The editor writes this file. At runtime, a loader reads it and populates the
\`.hs-track\` div with the correct \`data-from\`/\`data-to\`/\`style="height"\` attributes.
\`\`\`json
{
  "version": 1,
  "scene": "scenes/scene.spz",
  "animation": "scenes/anim.json",
  "acts": [
    { "id": "intro", "type": "act",      "from": "intro", "to": "marker2", "height": 200 },
    { "id": "loop",  "type": "pingpong", "from": "pingpong-start", "to": "pingpong-end", "height": 150 }
  ]
}
\`\`\`

### Rules
- Never import from \`holosplat/server\` in browser/client code — it is Node.js only.
- Never deploy \`holosplat/\` — it is in \`.vercelignore\` and must stay local-only.
- \`holosplat/holosplat.iife.js\` is gitignored — it is a build copy for the editor preview.
- Scene files (\`.spz\`, \`.ply\`, \`.splat\`) go in \`scenes/\` or \`public/scenes/\`.
<!-- /holosplat -->
`;

  if (fs.existsSync(claudePath)) {
    let existing = fs.readFileSync(claudePath, 'utf8');
    // Replace existing HoloSplat block if present, otherwise append
    const re = new RegExp(`\n${marker}[\\s\\S]*?<!-- /holosplat -->`, 'g');
    if (re.test(existing)) {
      existing = existing.replace(re, section.trimEnd());
      fs.writeFileSync(claudePath, existing);
    } else {
      fs.appendFileSync(claudePath, section);
    }
  } else {
    fs.writeFileSync(claudePath, `# Project\n\nAdd project-specific notes here.\n${section}`);
  }
}

function serverInstructions(framework) {
  const examples = {
    vite: `### Server API (Vite — vite.config.js)
\`\`\`js
import { createHsApiHandler } from 'holosplat/server';
export default defineConfig({
  plugins: [{
    name: 'holosplat',
    configureServer(server) {
      server.middlewares.use('/hs-api', createHsApiHandler());
    },
  }],
});
\`\`\``,
    express: `### Server API (Express)
\`\`\`js
import { createHsApiHandler } from 'holosplat/server';
app.use('/hs-api', createHsApiHandler());
\`\`\``,
    nextjs: `### Server API (Next.js — pages/api/hs-api/[...route].js)
\`\`\`js
import { createHsApiHandler } from 'holosplat/server';
const handler = createHsApiHandler();
export default function hsApi(req, res) {
  const sub = '/' + (req.query.route ?? []).join('/');
  const qs  = req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '';
  req.url   = sub + qs;
  handler(req, res);
}
export const config = { api: { bodyParser: false } };
\`\`\``,
    generic: `### Server API
\`\`\`js
import { createHsApiHandler } from 'holosplat/server';
// Mount at /hs-api in your server:
app.use('/hs-api', createHsApiHandler());  // Express/Connect
// or: server.middlewares.use('/hs-api', createHsApiHandler());  // Vite
\`\`\``,
  };
  return (examples[framework] || examples.generic) +
    '\nThis is required for the `/holosplat` editor to read and write files. **Only enable in development.**';
}
