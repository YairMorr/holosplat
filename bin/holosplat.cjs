#!/usr/bin/env node
'use strict';

const fs   = require('fs');
const path = require('path');

const pkg = path.join(__dirname, '..');   // root of the installed package
const cwd = process.cwd();               // new project root

const cmd = process.argv[2];

// ── init ──────────────────────────────────────────────────────────────────────
if (cmd === 'init') {
  const editorDir = path.join(cwd, 'holosplat');
  const scenesDir = path.join(cwd, 'scenes');

  // holosplat/ — editor UI
  if (!fs.existsSync(editorDir)) fs.mkdirSync(editorDir, { recursive: true });

  // Copy index.html, rewriting the script src to be self-contained
  const html = fs.readFileSync(path.join(pkg, 'holosplat', 'index.html'), 'utf8')
    .replace('../dist/holosplat.iife.js', './holosplat.iife.js');
  fs.writeFileSync(path.join(editorDir, 'index.html'), html);

  // Copy the IIFE build alongside the editor so it's fully self-contained
  fs.copyFileSync(
    path.join(pkg, 'dist', 'holosplat.iife.js'),
    path.join(editorDir, 'holosplat.iife.js')
  );

  // server.py — only copy if not already present
  const serverDst = path.join(cwd, 'server.py');
  if (!fs.existsSync(serverDst)) {
    fs.copyFileSync(path.join(pkg, 'server.py'), serverDst);
  } else {
    console.log('  (server.py already exists — skipped)');
  }

  // scenes/ — create with a .gitkeep so git tracks the empty folder
  if (!fs.existsSync(scenesDir)) {
    fs.mkdirSync(scenesDir, { recursive: true });
    fs.writeFileSync(path.join(scenesDir, '.gitkeep'), '');
  }

  // Suggest .gitignore additions
  const gitignorePath = path.join(cwd, '.gitignore');
  const addLines = ['holosplat/holosplat.iife.js'];
  if (fs.existsSync(gitignorePath)) {
    const existing = fs.readFileSync(gitignorePath, 'utf8');
    const missing  = addLines.filter(l => !existing.includes(l));
    if (missing.length) {
      fs.appendFileSync(gitignorePath, '\n# HoloSplat editor build copy\n' + missing.join('\n') + '\n');
      console.log(`  (added ${missing.join(', ')} to .gitignore)`);
    }
  }

  console.log('\n  HoloSplat initialised\n');
  console.log('  holosplat/index.html      ← art direction editor');
  console.log('  holosplat/holosplat.iife.js ← editor runtime (gitignored)');
  console.log('  server.py                 ← local dev server');
  console.log('  scenes/                   ← drop .spz / .ply / .splat files here');
  console.log('\n  Start editing:');
  console.log('    python server.py');
  console.log('    open http://localhost:8080/holosplat/\n');

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
  // Always overwrite server.py on upgrade
  fs.copyFileSync(path.join(pkg, 'server.py'), path.join(cwd, 'server.py'));

  console.log('\n  HoloSplat upgraded to', require(path.join(pkg, 'package.json')).version, '\n');

// ── help ──────────────────────────────────────────────────────────────────────
} else {
  console.log(`
  HoloSplat CLI

  npx holosplat init      Set up the editor and dev server in this project
  npx holosplat upgrade   Refresh editor files after updating the package
`);
}
