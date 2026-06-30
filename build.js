import { build, context } from 'esbuild';
import { readFileSync, writeFileSync } from 'fs';

const isWatch = process.argv.includes('--watch');

const common = {
  entryPoints: ['src/index.js'],
  bundle: true,
  minify: !isWatch,
  sourcemap: true,
};

// Stamp the editor overlay's title-bar version label from package.json —
// the single source of truth — so it can't drift out of sync with a
// manual edit. Matches either the unbuilt `__HS_VERSION__` placeholder or
// a previously-stamped version, so re-running this is idempotent.
function stampEditorVersion() {
  const { version } = JSON.parse(readFileSync('package.json', 'utf8'));
  const path = 'holosplat/editor.js';
  const src  = readFileSync(path, 'utf8');
  const updated = src.replace(
    /(<span id="__hs-ver"[^>]*>)v[^<]*(<\/span>)/,
    `$1v${version}$2`
  );
  if (updated !== src) writeFileSync(path, updated);
}
stampEditorVersion();

if (isWatch) {
  const ctx = await context({
    ...common,
    format: 'esm',
    outfile: 'dist/holosplat.esm.js',
  });
  await ctx.watch();
  console.log('HoloSplat: watching for changes...');
} else {
  await Promise.all([
    build({ ...common, format: 'esm',  outfile: 'dist/holosplat.esm.js' }),
    build({ ...common, format: 'iife', globalName: 'HoloSplat', outfile: 'dist/holosplat.iife.js' }),
  ]);
  console.log('HoloSplat: build complete → dist/');
}
