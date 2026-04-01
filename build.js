import { build, context } from 'esbuild';

const isWatch = process.argv.includes('--watch');

const common = {
  entryPoints: ['src/index.js'],
  bundle: true,
  minify: !isWatch,
  sourcemap: true,
};

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
