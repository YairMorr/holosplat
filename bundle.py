"""
HoloSplat bundler.

Reads source files in dependency order, strips import/export statements,
and produces IIFE + ESM builds in dist/.

Usage:
    python bundle.py           # development build (readable)
    python bundle.py --minify  # production build (minified + gzip stats)

Minification requires:  pip install rjsmin
"""
import re, os, sys, gzip

os.makedirs('dist', exist_ok=True)

MINIFY = '--minify' in sys.argv
VERSION = '0.1.0'

# ── Source files in dependency order (no circular imports) ────────────────────
FILES = [
    'src/shaders.js',
    'src/camera.js',
    'src/sorter.js',
    'src/loaders/fetch-utils.js',
    'src/loaders/splat-loader.js',
    'src/loaders/ply-loader.js',
    'src/loaders/spz-loader.js',
    'src/animation.js',
    'src/compress.js',
    'src/renderer.js',
    'src/viewer.js',
    'src/player.js',
    'src/index.js',
]

EXPORTS = 'create, player, Viewer, Animation, loadAnimation, compressToSpz, encodeSpz, parseSplat, parsePly'
BANNER  = f'/*! HoloSplat v{VERSION} – WebGPU Gaussian Splat viewer | MIT */'

# ── Source transforms ─────────────────────────────────────────────────────────

def strip_imports(src):
    """Remove internal relative import statements."""
    return re.sub(
        r"^import\s+\{[^}]+\}\s+from\s+['\"][^'\"]+['\"];\s*$",
        '', src, flags=re.MULTILINE)

def strip_exports(src):
    """Remove export keywords from declarations; drop bare export{} lines."""
    src = re.sub(
        r"\bexport\s+(default\s+)?(async\s+)?(function|class|const|let|var)\b",
        lambda m: (m.group(2) or '') + m.group(3), src)
    src = re.sub(r"^export\s+\{[^}]+\};\s*$", '', src, flags=re.MULTILINE)
    return src

# ── Minifier ──────────────────────────────────────────────────────────────────

def minify(js):
    """Minify JS. Uses rjsmin if available, otherwise strips blank lines."""
    try:
        import rjsmin
        return rjsmin.jsmin(js, keep_bang_comments=True)
    except ImportError:
        print('  [warn] rjsmin not found — install with: pip install rjsmin')
        print('         Falling back to whitespace-only stripping.')
        # Remove single-line // comments and collapse blank lines
        js = re.sub(r'[ \t]*//[^\n]*', '', js)
        js = re.sub(r'\n{2,}', '\n', js)
        return js.strip()

# ── Assemble source body ──────────────────────────────────────────────────────

parts = []
for path in FILES:
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()
    code = strip_imports(code)
    code = strip_exports(code)
    if not MINIFY:
        parts.append(f'// ── {path} ────────────────────────────\n' + code)
    else:
        parts.append(code)

body = '\n'.join(parts)

# ── Build outputs ─────────────────────────────────────────────────────────────

def write(path, content):
    if MINIFY:
        content = minify(content)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    raw_kb  = len(content.encode()) / 1024
    gz_kb   = len(gzip.compress(content.encode(), compresslevel=9)) / 1024
    tag     = ' [minified]' if MINIFY else ''
    print(f'  {path}{tag}  {raw_kb:.1f} kB  ({gz_kb:.1f} kB gzip)')

# IIFE — for <script src> / Webflow
iife = (
    BANNER + '\n' +
    f"var HoloSplat=(function(){{'use strict';\n{body}\nreturn{{{EXPORTS}}};}})();\n"
    if MINIFY else
    f"""/**
 * HoloSplat v{VERSION} – WebGPU Gaussian Splat viewer | MIT
 */
var HoloSplat = (function () {{
  'use strict';

{body}

  return {{ {EXPORTS} }};
}})();
"""
)
write('dist/holosplat.iife.js', iife)

# ESM — for bundlers / import()
esm = (
    BANNER + '\n' + body + f'\nexport{{{EXPORTS}}};\n'
    if MINIFY else
    f"""/**
 * HoloSplat v{VERSION} – WebGPU Gaussian Splat viewer | MIT
 */
{body}

export {{ {EXPORTS} }};
"""
)
write('dist/holosplat.esm.js', esm)

print('Build complete.' + (' Run without --minify for a readable dev build.' if MINIFY else ''))
