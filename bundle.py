"""
Simple bundler for HoloSplat.
Reads source files in dependency order, strips import/export statements,
and produces IIFE + ESM builds in dist/.
"""
import re, os

os.makedirs('dist', exist_ok=True)

# Dependency order — no circular imports
FILES = [
    'src/shaders.js',
    'src/camera.js',
    'src/sorter.js',
    'src/loaders/fetch-utils.js',
    'src/loaders/splat-loader.js',
    'src/loaders/ply-loader.js',
    'src/loaders/spz-loader.js',
    'src/renderer.js',
    'src/viewer.js',
    'src/index.js',
]

def strip_imports(src):
    """Remove import statements (all are internal relative imports)."""
    src = re.sub(r"^import\s+\{[^}]+\}\s+from\s+['\"][^'\"]+['\"];\s*$", '',
                 src, flags=re.MULTILINE)
    return src

def strip_exports(src):
    """Remove export keywords from declarations (export class → class, etc.)."""
    src = re.sub(r"\bexport\s+(default\s+)?(async\s+)?(function|class|const|let|var)\b",
                 lambda m: (m.group(2) or '') + m.group(3), src)
    # Remove bare 'export { ... }' lines
    src = re.sub(r"^export\s+\{[^}]+\};\s*$", '', src, flags=re.MULTILINE)
    return src

# Gather all source
parts = []
for path in FILES:
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()
    code = strip_imports(code)
    code = strip_exports(code)
    parts.append(f'// ── {path} ────────────────────────────\n' + code)

body = '\n'.join(parts)

# ── IIFE build ─────────────────────────────────────────────────────────────
iife = f"""/**
 * HoloSplat v0.1.0 – WebGPU Gaussian Splat viewer
 * https://github.com/your-org/holosplat
 * License: MIT
 */
var HoloSplat = (function () {{
  'use strict';

{body}

  return {{ create, Viewer }};
}})();
"""
with open('dist/holosplat.iife.js', 'w', encoding='utf-8') as f:
    f.write(iife)
print('  dist/holosplat.iife.js')

# ── ESM build ──────────────────────────────────────────────────────────────
esm = f"""/**
 * HoloSplat v0.1.0 – WebGPU Gaussian Splat viewer
 * https://github.com/your-org/holosplat
 * License: MIT
 */
{body}

export {{ create, Viewer }};
"""
with open('dist/holosplat.esm.js', 'w', encoding='utf-8') as f:
    f.write(esm)
print('  dist/holosplat.esm.js')

print('Build complete.')
