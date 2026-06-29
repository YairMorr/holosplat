#!/usr/bin/env python3
"""
HoloSplat local dev server.

- Serves the entire project with CORS headers so WebGPU can load scene files.
- Exposes a small /hs-api for the /holosplat editor to read and write files.
- Put .spz / .splat / .ply scene files in the scenes/ folder.

Usage:
    python server.py [port]       (default port: 8080)

Then open:
    http://localhost:8080/examples/viewer.html
    http://localhost:8080/holosplat/

Scene files are accessible at:
    http://localhost:8080/scenes/your-file.spz
"""

import http.server
import json
import os
import re
import sys
import threading
import urllib.parse

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

ROOT      = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.join(ROOT, 'scenes')
os.makedirs(SCENE_DIR, exist_ok=True)

SCENE_EXTS = {'.spz', '.splat', '.ply', '.spzv'}

# Protects all read-modify-write operations on HTML/JS source files.
# ThreadingHTTPServer uses one thread per request; without a lock, two
# concurrent saves can interleave: one thread truncates the file while
# another reads it, receiving 0 bytes and writing them back.
_file_lock = threading.Lock()


def _set_js_config_line(html, remove_pattern, new_line, insert_patterns):
    """Replace a `player({...})` config property with `new_line`.

    Removes every existing line matching `remove_pattern` first — this heals
    duplicate lines left behind by earlier saves (previously only the first
    match was ever updated, so stale duplicates accumulated across syncs) —
    then inserts `new_line` once at the first matching location in
    `insert_patterns` (regexes for "insert after this line", or starting with
    `(\\n` to mean "insert before this match", used for the closing `});`).
    """
    updated = re.sub(remove_pattern, '', html, flags=re.MULTILINE)
    for pat in insert_patterns:
        m = re.search(pat, updated)
        if m:
            ins = m.start() + 1 if pat.startswith(r'(\n') else m.end()
            return updated[:ins] + new_line + '\n' + updated[ins:]
    return updated


class Handler(http.server.SimpleHTTPRequestHandler):

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin',  '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, PUT, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Range, Content-Type')
        self.send_header('Access-Control-Expose-Headers','Content-Length, Content-Range')
        # Scene files (under /scenes/) only change via a manual re-export, so
        # let the browser skip the network entirely on repeat loads instead
        # of paying a revalidation round trip per file — that round trip,
        # multiplied across a dozen+ parts plus color variants, is what made
        # first loads visibly slower than cached repeat loads. Everything
        # else (HTML/JS/editor API) stays no-cache so live edits show up
        # immediately.
        if urllib.parse.urlparse(self.path).path.startswith('/scenes/'):
            self.send_header('Cache-Control', 'public, max-age=31536000, immutable')
        else:
            self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith('/hs-api/'):
            params = dict(urllib.parse.parse_qsl(parsed.query))
            if   parsed.path == '/hs-api/file': self._api_read(params.get('path', ''))
            elif parsed.path == '/hs-api/ls':   self._api_ls()
            else: self._json(404, {'error': 'not found'})
        else:
            super().do_GET()

    def do_PUT(self):
        parsed = urllib.parse.urlparse(self.path)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        if parsed.path == '/hs-api/file':
            self._api_write(params.get('path', ''))
        else:
            self._json(404, {'error': 'not found'})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if   parsed.path == '/hs-api/html-attr':   self._api_html_attr()
        elif parsed.path == '/hs-api/js-scenes':   self._api_js_scenes()
        elif parsed.path == '/hs-api/js-masks':    self._api_js_masks()
        elif parsed.path == '/hs-api/js-sh':       self._api_js_sh()
        elif parsed.path == '/hs-api/js-anim':     self._api_js_anim()
        elif parsed.path == '/hs-api/js-parts':    self._api_js_parts()
        elif parsed.path == '/hs-api/js-partsDir': self._api_js_partsDir()
        elif parsed.path == '/hs-api/js-zIndex':    self._api_js_zIndex()
        elif parsed.path == '/hs-api/js-aaDilation': self._api_js_aaDilation()
        elif parsed.path == '/hs-api/js-clips':     self._api_js_clips()
        else: self._json(404, {'error': 'not found'})

    # ── Helpers ────────────────────────────────────────────────────────────

    def _safe(self, rel):
        """Resolve rel to an absolute path that stays inside ROOT, or None."""
        if not rel:
            return None
        full = os.path.realpath(os.path.join(ROOT, rel))
        try:
            if os.path.commonpath([ROOT, full]) != ROOT:
                return None
        except ValueError:
            return None          # different drives on Windows
        if full == ROOT:
            return None          # can't read/write the project root itself
        return full

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _api_read(self, rel):
        path = self._safe(rel)
        if not path or not os.path.isfile(path):
            self._json(404, {'error': 'not found'})
            return
        with open(path, 'rb') as f:
            body = f.read()
        ext = os.path.splitext(rel)[1].lower()
        ct = {
            '.json': 'application/json',
            '.html': 'text/html; charset=utf-8',
            '.js':   'text/javascript; charset=utf-8',
            '.css':  'text/css; charset=utf-8',
        }.get(ext, 'text/plain; charset=utf-8')
        self.send_response(200)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _api_write(self, rel):
        path = self._safe(rel)
        if not path:
            self._json(403, {'error': 'forbidden'})
            return
        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with _file_lock:
            with open(path, 'wb') as f:
                f.write(body)
        self._json(200, {'ok': True})

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(length).decode('utf-8')

    def _api_html_attr(self):
        try:
            body = json.loads(self._read_body())
        except Exception:
            self._json(400, {'error': 'invalid JSON'}); return
        page  = body.get('page', '')
        el_id = body.get('id', '')
        attrs = body.get('attrs', {})
        if not page or not el_id or not attrs:
            self._json(400, {'error': 'missing fields'}); return
        full = self._safe(page.lstrip('/').split('?')[0])
        if not full or not os.path.isfile(full):
            self._json(404, {'error': 'not found'}); return
        with _file_lock:
            with open(full, encoding='utf-8') as f:
                html = f.read()
            id_pat = re.escape(el_id)
            tag_rx = re.compile(rf'(<[a-zA-Z][^>]*?\s(?:id="{id_pat}")[^>]*?)(?=\s*>)', re.DOTALL)
            m = tag_rx.search(html)
            if not m:
                self._json(404, {'error': f'element #{el_id} not found'}); return
            tag = m.group(1)
            for name, value in attrs.items():
                n = re.escape(name)
                tag = re.sub(rf'\s+{n}(?:="[^"]*"|=\'[^\']*\')?', '', tag)
                if value is not None:
                    escaped = str(value).replace('"', '&quot;')
                    tag += f' {name}="{escaped}"'
            html = html[:m.start()] + tag + html[m.end():]
            with open(full, 'w', encoding='utf-8') as f:
                f.write(html)
        self._json(200, {'ok': True})

    def _api_js_sh(self):
        try:
            body = json.loads(self._read_body())
        except Exception:
            self._json(400, {'error': 'invalid JSON'}); return
        page = body.get('page', '')
        sh   = int(body.get('sh', 0))
        full = self._safe(page.lstrip('/').split('?')[0])
        if not full or not os.path.isfile(full):
            self._json(404, {'error': 'not found'}); return
        with _file_lock:
            with open(full, encoding='utf-8') as f:
                html = f.read()
            m_indent = re.search(r'^([ \t]*)animation\s*:', html, re.MULTILINE)
            indent = m_indent.group(1) if m_indent else '      '
            new_line = f'{indent}sh: {sh}, // hs-sh'
            updated = _set_js_config_line(
                html,
                r'^[ \t]*sh\s*:\s*\d+[^\n]*//\s*hs-sh[^\n]*\n?',
                new_line,
                [
                    r'([ \t]*animation\s*:[^\n]*\n)',
                    r'(\n[ \t]*\}\s*\)\s*;)',
                ]
            )
            with open(full, 'w', encoding='utf-8') as f:
                f.write(updated)
        self._json(200, {'ok': True})

    def _api_js_anim(self):
        try:
            body = json.loads(self._read_body())
        except Exception:
            self._json(400, {'error': 'invalid JSON'}); return
        page = body.get('page', '')
        url  = body.get('url', '')
        full = self._safe(page.lstrip('/').split('?')[0])
        if not full or not os.path.isfile(full):
            self._json(404, {'error': 'not found'}); return
        with _file_lock:
            with open(full, encoding='utf-8') as f:
                html = f.read()
            m_indent = re.search(r'^([ \t]*)animation\s*:', html, re.MULTILINE)
            indent = m_indent.group(1) if m_indent else '      '
            escaped = url.replace("'", "\\'")
            new_line = f"{indent}animation: '{escaped}',"
            updated = _set_js_config_line(
                html,
                r"^[ \t]*animation\s*:\s*(['\"])[^'\"]*\1[^\n]*\n?",
                new_line,
                [
                    r'([ \t]*flipY\s*:[^\n]*\n)',
                    r'([ \t]*scenes\s*:[^\n]*\n)',
                    r'(\n[ \t]*\}\s*\)\s*;)',
                ]
            )
            with open(full, 'w', encoding='utf-8') as f:
                f.write(updated)
        self._json(200, {'ok': True})

    def _api_js_parts(self):
        try:
            body = json.loads(self._read_body())
        except Exception:
            self._json(400, {'error': 'invalid JSON'}); return
        page  = body.get('page', '')
        parts = body.get('parts', {})
        full  = self._safe(page.lstrip('/').split('?')[0])
        if not full or not os.path.isfile(full):
            self._json(404, {'error': 'not found'}); return
        with _file_lock:
            with open(full, encoding='utf-8') as f:
                html = f.read()
            m_indent = re.search(r'^([ \t]*)animation\s*:', html, re.MULTILINE)
            indent = m_indent.group(1) if m_indent else '      '
            json_str = json.dumps(parts, ensure_ascii=False)
            new_line = f'{indent}parts: {json_str}, // hs-parts'
            # Replace existing parts block (single or multi-line)
            updated = _set_js_config_line(
                html,
                r'^[ \t]*parts\s*:\s*\{[^}]*\}[^\n]*\n?',
                new_line,
                [
                    r'([ \t]*animation\s*:[^\n]*\n)',
                    r'([ \t]*flipY\s*:[^\n]*\n)',
                    r'([ \t]*scenes\s*:[^\n]*\n)',
                    r'(\n[ \t]*\}\s*\)\s*;)',
                ]
            )
            with open(full, 'w', encoding='utf-8') as f:
                f.write(updated)
        self._json(200, {'ok': True})

    def _api_js_partsDir(self):
        try:
            body = json.loads(self._read_body())
        except Exception:
            self._json(400, {'error': 'invalid JSON'}); return
        page     = body.get('page', '')
        partsDir = body.get('partsDir', '')
        full = self._safe(page.lstrip('/').split('?')[0])
        if not full or not os.path.isfile(full):
            self._json(404, {'error': 'not found'}); return
        with _file_lock:
            with open(full, encoding='utf-8') as f:
                html = f.read()
            m_indent = re.search(r'^([ \t]*)animation\s*:', html, re.MULTILINE)
            indent = m_indent.group(1) if m_indent else '      '
            escaped = partsDir.replace("'", "\\'")
            new_line = f"{indent}partsDir: '{escaped}',"
            updated = _set_js_config_line(
                html,
                r"^[ \t]*partsDir\s*:\s*(['\"])[^'\"]*\1[^\n]*\n?",
                new_line,
                [
                    r'([ \t]*animation\s*:[^\n]*\n)',
                    r'([ \t]*flipY\s*:[^\n]*\n)',
                    r'([ \t]*scenes\s*:[^\n]*\n)',
                    r'(\n[ \t]*\}\s*\)\s*;)',
                ]
            )
            with open(full, 'w', encoding='utf-8') as f:
                f.write(updated)
        self._json(200, {'ok': True})

    def _api_js_zIndex(self):
        try:
            body = json.loads(self._read_body())
        except Exception:
            self._json(400, {'error': 'invalid JSON'}); return
        page    = body.get('page', '')
        zi      = int(body.get('zIndex', 5))
        full = self._safe(page.lstrip('/').split('?')[0])
        if not full or not os.path.isfile(full):
            self._json(404, {'error': 'not found'}); return
        with _file_lock:
            with open(full, encoding='utf-8') as f:
                html = f.read()
            m_indent = re.search(r'^([ \t]*)animation\s*:', html, re.MULTILINE)
            indent = m_indent.group(1) if m_indent else '      '
            new_line = f'{indent}zIndex: {zi}, // hs-zi'
            updated = _set_js_config_line(
                html,
                r'^[ \t]*zIndex\s*:\s*-?\d+[^\n]*//\s*hs-zi[^\n]*\n?',
                new_line,
                [
                    r'([ \t]*animation\s*:[^\n]*\n)',
                    r'([ \t]*flipY\s*:[^\n]*\n)',
                    r'([ \t]*scenes\s*:[^\n]*\n)',
                    r'(\n[ \t]*\}\s*\)\s*;)',
                ]
            )
            with open(full, 'w', encoding='utf-8') as f:
                f.write(updated)
        self._json(200, {'ok': True})

    def _api_js_aaDilation(self):
        try:
            body = json.loads(self._read_body())
        except Exception:
            self._json(400, {'error': 'invalid JSON'}); return
        page = body.get('page', '')
        aa   = round(float(body.get('aaDilation', 0.15)), 4)
        full = self._safe(page.lstrip('/').split('?')[0])
        if not full or not os.path.isfile(full):
            self._json(404, {'error': 'not found'}); return
        with _file_lock:
            with open(full, encoding='utf-8') as f:
                html = f.read()
            m_indent = re.search(r'^([ \t]*)animation\s*:', html, re.MULTILINE)
            indent = m_indent.group(1) if m_indent else '      '
            new_line = f'{indent}aaDilation: {aa}, // hs-aa'
            updated = _set_js_config_line(
                html,
                r'^[ \t]*aaDilation\s*:\s*[0-9.]+[^\n]*//\s*hs-aa[^\n]*\n?',
                new_line,
                [
                    r'([ \t]*animation\s*:[^\n]*\n)',
                    r'(\n[ \t]*\}\s*\)\s*;)',
                ]
            )
            with open(full, 'w', encoding='utf-8') as f:
                f.write(updated)
        self._json(200, {'ok': True})

    def _api_js_scenes(self):
        try:
            body = json.loads(self._read_body())
        except Exception:
            self._json(400, {'error': 'invalid JSON'}); return
        page   = body.get('page', '')
        scenes = body.get('scenes', {})
        full = self._safe(page.lstrip('/').split('?')[0])
        if not full or not os.path.isfile(full):
            self._json(404, {'error': 'not found'}); return
        scenes_str = json.dumps(scenes, separators=(',', ':'))
        with _file_lock:
            with open(full, encoding='utf-8') as f:
                html = f.read()
            # Detect indentation from animation: line (or default 6 spaces)
            m_indent = re.search(r'^([ \t]*)animation\s*:', html, re.MULTILINE)
            indent = m_indent.group(1) if m_indent else '      '
            new_line = f'{indent}scenes: {scenes_str}, // hs-scenes'
            # Replace existing hs-scenes sentinel line
            updated = _set_js_config_line(
                html,
                r'^[ \t]*scenes\s*:[^\n]*//\s*hs-scenes[^\n]*\n?',
                new_line,
                [
                    r'([ \t]*animation\s*:[^\n]*\n)',
                    r'(\n[ \t]*\}\s*\)\s*;)',
                ]
            )
            with open(full, 'w', encoding='utf-8') as f:
                f.write(updated)
        self._json(200, {'ok': True})

    def _api_js_masks(self):
        try:
            body = json.loads(self._read_body())
        except Exception:
            self._json(400, {'error': 'invalid JSON'}); return
        page  = body.get('page', '')
        masks = body.get('masks', {})
        full = self._safe(page.lstrip('/').split('?')[0])
        if not full or not os.path.isfile(full):
            self._json(404, {'error': 'not found'}); return
        masks_str = json.dumps(masks, separators=(',', ':'))
        with _file_lock:
            with open(full, encoding='utf-8') as f:
                html = f.read()
            # Detect indentation from animation: line (or default 6 spaces)
            m_indent = re.search(r'^([ \t]*)animation\s*:', html, re.MULTILINE)
            indent = m_indent.group(1) if m_indent else '      '
            new_line = f'{indent}masks: {masks_str}, // hs-masks'
            # Replace existing hs-masks sentinel line
            updated = _set_js_config_line(
                html,
                r'^[ \t]*masks\s*:[^\n]*//\s*hs-masks[^\n]*\n?',
                new_line,
                [
                    r'([ \t]*animation\s*:[^\n]*\n)',
                    r'(\n[ \t]*\}\s*\)\s*;)',
                ]
            )
            with open(full, 'w', encoding='utf-8') as f:
                f.write(updated)
        self._json(200, {'ok': True})

    def _api_js_clips(self):
        try:
            body = json.loads(self._read_body())
        except Exception:
            self._json(400, {'error': 'invalid JSON'}); return
        page  = body.get('page', '')
        clips = body.get('clips', [])
        full = self._safe(page.lstrip('/').split('?')[0])
        if not full or not os.path.isfile(full):
            self._json(404, {'error': 'not found'}); return
        clips_str = json.dumps(clips, separators=(',', ':'))
        with _file_lock:
            with open(full, encoding='utf-8') as f:
                html = f.read()
            m_indent = re.search(r'^([ \t]*)animation\s*:', html, re.MULTILINE)
            indent = m_indent.group(1) if m_indent else '      '
            new_line = f'{indent}clips: {clips_str}, // hs-clips'
            updated = _set_js_config_line(
                html,
                r'^[ \t]*clips\s*:[^\n]*//\s*hs-clips[^\n]*\n?',
                new_line,
                [
                    r'([ \t]*animation\s*:[^\n]*\n)',
                    r'(\n[ \t]*\}\s*\)\s*;)',
                ]
            )
            with open(full, 'w', encoding='utf-8') as f:
                f.write(updated)
        self._json(200, {'ok': True})

    def _api_ls(self):
        result = {'spz': [], 'json': []}

        # scenes/ folder (recursive — assets live in subfolders, e.g. scenes/headphones/)
        if os.path.isdir(SCENE_DIR):
            for dirpath, _dirnames, filenames in os.walk(SCENE_DIR):
                for f in sorted(filenames):
                    ext = os.path.splitext(f)[1].lower()
                    rel = os.path.relpath(os.path.join(dirpath, f), ROOT).replace(os.sep, '/')
                    if   ext in SCENE_EXTS: result['spz'].append(rel)
                    elif ext == '.json':    result['json'].append(rel)

        # blender/ folder
        blender_dir = os.path.join(ROOT, 'blender')
        if os.path.isdir(blender_dir):
            for f in sorted(os.listdir(blender_dir)):
                if f.endswith('.json'):
                    result['json'].append(f'blender/{f}')

        # root-level JSON (configs, exports)
        skip = {'package.json', 'package-lock.json'}
        for f in sorted(os.listdir(ROOT)):
            if f.endswith('.json') and f not in skip and os.path.isfile(os.path.join(ROOT, f)):
                result['json'].append(f)

        result['spz'].sort()
        result['json'].sort()
        self._json(200, result)

    def log_message(self, fmt, *args):
        path = args[0] if args else ''
        code = args[1] if len(args) > 1 else ''
        print(f'  {code}  {path}')


# Serve from the project root
os.chdir(ROOT)


def list_scenes():
    return [f for f in sorted(os.listdir(SCENE_DIR))
            if os.path.splitext(f)[1].lower() in SCENE_EXTS]


print()
print('  HoloSplat Dev Server')
print('  ' + '=' * 40)
print(f'  Examples:  http://localhost:{PORT}/examples/viewer.html')
print(f'  Editor:    http://localhost:{PORT}/holosplat/')
print(f'  Scenes:    http://localhost:{PORT}/scenes/')
print()

scenes = list_scenes()
if scenes:
    print('  Scene files found:')
    for f in scenes:
        print(f'    http://localhost:{PORT}/scenes/{urllib.parse.quote(f)}')
else:
    print('  No scene files yet — drop .spz / .splat / .ply files into scenes/')
print()
print('  Press Ctrl+C to stop.')
print()

with http.server.ThreadingHTTPServer(('', PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n  Server stopped.')
