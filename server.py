#!/usr/bin/env python3
"""
HoloSplat local dev server.

- Serves the entire project with CORS headers so WebGPU can load scene files.
- Exposes a small /hs-api for the /holosplat editor to read and write files.
- Put .spz / .splat / .ply scene files in the scenes/ folder.

Usage:
    python server.py [port]       (default port: 8080)

Then open:
    http://localhost:8080/examples/index.html
    http://localhost:8080/holosplat/

Scene files are accessible at:
    http://localhost:8080/scenes/your-file.spz
"""

import http.server
import json
import os
import sys
import urllib.parse

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

ROOT      = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.join(ROOT, 'scenes')
os.makedirs(SCENE_DIR, exist_ok=True)

SCENE_EXTS = {'.spz', '.splat', '.ply'}


class Handler(http.server.SimpleHTTPRequestHandler):

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin',  '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, PUT, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Range, Content-Type')
        self.send_header('Access-Control-Expose-Headers','Content-Length, Content-Range')
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
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
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
        with open(path, 'wb') as f:
            f.write(body)
        self._json(200, {'ok': True})

    def _api_ls(self):
        result = {'spz': [], 'json': []}

        # scenes/ folder
        if os.path.isdir(SCENE_DIR):
            for f in sorted(os.listdir(SCENE_DIR)):
                ext = os.path.splitext(f)[1].lower()
                rel = f'scenes/{f}'
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
print(f'  Examples:  http://localhost:{PORT}/examples/index.html')
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

with http.server.HTTPServer(('', PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n  Server stopped.')
