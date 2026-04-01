#!/usr/bin/env python3
"""
HoloSplat local dev server.

- Serves the entire project with CORS headers so WebGPU can load scene files.
- Put .spz / .splat / .ply scene files in the scenes/ folder.

Usage:
    python server.py [port]       (default port: 8080)

Then open:
    http://localhost:8080/examples/index.html

Scene files are accessible at:
    http://localhost:8080/scenes/your-file.spz
"""

import http.server
import os
import sys
import urllib.parse

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

SCENE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenes')
os.makedirs(SCENE_DIR, exist_ok=True)

SCENE_EXTS = {'.spz', '.splat', '.ply'}


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Range')
        self.send_header('Access-Control-Expose-Headers', 'Content-Length, Content-Range')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):
        path = args[0] if args else ''
        code = args[1] if len(args) > 1 else ''
        print(f'  {code}  {path}')


# Serve from the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def list_scenes():
    files = []
    for f in sorted(os.listdir(SCENE_DIR)):
        if os.path.splitext(f)[1].lower() in SCENE_EXTS:
            files.append(f)
    return files


print()
print('  HoloSplat Dev Server')
print('  ' + '=' * 40)
print(f'  Demo:    http://localhost:{PORT}/examples/index.html')
print(f'  Scenes:  http://localhost:{PORT}/scenes/')
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
