#!/usr/bin/env python3
"""Local HTTP server for Eubot Coder webapp. Opens browser automatically."""
import http.server
import os
import sys
import threading
import webbrowser
from pathlib import Path

PORT = 3333
WEBAPP = Path(__file__).resolve().parent / "webapp"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEBAPP), **kw)

    def log_message(self, fmt, *args):
        pass  # silent

def main():
    os.chdir(str(WEBAPP))
    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Eubot Chat -> {url}")
    print("Premi Ctrl+C per chiudere.\n")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nChiuso.")
        server.shutdown()

if __name__ == "__main__":
    main()
