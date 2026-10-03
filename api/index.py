import os
import sys
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import HTML_DASHBOARD, detector


class handler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_HEAD(self):
        self._set_headers(200, content_type="text/html")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html", "/api/index.py"):
            self._set_headers(200, content_type="text/html")
            self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
        elif parsed.path == "/status":
            self._set_headers(200)
            resp = {
                "status": "online",
                "model_backend": detector.backend,
                "device": detector.device,
                "platform": "Vercel Serverless"
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self._set_headers(200, content_type="text/html")
            self.wfile.write(HTML_DASHBOARD.encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/api/detect", "/detect"):
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len)
            try:
                payload = json.loads(post_body.decode("utf-8"))
                sentence = payload.get("sentence", "")
                result = detector.detect(sentence)
                self._set_headers(200)
                self.wfile.write(json.dumps(result.to_dict()).encode("utf-8"))
            except Exception as e:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(b'{"error": "Not found"}')
