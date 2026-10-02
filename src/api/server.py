"""
FastAPI REST API server for Deep Learning Grammatical Error Detection (GED).
Exposes endpoints for single-sentence and batch detection, error explanation, and health checks.
Also includes a zero-dependency fallback server using Python's built-in http.server.
"""
import json
from typing import List, Optional
from ..model.detector import DeepGrammarDetector
from ..model.config import ModelConfig


# Shared detector instance
detector_instance = None


def get_detector() -> DeepGrammarDetector:
    global detector_instance
    if detector_instance is None:
        detector_instance = DeepGrammarDetector()
    return detector_instance


def create_fastapi_app():
    """Builds FastAPI application with Pydantic schemas."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        return None

    app = FastAPI(
        title="Deep Learning English Grammatical Error Detection API",
        version="1.0.0",
        description="REST API for sentence-level grammatical error detection, token localization, and linguistic taxonomy classification."
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class SentenceRequest(BaseModel):
        sentence: str

    class BatchRequest(BaseModel):
        sentences: List[str]

    @app.get("/")
    def root():
        detector = get_detector()
        return {
            "status": "online",
            "model_backend": detector.backend,
            "device": detector.device,
            "version": "1.0.0",
            "endpoints": ["/api/detect", "/api/detect-batch", "/api/categories"]
        }

    @app.get("/api/categories")
    def get_categories():
        return {"categories": ModelConfig().error_categories}

    @app.post("/api/detect")
    def detect_single(request: SentenceRequest):
        if not request.sentence.strip():
            raise HTTPException(status_code=400, detail="Sentence cannot be empty.")
        detector = get_detector()
        result = detector.detect(request.sentence)
        return result.to_dict()

    @app.post("/api/detect-batch")
    def detect_batch(request: BatchRequest):
        if not request.sentences:
            raise HTTPException(status_code=400, detail="Sentence list cannot be empty.")
        detector = get_detector()
        results = [detector.detect(s).to_dict() for s in request.sentences]
        return {"results": results, "total_processed": len(results)}

    return app


def run_builtin_server(port: int = 8000):
    """Zero-dependency HTTP server fallback using standard library http.server."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse

    detector = get_detector()

    class GEDHTTPHandler(BaseHTTPRequestHandler):
        def _set_headers(self, status=200):
            self.send_response(status)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_OPTIONS(self):
            self._set_headers(200)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._set_headers(200)
                resp = {
                    "status": "online",
                    "model_backend": detector.backend,
                    "device": detector.device,
                    "endpoints": ["/api/detect", "/api/categories"]
                }
                self.wfile.write(json.dumps(resp, indent=2).encode("utf-8"))
            elif parsed.path == "/api/categories":
                self._set_headers(200)
                resp = {"categories": ModelConfig().error_categories}
                self.wfile.write(json.dumps(resp, indent=2).encode("utf-8"))
            else:
                self._set_headers(404)
                self.wfile.write(b'{"error": "Not found"}')

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/detect":
                content_len = int(self.headers.get("Content-Length", 0))
                post_body = self.rfile.read(content_len)
                try:
                    payload = json.loads(post_body.decode("utf-8"))
                    sentence = payload.get("sentence", "")
                    result = detector.detect(sentence)
                    self._set_headers(200)
                    self.wfile.write(json.dumps(result.to_dict(), indent=2).encode("utf-8"))
                except Exception as e:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            else:
                self._set_headers(404)
                self.wfile.write(b'{"error": "Endpoint not found"}')

    print(f"\n[GED API] Starting HTTP Server on http://localhost:{port}")
    server = HTTPServer(("0.0.0.0", port), GEDHTTPHandler)
    server.serve_forever()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Start GED REST API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    app = create_fastapi_app()
    if app is not None:
        try:
            import uvicorn
            print(f"[GED API] Starting FastAPI + Uvicorn server on http://localhost:{args.port}...")
            uvicorn.run(app, host="0.0.0.0", port=args.port)
            return
        except Exception:
            pass

    # Fallback
    run_builtin_server(port=args.port)


if __name__ == "__main__":
    main()
