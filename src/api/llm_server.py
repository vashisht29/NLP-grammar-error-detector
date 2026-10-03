"""
Independent Self-Hosted LLM & Linguistic Reasoning Backend Server.
Runs locally on macOS (via Apple Silicon MPS/Metal) or on any independent Linux GPU server (CUDA).
Provides full REST API endpoints for grammatical error detection, frame semantics, Hinglish filtering, and slang normalization.
Zero dependency on external third-party closed APIs (no OpenAI, no Google Gemini).
"""
import os
import sys
import time
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from typing import List, Optional, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from src.model.detector import DeepGrammarDetector
from src.model.config import ModelConfig


# Global shared detector
_detector: Optional[DeepGrammarDetector] = None


def get_detector() -> DeepGrammarDetector:
    global _detector
    if _detector is None:
        print("[LLM Backend] Initializing DeepGrammarDetector on host hardware...")
        _detector = DeepGrammarDetector()
    return _detector


def create_app() -> Any:
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI and Pydantic must be installed to run the LLM server.")

    app = FastAPI(
        title="Independent Self-Hosted LLM Grammar Engine",
        version="2.0.0",
        description="High-performance linguistic reasoning and deep learning GEC backend running on local Apple Silicon MPS or independent GPU."
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
    @app.get("/health")
    def health_check():
        detector = get_detector()
        process = psutil.Process(os.getpid()) if "psutil" in sys.modules else None
        mem_mb = round(process.memory_info().rss / (1024 * 1024), 1) if process else "N/A"

        return {
            "status": "online",
            "server_type": "Independent Self-Hosted LLM Backend",
            "model_backend": detector.backend,
            "device": detector.device,
            "memory_usage_mb": mem_mb,
            "zero_external_apis": True,
            "supported_accelerators": ["Apple Silicon Metal (MPS)", "NVIDIA CUDA", "Multi-core CPU"],
            "features": [
                "Semantic Agent-Action Role Inversion",
                "Institution Motion Prepositions",
                "Interrogative Run-On Clause Splitting",
                "Hinglish Multi-Layer Guardrail",
                "100+ Slang Normalization & Phonetics",
                "Accidental Word-Gap Rejoining"
            ]
        }

    @app.get("/api/categories")
    def get_categories():
        return {"categories": ModelConfig().error_categories}

    @app.post("/api/detect")
    @app.post("/predict")
    def detect_sentence(req: SentenceRequest):
        text = req.sentence.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Sentence cannot be empty.")
        detector = get_detector()
        result = detector.detect(text)
        return result.to_dict()

    @app.post("/api/detect-batch")
    def detect_batch(req: BatchRequest):
        if not req.sentences:
            raise HTTPException(status_code=400, detail="Sentences list cannot be empty.")
        detector = get_detector()
        results = [detector.detect(s).to_dict() for s in req.sentences]
        return {
            "total_processed": len(results),
            "results": results
        }

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Start Independent LLM Grammar Backend")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    args = parser.parse_args()

    app = create_app()
    import uvicorn
    print(f"\n=======================================================")
    print(f"🚀 Independent LLM Backend starting on http://{args.host}:{args.port}")
    print(f"🔒 Zero dependence on third-party cloud APIs (100% self-hosted)")
    print(f"=======================================================\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
