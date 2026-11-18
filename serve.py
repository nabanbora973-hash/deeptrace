"""Production entrypoint: honours $PORT (Render, Cloud Run, Hugging Face Spaces).

    python serve.py            # local
    PORT=7860 python serve.py  # Hugging Face Spaces / Cloud Run
"""

from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "deeptrace.api:app",
        host="0.0.0.0",
        port=port,
        workers=int(os.environ.get("WEB_CONCURRENCY", 1)),
        timeout_keep_alive=120,
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
