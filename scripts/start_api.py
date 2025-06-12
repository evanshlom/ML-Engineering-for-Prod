#!/usr/bin/env python
"""Start the API server."""

import sys
sys.path.append('.')

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )