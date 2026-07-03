"""Production entrypoint: waitress in front of our Flask app."""

import os

from waitress import serve

from app import app

if __name__ == "__main__":
    serve(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8089")),
    )
