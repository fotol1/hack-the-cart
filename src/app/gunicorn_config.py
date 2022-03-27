import os
import logging


port = os.getenv("PORT", "1313")
workers = os.getenv("WORKERS", "1")
bind = f"0.0.0.0:{port}"
worker_class = "uvicorn.workers.UvicornWorker"


def on_starting(_) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
