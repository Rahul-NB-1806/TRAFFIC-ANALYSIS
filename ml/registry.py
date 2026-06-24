import os
import glob
import json
import joblib
from datetime import datetime
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = "./models"


def _ensure_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def _metadata_path(model_path: str) -> str:
    base, _ = os.path.splitext(model_path)
    return base + "_meta.json"


def save_model(model, metrics: dict, name: str = "model") -> str:
    """Save model with timestamp. Returns path."""
    _ensure_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.pkl"
    path = os.path.join(MODELS_DIR, filename)

    try:
        joblib.dump(model, path)
    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}")
        raise

    metadata = {
        "name": name,
        "timestamp": timestamp,
        "saved_at": datetime.now().isoformat(),
        "metrics": metrics,
    }
    meta_path = _metadata_path(path)
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save metadata to {meta_path}: {e}")
        # Clean up model file if metadata fails
        os.remove(path)
        raise

    logger.info(f"Model saved to {path}")
    return path


def load_model(tag: str = "latest") -> Tuple[object, dict]:
    """Load model. 'latest' picks most recent. Returns (model, metadata)."""
    _ensure_dir()
    pattern = os.path.join(MODELS_DIR, "*.pkl")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No model files found in {MODELS_DIR}")

    if tag == "latest":
        path = files[-1]
    else:
        matches = [f for f in files if tag in os.path.basename(f)]
        if not matches:
            raise FileNotFoundError(f"No model found matching tag '{tag}' in {MODELS_DIR}")
        path = matches[-1]

    try:
        model = joblib.load(path)
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise

    meta_path = _metadata_path(path)
    metadata = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {meta_path}: {e}")

    logger.info(f"Model loaded from {path}")
    return model, metadata


def list_models() -> list[dict]:
    """List all saved models with metadata."""
    _ensure_dir()
    pattern = os.path.join(MODELS_DIR, "*.pkl")
    files = sorted(glob.glob(pattern))

    result = []
    for path in files:
        meta_path = _metadata_path(path)
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read metadata for {path}: {e}")
        meta["filename"] = os.path.basename(path)
        meta["filepath"] = path
        meta["file_size_bytes"] = os.path.getsize(path)
        result.append(meta)

    return result


def delete_model(tag: str) -> bool:
    """Delete a model by tag/name."""
    _ensure_dir()
    pattern = os.path.join(MODELS_DIR, "*.pkl")
    files = glob.glob(pattern)

    target = None
    if tag == "latest":
        files_sorted = sorted(files)
        if not files_sorted:
            logger.warning("No models to delete")
            return False
        target = files_sorted[-1]
    else:
        matches = [f for f in files if tag in os.path.basename(f)]
        if not matches:
            logger.warning(f"No model found matching tag '{tag}'")
            return False
        target = matches[-1]

    try:
        os.remove(target)
        meta_path = _metadata_path(target)
        if os.path.exists(meta_path):
            os.remove(meta_path)
        logger.info(f"Deleted model {target}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete model {target}: {e}")
        return False
