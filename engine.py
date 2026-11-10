"""Shared one-thread TF-IDF model for Streamlit UI and REST routes."""

import os
import warnings
from functools import lru_cache
from pathlib import Path
from threading import Lock


def configure_cpu():
    # Set these before loading native libraries, including when run without Docker.
    for name in (
        "ORT_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_BLOCKTIME"] = "0"
    # Linux threads inherit this affinity. Never assume container CPU IDs start at 0.
    if os.environ.get("CPU_AFFINITY", "1") == "1" and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        except OSError as error:
            warnings.warn(f"CPU affinity unavailable; using one-thread limits: {error}", RuntimeWarning)


configure_cpu()

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.exceptions import InconsistentVersionWarning


MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", Path(__file__).resolve().parent / "model.joblib")
)
CACHE_SIZE = int(os.environ.get("PREDICTION_CACHE_SIZE", "128"))
if CACHE_SIZE < 0:
    raise ValueError("PREDICTION_CACHE_SIZE must be non-negative")


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=20000)
    threshold: float | None = Field(default=None, gt=0, le=1)


class PredictResponse(BaseModel):
    labels: list[str]
    probabilities: dict[str, float]
    threshold: float


class Labeler:
    def __init__(self, model_path: Path):
        # Load only the local, trusted model packaged with this application.
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            bundle = joblib.load(model_path)
        if not isinstance(bundle, dict):
            raise ValueError("model.joblib must contain a model/labels dictionary")
        self.model = bundle["model"]
        # Also constrain nested FeatureUnion/OneVsRest joblib parallelism.
        self.model.set_params(**{
            name: 1 for name in self.model.get_params(deep=True)
            if name == "n_jobs" or name.endswith("__n_jobs")
        })
        self.labels = list(bundle["labels"])
        self.threshold = float(bundle.get("threshold", 0.5))
        if (
            not self.labels
            or any(not isinstance(label, str) or not label.strip() for label in self.labels)
            or len(set(self.labels)) != len(self.labels)
        ):
            raise ValueError("The model bundle must contain unique, non-empty labels")
        if not 0 < self.threshold <= 1:
            raise ValueError("Invalid default threshold in model bundle")
        self.lock = Lock()
        self._cached_probabilities = lru_cache(maxsize=CACHE_SIZE)(self._predict_probabilities)
        # Fail startup for unfitted models or incompatible probability outputs.
        self.predict("uji model", None)

    def _predict_probabilities(self, text: str) -> tuple[float, ...]:
        probabilities = np.asarray(self.model.predict_proba([text]), dtype=float)
        if (
            probabilities.shape != (1, len(self.labels))
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
        ):
            raise ValueError("Probability output does not match the model's label configuration")
        return tuple(probabilities[0].tolist())

    def predict(self, text: str, threshold: float | None) -> PredictResponse:
        cutoff = self.threshold if threshold is None else threshold
        with self.lock:
            values = self._cached_probabilities(text.lower().strip())
        return PredictResponse(
            labels=[
                label for label, probability in zip(self.labels, values)
                if probability >= cutoff
            ],
            probabilities=dict(zip(self.labels, values)),
            threshold=cutoff,
        )


@lru_cache(maxsize=1)
def get_labeler() -> Labeler:
    return Labeler(MODEL_PATH)
