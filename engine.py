"""Shared one-thread IndoBERT ONNX INT8 model for Streamlit UI and REST routes."""

import json
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

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Tokenizer


MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", Path(__file__).resolve().parent / "indobert-labeler")
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
        with (model_path / "label_config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)

        self.labels = config["labels"]
        self.threshold = float(config.get("threshold", 0.5))
        self.max_length = int(config.get("max_length", 128))
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("label_config.json must contain unique, non-empty labels")
        if not 0 < self.threshold <= 1 or self.max_length < 1:
            raise ValueError("Invalid threshold or max_length in label_config.json")

        with (model_path / "config.json").open(encoding="utf-8") as handle:
            model_config = json.load(handle)
        model_labels = [model_config["id2label"][str(index)] for index in range(len(self.labels))]
        if self.labels != model_labels:
            raise ValueError("Label order differs between label_config.json and config.json")
        if self.max_length > model_config["max_position_embeddings"]:
            raise ValueError("max_length exceeds the model's position limit")

        self.tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(
            pad_id=model_config["pad_token_id"], pad_token="[PAD]"
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        model_file = model_path / "model.int8.onnx"
        with model_file.open("rb") as handle:
            if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("model.int8.onnx is a Git LFS pointer; fetch the real weights with git lfs pull")
        self.model = ort.InferenceSession(
            str(model_file), options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.model.get_inputs()}
        if self.input_names != {"input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError("Unexpected ONNX inputs")
        output = self.model.get_outputs()[0]
        if output.name != "logits" or output.shape[-1] != len(self.labels):
            raise ValueError("ONNX classifier output does not match the label configuration")
        self.lock = Lock()
        # Cache probabilities, so changing only the threshold never reruns the model.
        self._cached_probabilities = lru_cache(maxsize=CACHE_SIZE)(self._predict_probabilities)

    def _predict_probabilities(self, text: str) -> tuple[float, ...]:
        encoding = self.tokenizer.encode(text)
        inputs = {
            "input_ids": np.asarray([encoding.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
            "token_type_ids": np.asarray([encoding.type_ids], dtype=np.int64),
        }
        logits = self.model.run(["logits"], inputs)[0][0]
        # Clipping keeps sigmoid stable even for extreme logits.
        return tuple((1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))).tolist())

    def predict(self, text: str, threshold: float | None) -> PredictResponse:
        cutoff = self.threshold if threshold is None else threshold
        with self.lock:
            probabilities = self._cached_probabilities(text.lower().strip())

        return PredictResponse(
            labels=[
                label
                for label, probability in zip(self.labels, probabilities)
                if probability >= cutoff
            ],
            probabilities=dict(zip(self.labels, probabilities)),
            threshold=cutoff,
        )


@lru_cache(maxsize=1)
def get_labeler() -> Labeler:
    return Labeler(MODEL_PATH)
