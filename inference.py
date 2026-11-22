"""Real, local Transformers inference. No user-text result cache."""
import threading
import time
from functools import lru_cache
from pathlib import Path
from config import MODEL_ID, MODEL_CACHE


def class_labels(id2label):
    labels = [str(id2label.get(i, id2label.get(str(i), ""))).capitalize() for i in range(3)]
    if set(labels) != {"Positive", "Neutral", "Negative"}:
        raise ValueError(f"Unsupported model label mapping: {id2label}")
    return labels


def preprocess(text):
    return " ".join("@user" if t.startswith("@") and len(t) > 1 else "http" if t.startswith("http") else t
                    for t in text.split(" "))


MODEL_FILES = ("config.json", "pytorch_model.bin", "vocab.json", "merges.txt", "special_tokens_map.json")


def resolve_model_snapshot(cache_dir=MODEL_CACHE):
    """Resolve actual files before Transformers loads this PyTorch-only checkpoint."""
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    cached = [try_to_load_from_cache(MODEL_ID, name, cache_dir=cache_dir) for name in MODEL_FILES]
    if (all(isinstance(p, str) and Path(p).is_file() for p in cached)
            and len({Path(p).parent for p in cached}) == 1):
        return Path(cached[0]).parent
    snapshot = Path(snapshot_download(repo_id=MODEL_ID, cache_dir=cache_dir,
                                      allow_patterns=list(MODEL_FILES), max_workers=1))
    missing = [name for name in MODEL_FILES if not (snapshot / name).is_file()]
    if missing:
        raise OSError("Model download is incomplete. Missing files: " + ", ".join(missing))
    return snapshot


class SentimentEngine:
    def __init__(self, cache_dir=MODEL_CACHE):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        started = time.perf_counter()
        self.torch = torch
        self.lock = threading.Lock()
        snapshot = resolve_model_snapshot(cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        # The original Cardiff NLP repository publishes pytorch_model.bin, not safetensors.
        # An explicit format and verified local path avoid Transformers' None checkpoint bug.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            snapshot, local_files_only=True, use_safetensors=False)
        self.labels = class_labels(self.model.config.id2label)
        self.device = "cpu"
        if torch.cuda.is_available():
            try:
                self.model.to("cuda")
                self.device = "cuda"
            except RuntimeError:
                self.model.to("cpu")
        self.model.eval()
        # RoBERTa position embeddings include a padding offset.
        self.max_length = min(512, self.tokenizer.model_max_length,
                              self.model.config.max_position_embeddings - 2)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.load_seconds = time.perf_counter() - started

    def predict(self, texts, batch_size=16, progress=None):
        if not 1 <= batch_size <= 64:
            raise ValueError("Batch size must be between 1 and 64.")
        output = []
        with self.lock, self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = [preprocess(t) for t in texts[start:start + batch_size]]
                lengths = [len(self.tokenizer.encode(t, truncation=False)) for t in batch]
                encoded = self.tokenizer(batch, padding=True, truncation=True,
                                         max_length=self.max_length, return_tensors="pt")
                try:
                    probabilities = self.model(**{k: v.to(self.device) for k, v in encoded.items()}).logits.softmax(-1)
                except RuntimeError:
                    if self.device != "cuda":
                        raise
                    # Retry this batch on CPU if GPU execution is unavailable or out of memory.
                    self.model.to("cpu")
                    self.device = "cpu"
                    self.torch.cuda.empty_cache()
                    probabilities = self.model(**encoded).logits.softmax(-1)
                for scores, length in zip(probabilities.cpu().tolist(), lengths):
                    winner = max(range(len(scores)), key=scores.__getitem__)
                    result = {f"{label.lower()}_score": float(score) for label, score in zip(self.labels, scores)}
                    result.update(sentiment=self.labels[winner], confidence=float(scores[winner]),
                                  truncated=length > self.max_length)
                    output.append(result)
                if progress:
                    progress(min(start + batch_size, len(texts)), len(texts))
        return output


def get_engine():
    # Import Streamlit lazily so pure validation tests need no running UI.
    import streamlit as st
    return _cached_loader()()


@lru_cache(maxsize=1)
def _cached_loader():
    import streamlit as st
    return st.cache_resource(show_spinner=False, max_entries=1)(SentimentEngine)
