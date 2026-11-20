"""Real, local Transformers inference. No user-text result cache."""
import threading
import time
from functools import lru_cache
from config import MODEL_ID, MODEL_CACHE


def class_labels(id2label):
    labels = [str(id2label.get(i, id2label.get(str(i), ""))).capitalize() for i in range(3)]
    if set(labels) != {"Positive", "Neutral", "Negative"}:
        raise ValueError(f"Unsupported model label mapping: {id2label}")
    return labels


def preprocess(text):
    return " ".join("@user" if t.startswith("@") and len(t) > 1 else "http" if t.startswith("http") else t
                    for t in text.split(" "))


class SentimentEngine:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        started = time.perf_counter()
        self.torch = torch
        self.lock = threading.Lock()
        def load(loader):
            try:
                return loader.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE, local_files_only=True)
            except OSError:
                return loader.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)
        # Prefer complete on-disk assets: warm starts and offline sessions need no metadata request.
        self.tokenizer = load(AutoTokenizer)
        self.model = load(AutoModelForSequenceClassification)
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
