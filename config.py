from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_CACHE = ROOT / ".model-cache"
MAX_ROWS = 5000
MAX_BYTES = 10 * 1024 * 1024
MAX_CHARACTERS = 20000
LABELS = ("Positive", "Neutral", "Negative")
COLORS = {"Positive": "#181818", "Neutral": "#858585", "Negative": "#c6c6c6"}
