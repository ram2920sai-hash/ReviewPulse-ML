"""Pure data operations. Original fields are never overwritten."""
import csv
import io
import pandas as pd
from config import MAX_BYTES, MAX_ROWS, MAX_CHARACTERS


def validate_review(value):
    if not isinstance(value, str) or not value.strip():
        return "Invalid: empty or non-text review"
    if len(value) > MAX_CHARACTERS:
        return f"Invalid: exceeds {MAX_CHARACTERS:,} characters"
    return "Ready"


def read_csv(content):
    if len(content) > MAX_BYTES:
        raise ValueError("CSV exceeds the 10 MB upload limit.")
    try:
        text = content.decode("utf-8-sig")
        rows = csv.reader(io.StringIO(text))
        header = next(rows)
        if not header or any(not c.strip() for c in header) or len(header) != len(set(header)):
            raise ValueError("CSV needs unique, non-empty column names.")
        count = 0
        for row in rows:
            if not row:
                continue
            count += 1
            if count > MAX_ROWS:
                raise ValueError(f"CSV exceeds the {MAX_ROWS:,}-review limit.")
            if len(row) != len(header):
                raise ValueError(f"CSV row {count + 1} has a different number of fields than the header.")
        # Bound parsing too, rather than reading an unlimited frame before checking.
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, nrows=MAX_ROWS + 1)
    except (UnicodeDecodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration) as exc:
        raise ValueError("Upload a valid UTF-8 CSV with a header row.") from exc
    if len(frame) > MAX_ROWS:
        raise ValueError(f"CSV exceeds the {MAX_ROWS:,}-review limit.")
    if frame.empty:
        raise ValueError("CSV contains no review rows.")
    return frame


def suggest_column(frame):
    preferred = ("review", "reviews", "feedback", "text", "comment", "comments")
    return next((c for p in preferred for c in frame.columns if c.lower().strip() == p), frame.columns[0])


def analyze_frame(frame, column, engine, batch_size=16, progress=None):
    statuses = [validate_review(v) for v in frame[column]]
    unique = list(dict.fromkeys(v.strip() for v, s in zip(frame[column], statuses) if s == "Ready"))
    predictions = engine.predict(unique, batch_size, progress) if unique else []
    if len(predictions) != len(unique):
        raise ValueError("The model returned an unexpected number of predictions.")
    lookup = dict(zip(unique, predictions))
    metadata = []
    for value, status in zip(frame[column], statuses):
        if status != "Ready":
            metadata.append({"sentiment": None, "confidence": None, "positive_score": None,
                             "neutral_score": None, "negative_score": None, "status": status})
        else:
            item = dict(lookup[value.strip()])
            item["status"] = "Analyzed · truncated" if item.pop("truncated", False) else "Analyzed"
            metadata.append(item)
    prefix = "rp_"
    while any(prefix + name in frame.columns for name in metadata[0]):
        prefix = "rp_" + prefix
    output = frame.reset_index(drop=True).copy()
    for name in metadata[0]:
        output[prefix + name] = [r[name] for r in metadata]
    return output, prefix, len(unique)


def spreadsheet_safe(frame):
    def escape(value):
        if isinstance(value, str) and (value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))):
            return "'" + value
        return value
    # Protect native table downloads too. The original analysis frame remains unchanged.
    return frame.map(escape)


def safe_csv(frame):
    return spreadsheet_safe(frame).to_csv(index=False).encode("utf-8-sig")
