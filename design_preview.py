"""Interface-only fixture: curated labels, no model predictions or invented confidence."""
import pandas as pd
from config import ROOT
from data import read_csv


def example_record():
    frame = read_csv((ROOT / "samples" / "customer_reviews.csv").read_bytes())
    frame["rp_sentiment"] = ["Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral",
                              "Positive", "Negative", "Neutral", "Positive", None, "Positive"]
    for column in ("confidence", "positive_score", "neutral_score", "negative_score"):
        frame["rp_" + column] = pd.Series([None] * len(frame), dtype="float64")
    frame["rp_status"] = ["Design example · manually labeled"] * len(frame)
    frame.loc[10, "rp_status"] = "Invalid: empty review"
    return dict(frame=frame, prefix="rp_", column="review", source="Interface preview · manually labeled examples",
                unique=10, seconds=None, device="not running", preparation=0, timestamp="Design preview", preview=True)
