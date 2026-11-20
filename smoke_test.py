"""Run explicitly: python smoke_test.py (downloads real model on first use)."""
import time
import pandas as pd
from config import ROOT
from inference import SentimentEngine
from data import analyze_frame, read_csv, safe_csv

if __name__ == "__main__":
    engine = SentimentEngine()
    started = time.perf_counter()
    predictions = engine.predict(["I absolutely love this product!", "This was terrible and a waste of money."])
    for prediction in predictions:
        assert abs(sum(prediction[f"{label}_score"] for label in ("positive", "neutral", "negative")) - 1) < 1e-5
    assert predictions[0]["sentiment"] == "Positive"
    assert predictions[1]["sentiment"] == "Negative"
    print(f"Real model: {engine.device}; load {engine.load_seconds:.2f}s; two reviews {time.perf_counter() - started:.2f}s")
    frame = read_csv((ROOT / "samples" / "customer_reviews.csv").read_bytes())
    result, prefix, unique = analyze_frame(frame, "review", engine, batch_size=4)
    assert len(result) == len(frame)
    assert unique == 10
    assert result[prefix + "status"].str.startswith("Invalid").sum() == 1
    assert len(read_csv(safe_csv(result))) == len(frame)
    assert engine.predict(["great " * 1000], 1)[0]["truncated"]
    print("Real batch, duplicate mapping, invalid row, CSV export, and truncation checks passed.")
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    with patch("inference.get_engine", return_value=engine), patch.object(engine, "predict", wraps=engine.predict) as observed:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        app.radio(key="page").set_value("Review studio").run()
        app.text_area(key="review_text").input("I love this product, excellent quality!")
        app.button(key="FormSubmitter:review_form-Analyze sentiment").click().run(timeout=60)
        assert not app.exception
        assert app.session_state["single_result"]["sentiment"] == "Positive"
        app.radio(key="page").set_value("Overview").run()
        app.multiselect(key="overview_filter").set_value(["Positive"]).run()
        app.text_input(key="overview_search").input("quality").run()
        assert not app.exception
        assert observed.call_count == 1
    print("Real-model Streamlit submit, dashboard, filtering, and no-reinference checks passed.")
