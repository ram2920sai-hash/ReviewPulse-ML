import pandas as pd
import pytest
from data import analyze_frame, read_csv, safe_csv, validate_review
from inference import class_labels, preprocess


def test_model_labels_follow_config_not_assumed_order():
    assert class_labels({0: "positive", 1: "negative", 2: "neutral"}) == ["Positive", "Negative", "Neutral"]
    assert class_labels({"0": "negative", "1": "neutral", "2": "positive"})[0] == "Negative"
    with pytest.raises(ValueError):
        class_labels({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_input_validation_and_csv_limits():
    assert validate_review("   ").startswith("Invalid")
    assert validate_review(123).startswith("Invalid")
    assert validate_review("a" * 20001).startswith("Invalid")
    for bad in (b"", b"review\n", b"review,review\na,b\n", b"\xff", b"\na\n", b"review\na,b\n"):
        with pytest.raises(ValueError):
            read_csv(bad)
    with pytest.raises(ValueError, match="5,000"):
        read_csv(("review\n" + "fine\n" * 5001).encode())
    with pytest.raises(ValueError, match="10 MB"):
        read_csv(b"a" * (10 * 1024 * 1024 + 1))


def test_deduplication_row_order_invalid_rows_and_column_collisions():
    class RecordingEngine:
        def predict(self, texts, batch_size, progress):
            self.texts = texts
            return [dict(sentiment="Positive", confidence=.8, positive_score=.8, neutral_score=.1,
                         negative_score=.1, truncated=False) for text in texts]
    engine = RecordingEngine()
    source = pd.DataFrame({"id": [5, 3, 8, 2], "review": [" good ", "", "good", None], "rp_status": ["keep"] * 4})
    result, prefix, unique = analyze_frame(source, "review", engine)
    assert unique == 1 and engine.texts == ["good"]
    assert list(result.id) == [5, 3, 8, 2]
    assert list(result.rp_status) == ["keep"] * 4
    assert prefix != "rp_"
    assert result.loc[1, prefix + "status"].startswith("Invalid")
    assert pd.isna(result.loc[3, prefix + "confidence"])
    assert result.loc[0, "review"] == " good "


def test_safe_export_leaves_original_untouched():
    source = pd.DataFrame({"review": ["=SUM(A1)", "  +cmd", "@name", "plain", "\tformula"]})
    exported = safe_csv(source).decode("utf-8-sig")
    assert "'=SUM(A1)" in exported and "'  +cmd" in exported and "'@name" in exported
    assert source.iloc[0, 0] == "=SUM(A1)"


def test_authors_preprocessing():
    assert preprocess("Hi @someone https://example.com") == "Hi @user http"
