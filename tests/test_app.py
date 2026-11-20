from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_pages_render_without_model_loading():
    at = AppTest.from_file(str(APP)).run(timeout=30)
    assert not at.exception
    assert "dataset" not in at.session_state
    for page in ["Review studio", "Batch analysis", "Methodology", "Overview"]:
        at.radio(key="page").set_value(page).run()
        assert not at.exception


def test_blank_review_does_not_run_model():
    at = AppTest.from_file(str(APP)).run(timeout=30)
    at.radio(key="page").set_value("Review studio").run()
    at.button(key="FormSubmitter:review_form-Analyze sentiment").click().run()
    assert not at.exception
    assert at.warning
    assert "dataset" not in at.session_state


def test_results_survive_navigation_and_filters_without_reinference(monkeypatch):
    import inference
    class TestEngine:
        device = "cpu"
        calls = 0
        def predict(self, texts, batch_size, progress):
            self.calls += 1
            return [dict(sentiment="Positive", confidence=.8, positive_score=.8, neutral_score=.15,
                         negative_score=.05, truncated=False) for _ in texts]
    engine = TestEngine()
    monkeypatch.setattr(inference, "get_engine", lambda: engine)
    at = AppTest.from_file(str(APP)).run(timeout=30)
    at.radio(key="page").set_value("Review studio").run()
    at.text_area(key="review_text").input("Great product")
    at.button(key="FormSubmitter:review_form-Analyze sentiment").click().run()
    assert not at.exception
    assert engine.calls == 1, ([e.value for e in at.error], [w.value for w in at.warning], [c.value for c in at.code])
    at.radio(key="page").set_value("Overview").run()
    assert not at.exception
    at.multiselect(key="overview_filter").set_value(["Positive"]).run()
    at.text_input(key="overview_search").input("Great").run()
    assert not at.exception
    assert engine.calls == 1
    at.radio(key="page").set_value("Batch analysis").run()
    assert not at.exception
    assert len(at.session_state["dataset"]["frame"]) == 1


def test_interface_preview_renders_without_inference(monkeypatch):
    import sys
    import inference
    def forbidden():
        raise AssertionError("Interface preview must never load the model")
    monkeypatch.setattr(sys, "argv", ["app.py", "--interface-preview"])
    monkeypatch.setattr(inference, "get_engine", forbidden)
    at = AppTest.from_file(str(APP)).run(timeout=30)
    assert not at.exception
    record = at.session_state["dataset"]
    assert record["preview"] and record["seconds"] is None
    assert record["frame"]["rp_confidence"].isna().all()
    for page in ["Review studio", "Batch analysis", "Methodology", "Overview"]:
        at.radio(key="page").set_value(page).run()
        assert not at.exception
    at.text_input(key="overview_search").input("headphones").run()
    assert not at.exception
