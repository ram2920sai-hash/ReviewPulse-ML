import time
import sys
from datetime import datetime
import pandas as pd
import streamlit as st
from config import MODEL_ID, MAX_ROWS, MAX_CHARACTERS, ROOT, LABELS
from data import read_csv, suggest_column, analyze_frame, safe_csv, validate_review, spreadsheet_safe
from inference import get_engine
from ui import style, heading, card, distribution, sentiment_card, probability_chart, sculpture_card

st.set_page_config(page_title="ReviewPulse · Customer intelligence", page_icon="◈", layout="wide")
style()
PREVIEW = "--interface-preview" in sys.argv
if PREVIEW and "preview_initialized" not in st.session_state:
    from design_preview import example_record
    st.session_state["dataset"] = example_record()
    st.session_state["preview_initialized"] = True
if PREVIEW:
    st.html('<div class="preview-banner"><span class="preview-tag">DESIGN PREVIEW</span><span>Explore the interface · Example labels are manually assigned. No model is running.</span></div>')


def navigate(page):
    st.session_state["page"] = page


def run_analysis(frame, column, source, batch_size=16):
    if PREVIEW:
        st.info("This is your interface preview. Live analysis will be enabled after the design is approved.")
        return None
    status = st.status("Preparing the sentiment model…", expanded=True)
    try:
        status.write("First use downloads model files. Later analyses reuse the loaded model.")
        started = time.perf_counter()
        engine = get_engine()
        preparation = time.perf_counter() - started
        status.write(f"Model ready on {engine.device.upper()} · preparation {preparation:.2f}s")
        progress = st.progress(0, text="Analyzing reviews locally…")
        started = time.perf_counter()
        result, prefix, unique = analyze_frame(frame, column, engine, batch_size,
            lambda done, total: progress.progress(done / total, text=f"Analyzed {done:,} / {total:,} unique reviews"))
        duration = time.perf_counter() - started
        progress.empty()
        record = dict(frame=result, prefix=prefix, column=column, source=source, unique=unique,
                      seconds=duration, device=engine.device, preparation=preparation,
                      timestamp=datetime.now().strftime("%d %b %Y, %H:%M"))
        # Only this browser session owns the review results. Replace, do not accumulate unbounded history.
        st.session_state["dataset"] = record
        status.update(label=f"Analysis complete · {duration:.2f}s processing", state="complete", expanded=False)
        return record
    except Exception as exc:
        status.update(label="Analysis could not finish", state="error", expanded=False)
        st.error("The real model is unavailable or processing failed. No predictions have been invented.")
        st.info("Check your internet connection for the first model download, available disk space, and memory. "
                "Retry with a smaller batch size. Existing successful results are kept.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {str(exc)[:1200]}", language=None)
        return None


def results_table(record, key):
    frame, prefix = record["frame"], record["prefix"]
    a, b = st.columns([1, 2])
    selected = a.multiselect("Sentiment", list(LABELS) + ["Invalid"], key=key + "_filter", placeholder="All sentiments")
    search = b.text_input("Search reviews", key=key + "_search", placeholder="Find a word or phrase…")
    visible = frame
    if selected:
        mask = frame[prefix + "sentiment"].isin(selected)
        if "Invalid" in selected:
            mask |= frame[prefix + "sentiment"].isna()
        visible = visible[mask]
    if search:
        visible = visible[visible[record["column"]].astype(str).str.contains(search, case=False, regex=False)]
    detailed = st.toggle("Show probability columns", key=key + "_detail", disabled=record.get("preview", False))
    hidden = [prefix + name for name in ("positive_score", "neutral_score", "negative_score")] if not detailed else []
    if record.get("preview"):
        hidden.append(prefix + "confidence")
    st.caption(f"{len(visible):,} of {len(frame):,} rows · All original columns preserved")
    st.dataframe(spreadsheet_safe(visible.drop(columns=hidden)), hide_index=True, width="stretch", height=350,
                 column_config={record["column"]: st.column_config.TextColumn("Customer review", width="large"),
                                prefix + "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                                prefix + "status": st.column_config.TextColumn("Status", width="medium"),
                                prefix + "confidence": st.column_config.ProgressColumn("Model confidence", min_value=0, max_value=1, format="%.3f")})
    st.download_button("Download filtered results", safe_csv(visible), "reviewpulse_filtered.csv", "text/csv", key=key + "_download")
    st.caption("Spreadsheet protection: formula-like text is prefixed with an apostrophe in downloaded files.")


with st.sidebar:
    st.html('<div class="brand"><span class="brand-icon">≋</span>ReviewPulse<span style="color:#777;font-size:12px;letter-spacing:1px">ML</span></div><div class="brand-sub">CUSTOMER INTELLIGENCE</div>')
    st.caption("WORKSPACE")
    page = st.radio("Navigation", ["Overview", "Review studio", "Batch analysis", "Methodology"], key="page", label_visibility="collapsed",
                    format_func=lambda v: {"Overview": "◫   Overview", "Review studio": "✧   Review studio", "Batch analysis": "▤   Batch analysis", "Methodology": "◎   Methodology"}[v])
    st.html('<div class="sidebar-note"><strong>Your reviews. A clearer signal.</strong><p>Turn customer feedback into a view you can act on. Three sentiments. One workspace.</p></div>')
    st.caption("MODEL")
    st.markdown("**Twitter RoBERTa · English**")
    st.caption("Pretrained · Runs locally after download")
    if "dataset" in st.session_state:
        st.caption(f'Last analysis: {st.session_state.dataset["timestamp"]}')
        if st.button("Clear session results", width="stretch"):
            for name in ("dataset", "single_result"):
                st.session_state.pop(name, None)
            st.rerun()
    if PREVIEW and st.button("Restore design examples", width="stretch"):
        from design_preview import example_record
        st.session_state["dataset"] = example_record()
        st.rerun()

if page == "Overview":
    hero_left, hero_right = st.columns([1.6, 1], gap="large")
    with hero_left:
        heading("YOUR CUSTOMER INTELLIGENCE STUDIO", "Feedback, understood.",
                "A thoughtful space to understand your customers. Turn every review into a clearer perspective.")
        tool_b, tool_c = st.columns(2)
        tool_b.button("Analyze a review", key="hero_analyze", type="primary", width="stretch", on_click=navigate, args=("Review studio",))
        tool_c.button("Import CSV", width="stretch", on_click=navigate, args=("Batch analysis",))
    with hero_right:
        sculpture_card()
    record = st.session_state.get("dataset")
    cols = st.columns(4)
    if record:
        frame, prefix = record["frame"], record["prefix"]
        valid = frame[frame[prefix + "sentiment"].notna()]
        values = [("Analyzed reviews", f"{len(valid):,}", f"{len(frame) - len(valid)} invalid rows excluded", "cyan"),
                  ("Positive reviews", f'{(valid[prefix + "sentiment"] == "Positive").mean():.0%}' if len(valid) else "—", "Share of analyzed reviews", "green"),
                  ("Average confidence", f'{valid[prefix + "confidence"].mean():.1%}' if valid[prefix + "confidence"].notna().any() else "—", "Available after live analysis" if record.get("preview") else "Model score · not accuracy", "violet"),
                  ("Processing time", f'{record["seconds"]:.2f}s' if record["seconds"] is not None else "—", "Available after live analysis" if record.get("preview") else f'{record["unique"]} unique texts · {record["device"].upper()}', "default")]
    else:
        values = [("Analyzed reviews", "—", "Your workspace is ready", "cyan"), ("Positive reviews", "—", "Your feedback, measured", "green"),
                  ("Average confidence", "—", "Real model probabilities", "violet"), ("Processing time", "—", "Measured on every analysis", "default")]
    for col, args in zip(cols, values):
        with col:
            card(*args)
    if not record:
        st.html('<div class="empty"><div class="empty-icon">◈</div><h3>Meet your next customer insight.</h3><p>Analyze a single review or bring a CSV. Your dashboard comes to life with real predictions, clear charts, and useful filters.</p></div>')
        a, b, c = st.columns(3)
        a.button("Analyze a review", type="primary", width="stretch", on_click=navigate, args=("Review studio",))
        b.button("Upload a CSV", width="stretch", on_click=navigate, args=("Batch analysis",))
        if c.button("Analyze sample dataset", width="stretch"):
            demo = read_csv((ROOT / "samples" / "customer_reviews.csv").read_bytes())
            if run_analysis(demo, "review", "Sample dataset · demonstration"):
                st.rerun()
        st.caption("Sample mode uses clearly labeled example reviews and real model inference.")
    else:
        if record.get("preview"):
            st.caption("DESIGN EXAMPLES · 12 sample rows · Manually assigned sentiments · Confidence and timings intentionally unavailable")
        else:
            st.caption(f'Source: {record["source"]} · Dashboard represents the latest successful analysis.')
        if len(valid):
            distribution(frame, prefix)
        else:
            st.warning("No valid reviews to chart. Review the row statuses below.")
        st.markdown("### Explore your feedback")
        results_table(record, "overview")

elif page == "Review studio":
    heading("ONE REVIEW. INSTANT CLARITY.", "Read between the lines.",
            "Go beyond the words. Explore the sentiment and confidence behind a customer review.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("#### Try a starting point")
        examples = {"Delighted customer": "The product is fantastic, delivery was quick, and support was so helpful!",
                    "Unhappy customer": "The item broke after two days and customer support ignored my messages.",
                    "Just the facts": "The package contains two cables and a printed instruction manual."}
        example_cols = st.columns(3)
        for col, (label, text) in zip(example_cols, examples.items()):
            if col.button(label, width="stretch"):
                st.session_state["review_text"] = text
        with st.form("review_form"):
            review_value = st.text_area("Customer review", key="review_text", height=210, max_chars=MAX_CHARACTERS,
                         placeholder="Paste a customer review here. What did they love? What could be better?")
            st.caption("English reviews · Maximum 20,000 characters · Long reviews may be truncated to the model's token limit.")
            submitted = st.form_submit_button("Analyze sentiment", type="primary", width="stretch")
        if submitted:
            text = review_value
            problem = validate_review(text)
            if problem != "Ready":
                st.warning(problem)
            else:
                record = run_analysis(pd.DataFrame({"review": [text]}), "review", "Single review")
                if record:
                    row = record["frame"].iloc[0]
                    st.session_state["single_result"] = {name: row[record["prefix"] + name] for name in
                        ("sentiment", "confidence", "positive_score", "neutral_score", "negative_score", "status")}
                    st.session_state["single_result"]["review"] = text
    with right:
        result = st.session_state.get("single_result")
        if result:
            st.markdown("#### Your analysis")
            sentiment_card(result)
            probability_chart(result)
            st.caption("Confidence is a model score, not a guarantee that the prediction is correct.")
            with st.expander("Analyzed review"):
                st.write(result["review"])
            if "truncated" in result["status"]:
                st.warning("This review exceeded the model token limit. Only the beginning was analyzed.")
        else:
            st.html('<div class="studio-placeholder"><div class="eyebrow">SENTIMENT EXPLORER</div><div class="waveform"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><h3>A little context.<br>A clearer perspective.</h3><p>Enter a customer review to uncover the sentiment behind the words.</p><div class="sentiment-chips"><span class="chip-positive">Positive</span><span class="chip-neutral">Neutral</span><span class="chip-negative">Negative</span></div></div>')
        st.caption("Local inference · No paid API · Results stay in this browser session")

elif page == "Batch analysis":
    heading("BIG PICTURE. EVERY DETAIL.", "From feedback to foresight.",
            "Bring your reviews together. Analyze in batches, explore patterns, and export the results.")
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        with st.container(border=True):
            st.markdown("### Bring your customer feedback")
            st.caption(f"UTF-8 CSV · Up to 10 MB · {MAX_ROWS:,} rows · {MAX_CHARACTERS:,} characters per review")
            upload = st.file_uploader("Choose your CSV file", type=["csv"])
            st.download_button("Download sample CSV", (ROOT / "samples" / "customer_reviews.csv").read_bytes(), "sample_reviews.csv", "text/csv")
    with b, st.container(border=True):
        st.markdown("### A smoother workflow")
        st.markdown("**Upload** your CSV with a review column.\n\n**Choose** the column and processing batch size.\n\n**Explore** sentiment, filter feedback, and download.")
        st.caption("Duplicate reviews are processed once. Original rows and their order are preserved.")
    if upload:
        try:
            source_frame = read_csv(upload.getvalue())
            st.markdown("#### File preview")
            st.dataframe(spreadsheet_safe(source_frame.head(8)), hide_index=True, width="stretch")
            with st.form("batch_form"):
                c1, c2 = st.columns([2, 1])
                column = c1.selectbox("Review column", list(source_frame.columns), index=list(source_frame.columns).index(suggest_column(source_frame)))
                batch = c2.selectbox("Batch size", [4, 8, 16, 32], index=2, help="Choose 4 or 8 on a lower-memory computer.")
                st.caption(f"{len(source_frame):,} rows ready · Empty reviews receive an explicit invalid status.")
                process = st.form_submit_button("Analyze uploaded reviews", type="primary", width="stretch")
            if process:
                run_analysis(source_frame, column, "Uploaded CSV", batch)
        except ValueError as exc:
            st.error(str(exc))
    record = st.session_state.get("dataset")
    if record:
        st.markdown("### Latest successful analysis")
        if record.get("preview"):
            st.caption("Interface preview · Manually labeled examples · No inference has run")
        else:
            st.caption(f'Source: {record["source"]} · {record["timestamp"]} · {record["seconds"]:.2f}s inference and data processing · {record["preparation"]:.2f}s model preparation')
        results_table(record, "batch")
        st.download_button("Download complete analysis", safe_csv(record["frame"]), "reviewpulse_complete.csv", "text/csv")

else:
    heading("BUILT TO BE UNDERSTOOD.", "Intelligence, with context.",
            "A transparent look at the model, the process, and what the results really mean.")
    a, b = st.columns(2, gap="large")
    with a, st.container(border=True):
        st.markdown("### The model behind the signal")
        st.markdown(f"**Model:** `{MODEL_ID}`\n\nReviewPulse integrates a pretrained English sentiment classifier. It does not train a new model from scratch.")
        st.markdown("The model assigns a probability to **positive**, **neutral**, and **negative** sentiment. The class with the highest probability becomes the prediction. Labels are read from the model configuration.")
        st.link_button("Read the original model card", f"https://huggingface.co/{MODEL_ID}")
    with b, st.container(border=True):
        st.markdown("### From words to a useful signal")
        st.markdown("**Validate** · Check text, file limits, and review columns.\n\n**Prepare** · Normalize usernames and links as recommended by the model authors.\n\n**Infer** · Tokenize, batch, and run the local model.\n\n**Explore** · Review probabilities, charts, and exportable results.")
    st.markdown("### Know the boundaries")
    st.info("This model is intended for English. Telugu, Telugu transliteration, and mixed-language feedback are not validated. Sarcasm, context, and mixed emotions can lead to incorrect predictions.")
    st.markdown("The model was developed using Twitter data; customer reviews can differ. Long text is truncated to fit the token window and flagged in the result. Confidence is not accuracy, and positive-review percentage is not a validated customer-satisfaction score.")
    st.markdown("### Performance & privacy")
    st.markdown("The tokenizer and model are cached in memory. Duplicate texts are inferred once per batch run. Results live in session state, so changing filters or navigating does not re-run the model. Reviews are not sent to a remote inference API. The first model download needs internet access and may take several minutes. A browser refresh or server restart may reset your results; download anything you want to keep.")
    st.markdown("### Honest evaluation")
    st.markdown("No accuracy claim is made without a representative labeled test set. For a capstone evaluation, independently label customer reviews, keep them separate from any tuning data, and report precision, recall, macro F1, and a confusion matrix. Include difficult and misclassified examples.")

st.html('<div class="footer"><span>ReviewPulse ML · Customer feedback, understood.</span><span>Python / Streamlit / Transformers</span></div>')
