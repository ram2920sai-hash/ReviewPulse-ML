# ReviewPulse ML

A local customer-feedback analyzer with a monochrome glass dashboard inspired by your reference, layered white panels, soft shadows, a custom 3D sculptural artwork, single-review studio, CSV batch workflow, real sentiment probabilities, and safe downloadable results.

## Run

Open a terminal in this folder. Python 3.11 or newer is recommended (a compatible PyTorch wheel is required for your Python version).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit. First analysis downloads the Cardiff NLP model into `.model-cache/` (allow disk space for the model and dependencies). Subsequent analyses reuse the loaded model. No Hugging Face token or paid API is required. The initial download requires internet access; cached files support later offline use.

Model loading explicitly downloads and verifies the original model's configuration, PyTorch weights, and tokenizer files before opening the local snapshot. A complete cached snapshot is reused without network access. This avoids a Transformers checkpoint-resolution error on fresh cloud servers that have a configuration file but no downloaded weights.

On this computer, dependencies were installed in the parent workspace's `.venv`. Double-click `Start-ReviewPulse.cmd` to use that environment. For a standalone copy, run `Setup-ReviewPulse.cmd` once, then `Start-ReviewPulse.cmd`. A portable source ZIP excludes model files and environments; a new computer needs the initial model download.

## Interface preview

Use `Preview-Interface.cmd`, or run `python -m streamlit run app.py -- --interface-preview`. This mode only needs Streamlit, Pandas, and Plotly. It uses clearly labeled, manually curated example sentiments to show the dashboard layout; it supplies no model confidence or performance figures. Live inference is disabled in this mode. Upload preview, navigation, filters, charts, and example CSV downloads remain available. Launch normally for actual model analysis.

## Demo walkthrough

1. Open Overview: the empty state contains no fabricated metrics.
2. Open Review studio, choose an example, and analyze. Discuss sentiment and all three class probabilities.
3. Open Batch analysis and download the sample CSV. Upload it, choose `review`, and analyze.
4. The sample has 12 rows: 11 valid reviews, one empty row, and one duplicate. Ten unique texts require inference.
5. Open Overview to explore the latest analysis. Filter and search without re-running the model.
6. Download filtered or complete results. Explain invalid rows, source attribution, and model confidence.
7. Open Methodology to explain the pretrained model and its limitations.

## Architecture

- `app.py`: navigation, forms, session results, and orchestration.
- `ui.py` and `assets/style.css`: visual components and charts.
- `inference.py`: cached model, configuration-based label mapping, thread-safe batched inference, and CPU fallback.
- `data.py`: validation, deduplication, order-preserving result mapping, and CSV export protection.
- `config.py`: explicit limits and visual constants.

The black ribbon-and-sphere artwork in `assets/sculpture.png` was created with the built-in image generator. Its brief: a black satin three-loop torus surrounding a speckled white stone sphere, on black, with soft studio edge light and no text. Fonts use DM Sans and Manrope when Google Fonts is reachable, with local sans-serif fallbacks. Inference runs locally regardless of font availability.

## Limits and behavior

UTF-8 CSV files up to 10 MB / 5,000 rows; 20,000 characters per review. Text exceeding the model token window is truncated and explicitly flagged. Smaller batches reduce peak memory. CSV columns must have unique non-empty headers. Original fields are preserved; generated fields use `rp_`, with an additional prefix if required to avoid collisions. CSV text values are retained as strings. Formula-like exported values are prefixed with an apostrophe for spreadsheet safety.

The dashboard represents the latest successful analysis, including a single review or an explicitly labeled sample run. Results are isolated to the browser session, replaced on the next successful run, and are not stored in a global data cache or persistent database. Refresh/restart can lose results. The shared model resource contains weights and tokenizer, not a review history. Uploaded filenames are not displayed or retained in analysis records. Download results to keep them.

## Testing

Install development dependencies with `python -m pip install -r requirements-dev.txt` before running the tests. Production hosting installs only `requirements.txt`.

```powershell
python -m pytest -q
python smoke_test.py
```

The regular suite checks data handling and real Streamlit page rendering without downloading the model. Test doubles only isolate data-processing tests; the app never uses fake predictions. `smoke_test.py` downloads/loads the actual model and checks single reviews, batch processing, duplicate mapping, invalid rows, safe export, and long-input truncation.

For a separate human-labeled CSV with `review` and `label` columns, run `python evaluate.py labeled_reviews.csv`. This reports actual accuracy, per-class precision/recall/F1, macro F1, and a confusion matrix. Do not use model-generated labels as ground truth. No measured evaluation score is bundled with this project.

### Verified on this computer

On 20 November 2026, Python 3.14 / CPU: all 9 automated tests passed. Actual model checks passed for single reviews, batch inference, duplicate mapping, invalid rows, truncation, export, Streamlit submission, dashboard navigation, and filtering without repeat inference. A warm local-cache smoke run loaded the model in 0.46 seconds and analyzed two short reviews in 0.16 seconds. These are observations, not latency guarantees.

Browser checks covered a real single-review prediction, sample CSV upload, review-column selection, analysis, dashboard charts, and a downloaded results file. The 12-row sample produced 11 valid rows (5 positive, 3 neutral, 3 negative) and one explicit invalid row; its 10 unique texts were processed in 0.24 seconds in that browser run. The actual downloaded results are included in `samples/analyzed_example.csv`. This is a model output example, not a ground-truth evaluation set.

Desktop and mobile layouts were inspected; the 390px mobile breakpoint had no horizontal page overflow. The preview image is in `assets/dashboard-preview.jpg`.

## Methodology and attribution

Model: [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), licensed CC BY 4.0 as stated in its model card. Use the model card for author attribution and the TimeLMs paper reference. Usernames and links are normalized following the authors' example. Label ordering is obtained from the model configuration rather than hard-coded.

This is an integration of a pretrained English model, not a newly trained classifier. Twitter training data can differ from customer-review language. Sarcasm, mixed sentiments, Telugu/transliteration, and other languages are not validated. Probability is not calibrated certainty; positive-review share is not a validated satisfaction score. No accuracy score is claimed. Evaluate independently on representative labeled customer reviews before making deployment claims.

Streamlit caching reference: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource

## Online deployment

Streamlit Community Cloud configuration: put this folder's contents at the repository root, choose branch `main`, entrypoint `app.py`, and Python 3.12 in Advanced settings. Runtime dependencies are pinned; Linux uses the CPU-only PyTorch wheel from the official PyTorch index. The shared configuration leaves the bind address to the hosting platform; local launch shortcuts explicitly bind to loopback.

Do not commit `.venv`, `.model-cache`, logs, uploaded reviews, secrets, or `.env` files. The model downloads from its original Hugging Face repository on first analysis, then stays cached while the process lives. No API key is required. Confirm available memory with a real CSV run before handing over a public URL. An initial app build or wake-up can be slower than a warm analysis. Public hosting has not been verified until an actual deployed URL passes testing.

Deployment reference: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Troubleshooting

- Model download failure: verify internet access and disk space, then retry. No fake fallback is used.
- Out of memory: use batch size 4 or 8, close memory-heavy applications, and restart. GPU runtime errors retry on CPU.
- Dependency installation failure: use a Python version supported by the selected PyTorch release and install from PyPI.
- CSV error: save as UTF-8 with unique column names and stay within the declared limits.
- No results after refresh: session results are temporary; use the CSV download before closing.
