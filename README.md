# Smart Research Paper Summarization Assistant (prototype)

Chat-style Streamlit app: upload a paper PDF, then summarize it, extract datasets/methods/results,
list limitations, or ask questions. Every answer cites page numbers and has an "Inspect evidence" panel.

## Run
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add ANTHROPIC_API_KEY (optional)
streamlit run app.py
```
Works on Windows/macOS/Linux with Python 3.10+ and any modern browser. Without an API key the app
still runs and returns extractive (non-AI) results.

## Structure
| File | Role |
|---|---|
| `core.py` | PDF extraction (PyMuPDF), section detection, TF-IDF retrieval, local library storage |
| `llm.py` | LLM adapter (Anthropic API; key from environment). Replace to use another provider |
| `analysis.py` | Picks evidence, calls the LLM, falls back to extractive output on any failure |
| `app.py` | Streamlit chat UI |

## Behaviour and limits
- Scanned/image-only pages are flagged and skipped (no OCR). Uncertain section boundaries are flagged.
- Retrieval is TF-IDF (swap for Sentence Transformers + FAISS for larger libraries).
- Saved papers are stored as JSON in `data/` (extracted text only). Delete them from the sidebar.
- Use only public, synthetic or explicitly permitted papers. Do not commit `data/` or `.env`.
- AI output can be wrong: verify against the PDF; human review is required for important decisions.

## Recovery
Malformed PDFs show an error and are ignored. If the AI service times out or is rate limited, the app
shows a warning and returns extractive results. If a saved file in `data/` is corrupt it is skipped;
delete it manually and re-upload the paper.
