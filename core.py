"""Ingestion, preprocessing, retrieval and storage (no UI, no LLM)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).parent / "data"

HEAD = re.compile(
    r"^\s*(?:\d+\.?\s*|[IVX]+\.?\s*)?(abstract|introduction|related work|background|methods?|"
    r"methodology|materials and methods|experiments?|experimental setup|results?|"
    r"results and discussion|discussion|conclusions?|limitations?|references|bibliography)\s*:?\s*$",
    re.I,
)
SECMAP = {
    "abstract": "abstract", "introduction": "intro", "related work": "intro", "background": "intro",
    "method": "methods", "methods": "methods", "methodology": "methods",
    "materials and methods": "methods", "experiment": "methods", "experiments": "methods",
    "experimental setup": "methods", "result": "results", "results": "results",
    "results and discussion": "results", "discussion": "discussion", "conclusion": "discussion",
    "conclusions": "discussion", "limitation": "discussion", "limitations": "discussion",
    "references": "references", "bibliography": "references",
}
RE_EXTRACT = re.compile(
    r"\b(dataset|corpus|benchmark|accuracy|f1|precision|recall|bleu|rouge|auc|rmse|metric|"
    r"outperform|we (propose|use|train))\b|\d+(\.\d+)?\s?%", re.I)
RE_LIMITS = re.compile(
    r"\b(limitation|limited|future work|however|constraint|cannot|did not|small sample|bias|"
    r"drawback|shortcoming)\b", re.I)


@dataclass
class Sentence:
    text: str
    page: int
    section: str


@dataclass
class Paper:
    id: str
    name: str
    title: str
    n_pages: int
    sentences: list
    unreadable: list
    sections_found: int
    tags: str = ""


def paper_id(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:10]


def _split(buf: str, page: int, section: str) -> list[Sentence]:
    out = []
    for t in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", buf):
        t = t.strip()
        if 35 < len(t) < 450 and section != "references":
            out.append(Sentence(t, page, section))
    return out


def parse_pdf(data: bytes, name: str, progress=None) -> Paper:
    """Extract text with page numbers and detect sections. Raises ValueError on bad input."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:  # malformed / not a PDF
        raise ValueError(f"Could not open '{name}' as a PDF ({e}).") from e
    sents: list[Sentence] = []
    unreadable: list[int] = []
    section, found, first_lines = "unknown", 0, []
    for i, page in enumerate(doc, 1):
        if progress:
            progress(i, doc.page_count)
        lines = [re.sub(r"\s+", " ", l).strip() for l in page.get_text("text").splitlines()]
        lines = [l for l in lines if l]
        if len("".join(lines)) < 40:  # likely scanned / image-only page
            unreadable.append(i)
            continue
        if i == 1 or not first_lines:
            first_lines = lines[:8]
        buf = ""
        for l in lines:
            m = HEAD.match(l)
            if m:
                sents += _split(buf, i, section)
                buf, section, found = "", SECMAP[m.group(1).lower()], found + 1
            else:
                buf += l + " "
        sents += _split(buf, i, section)
    title = (doc.metadata or {}).get("title") or next((l for l in first_lines if len(l) > 12), name)
    return Paper(paper_id(data), name, title[:140], doc.page_count, sents, unreadable, found)


# ---------- retrieval ----------
_INDEX: dict = {}


def _index(p: Paper):
    if p.id not in _INDEX:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform([s.text for s in p.sentences])
        _INDEX[p.id] = (vec, X)
    return _INDEX[p.id]


def retrieve(p: Paper, query: str, k: int = 8, min_score: float = 0.1) -> list[Sentence]:
    if not p.sentences:
        return []
    try:
        vec, X = _index(p)
        scores = (X @ vec.transform([query]).T).toarray().ravel()
    except ValueError:  # empty vocabulary
        return []
    return [p.sentences[i] for i in np.argsort(-scores)[:k] if scores[i] >= min_score]


def top_sentences(p: Paper, sections: list[str], n: int) -> list[Sentence]:
    pool = [i for i, s in enumerate(p.sentences) if s.section in sections]
    if not pool:
        return []
    _, X = _index(p)
    best = sorted(pool, key=lambda i: -float(X[i].sum()))[:n]
    return [p.sentences[i] for i in sorted(best)]


SUMMARY_PARTS = [("Overview", ["abstract"], 4), ("Methods", ["methods"], 5),
                 ("Results", ["results"], 5), ("Discussion", ["discussion"], 4)]


def context_for(p: Paper, kind: str, question: str = "") -> list[Sentence]:
    """Passages handed to the summariser / LLM for each task."""
    if kind == "summary":
        if p.sections_found < 2:  # uncertain boundaries -> use the whole paper
            return top_sentences(p, ["unknown", "intro", "abstract", "methods", "results", "discussion"], 18)
        return [s for _, secs, n in SUMMARY_PARTS for s in top_sentences(p, secs, n)]
    if kind == "extract":
        return [s for s in p.sentences if RE_EXTRACT.search(s.text)][:20]
    if kind == "limits":
        return [s for s in p.sentences if RE_LIMITS.search(s.text)][:14]
    return retrieve(p, question, 8)


# ---------- library storage ----------
def save_paper(p: Paper) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / f"{p.id}.json").write_text(json.dumps(asdict(p)), encoding="utf-8")


def load_library() -> list[Paper]:
    papers = []
    for f in sorted(DATA_DIR.glob("*.json")) if DATA_DIR.exists() else []:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d["sentences"] = [Sentence(**s) for s in d["sentences"]]
            papers.append(Paper(**d))
        except Exception:  # skip corrupt files instead of crashing
            continue
    return papers


def delete_paper(p: Paper) -> None:
    (DATA_DIR / f"{p.id}.json").unlink(missing_ok=True)
    _INDEX.pop(p.id, None)
