"""Glue between retrieval (core) and generation (llm), with extractive fallback."""
import re
from dataclasses import dataclass, field

import core
import llm

TASKS = {
    "summary": "Write a structured summary with these headings, each followed by 2-3 short bullets: "
               "Overview, Methods, Results, Discussion.",
    "extract": "Under the headings Datasets, Methods, Metrics, Key results, list what the paper reports "
               "as short bullets. Write 'Not found' under any heading with no support.",
    "limits": "List the limitations of this work as bullets, marking which are explicitly stated by the "
              "authors and which are only implied. Finish with an 'Ambiguities' heading if anything is unclear.",
}


@dataclass
class Result:
    text: str
    evidence: list = field(default_factory=list)
    ai: bool = False
    warning: str = ""


def detect_kind(text: str) -> str:
    short = len(text) < 60
    if re.search(r"^summari[sz]e|summary", text, re.I):
        return "summary"
    if short and re.search(r"extract|dataset|metric", text, re.I):
        return "extract"
    if short and re.search(r"limitation", text, re.I):
        return "limits"
    return "qa"


def _bullets(sents) -> str:
    return "\n".join(f"- {s.text} [p.{s.page}]" for s in sents) or "_Not found in the detected text._"


def _extractive(p, kind: str, evidence) -> str:
    if kind == "summary":
        if p.sections_found < 2:
            return "**Summary (sections uncertain)**\n" + _bullets(evidence)
        return "\n\n".join(f"**{t}**\n{_bullets(core.top_sentences(p, secs, n))}"
                           for t, secs, n in core.SUMMARY_PARTS)
    heading = {"extract": "Passages about datasets, methods and results",
               "limits": "Passages mentioning limitations",
               "qa": "Most relevant passages"}[kind]
    return f"**{heading}**\n{_bullets(evidence[:5] if kind == 'qa' else evidence)}"


def run(p, kind: str, question: str) -> Result:
    evidence = core.context_for(p, kind, question)
    if not evidence:
        return Result("I couldn't find this in the paper. Try rephrasing, or the paper may not cover it.")
    warning = ""
    if llm.available():
        try:
            prompt = llm.build_prompt(p.title, TASKS.get(kind) or f"Answer this question: {question}", evidence)
            return Result(llm.generate(prompt), evidence, ai=True)
        except llm.LLMError as e:
            warning = f"{e} Showing extractive results instead."
    else:
        warning = "No ANTHROPIC_API_KEY set - showing extractive results instead of AI-written text."
    return Result(_extractive(p, kind, evidence), evidence, ai=False, warning=warning)
