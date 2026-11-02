"""Approved-LLM adapter. Swap this file to use a different provider."""
import os


class LLMError(Exception):
    """Raised with a user-readable message when the AI service can't be used."""


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def build_prompt(title: str, task: str, passages) -> str:
    excerpts = "\n".join(f"[p.{s.page}] {s.text}" for s in passages)
    return (
        f'You are a research-paper assistant. Use ONLY the excerpts below from the paper "{title}". '
        "Put a page citation like [p.3] after every claim. If the excerpts do not contain the answer, "
        "say clearly that it was not found in the paper. Never invent facts. "
        'Use Markdown: bold headings and bullets starting with "- ".\n\n'
        f"EXCERPTS:\n{excerpts}\n\nTASK: {task}"
    )


def generate(prompt: str) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise LLMError("The 'anthropic' package is not installed.") from e
    client = anthropic.Anthropic(timeout=60)  # key is read from ANTHROPIC_API_KEY
    try:
        msg = client.messages.create(
            model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"),
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.RateLimitError as e:
        raise LLMError("Rate limited by the AI service - try again shortly.") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not reach the AI service (network or timeout).") from e
    except anthropic.APIError as e:
        raise LLMError(f"AI service error: {e}") from e
    return "".join(b.text for b in msg.content if b.type == "text")
