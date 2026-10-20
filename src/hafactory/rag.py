"""Private assistant over the synthetic note corpus.

Answers are sentences copied from retrieved notes. A sentence that is not a
substring of a cited note is a groundedness failure and the call fail-closes.
No network call is made. A hosted model API is out of policy for this path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data" / "synthetic_notes.json"

# Questions about the export rule stay answerable. The export action itself
# is a separate endpoint and still requires an approver.
HIGH_RISK_TERMS = ("prescribe", "diagnosis", "diagnose", "ehr")


class GroundednessError(ValueError):
    pass


def load_notes(path: Path | None = None) -> list[dict]:
    payload = json.loads((path or DEFAULT_CORPUS).read_text(encoding="utf-8"))
    if payload.get("patient_records") != 0:
        raise ValueError("Corpus must declare patient_records = 0")
    return list(payload["notes"])


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9-]+", text.lower()) if len(token) > 2}


def retrieve(question: str, notes: list[dict], limit: int = 2) -> list[dict]:
    asked = _tokens(question)
    scored = []
    for note in notes:
        overlap = asked & _tokens(note["title"] + " " + note["text"])
        if overlap:
            scored.append((len(overlap), note))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [note for _, note in scored[:limit]]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def grounded_answer(question: str, notes: list[dict]) -> dict:
    lowered = question.lower()
    if any(term in lowered for term in HIGH_RISK_TERMS):
        return {
            "status": "approval-required",
            "answer": "",
            "citations": [],
            "reason": "High-risk clinical action. A named approver must authorize it. No answer was generated.",
        }
    selected = retrieve(question, notes)
    if not selected:
        return {
            "status": "refused",
            "answer": "",
            "citations": [],
            "reason": "No synthetic note supports this question. Fail closed.",
        }
    cited = {note["id"]: note["text"] for note in selected}
    chosen_sentences = []
    for note in selected:
        for sentence in _sentences(note["text"]):
            if _tokens(question) & _tokens(sentence):
                chosen_sentences.append(sentence)
    if not chosen_sentences:
        return {
            "status": "refused",
            "answer": "",
            "citations": [],
            "reason": "Retrieved notes did not contain a sentence that overlaps the question.",
        }
    answer = " ".join(chosen_sentences)
    assert_grounded(answer, cited)
    return {
        "status": "answered",
        "answer": answer,
        "citations": [{"id": note["id"], "title": note["title"]} for note in selected],
        "reason": "Extracted from the synthetic corpus. No hosted model was called.",
    }


def assert_grounded(answer: str, cited_texts: dict[str, str]) -> None:
    if not answer.strip():
        raise GroundednessError("empty answer")
    blob = "\n".join(cited_texts.values())
    for sentence in _sentences(answer):
        if sentence not in blob:
            raise GroundednessError("answer sentence is not in a cited note")
