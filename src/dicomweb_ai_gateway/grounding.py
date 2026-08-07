"""Report-grounding and verification.

The gateway's trust layer: reconcile machine-produced structured findings
against the radiologist's free-text report and flag discrepancies.  Each
finding gets one of:

* ``matched``     — the report supports the finding,
* ``unsupported`` — the report is silent about the finding,
* ``contradicted``— the report explicitly disagrees with the finding.

Two engines are provided:

* :class:`KeywordGroundingEngine` — deterministic, offline, zero-dependency.
  It tokenises the report and the finding and scores lexical overlap.  This is
  the default so the gateway works with no LLM at all.
* :class:`LLMGroundingEngine` — calls a local, Ollama-compatible endpoint
  (the paper's "locally hosted LLM") for nuanced reconciliation when available.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import ClassVar

from .schemas import Finding, GroundingVerdict

logger = logging.getLogger(__name__)

# "normal"/"unremarkable" are deliberately NOT stopwords — they are clinically
# meaningful for the normality-detection branch.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "of", "in", "on", "for", "to",
    "is", "are", "was", "were", "shows", "shown", "there", "no",
}


class BaseGroundingEngine(ABC):
    """Contract for report-grounding engines."""

    name: str = "base"

    @abstractmethod
    def verify(self, findings: list[Finding], report_text: str) -> list[GroundingVerdict]:
        """Return one verdict per finding."""


class KeywordGroundingEngine(BaseGroundingEngine):
    """Offline lexical grounding: token overlap between finding and report.

    Verdict semantics (documented so behaviour is auditable):

    * ``matched``       — (a) the finding's concept tokens appear in the report
                          and the concept is not itself a "normal" claim, or
                          (b) the report is unremarkable and the finding claims
                          normality (the two agree).
    * ``contradicted``  — (a) a negation word sits within two words of a concept
                          token (explicit denial), or (b) the whole report is
                          "unremarkable"/"no evidence" while the finding claims a
                          structure/abnormality (discrepancy).
    * ``unsupported``   — otherwise (the report is simply silent about it).
    """

    name = "keyword"

    #: Negation words that, adjacent to a concept token, mark a contradiction.
    _NEGATIONS: ClassVar[set[str]] = {"no", "not", "without", "absent", "negative"}
    #: Tokens that indicate a finding claims normality (not an abnormality).
    _NORMAL: ClassVar[set[str]] = {"normal", "unremarkable"}
    #: Phrases that mark the entire report as unremarkable.
    _UNREMARKABLE: ClassVar[tuple[str, ...]] = ("unremarkable", "no evidence", "no abnormality")

    def verify(self, findings: list[Finding], report_text: str) -> list[GroundingVerdict]:
        report_tokens = self._tokens(report_text)
        # Punctuation-stripped word list for the negation-window scan.
        plain = re.findall(r"[a-z0-9]+", report_text.lower())
        lowered = report_text.lower()
        report_unremarkable = "unremarkable" in report_tokens or any(p in lowered for p in ("no evidence", "no abnormality"))

        verdicts: list[GroundingVerdict] = []
        for f in findings:
            concept = self._tokens(f.code_meaning) | self._tokens(f.value)
            overlap = concept & report_tokens

            if overlap and not (concept & self._NORMAL):
                verdict = "matched"
                explanation = f"Report mentions: {', '.join(sorted(overlap))}."
            elif report_unremarkable and (concept & self._NORMAL):
                verdict = "matched"
                explanation = "Report is unremarkable and this finding claims normality; they agree."
            elif self._negated(concept, plain):
                verdict = "contradicted"
                explanation = "Report explicitly denies this finding."
            elif report_unremarkable and not (concept & self._NORMAL):
                verdict = "contradicted"
                explanation = "Report is unremarkable but this finding claims an abnormality or structure."
            else:
                verdict = "unsupported"
                explanation = "Report is silent about this finding."

            verdicts.append(
                GroundingVerdict(
                    code_value=f.code_value,
                    code_meaning=f.code_meaning,
                    verdict=verdict,
                    explanation=explanation,
                )
            )
        return verdicts

    @staticmethod
    def _negated(concept: set[str], plain: list[str]) -> bool:
        """True if a negation word appears within two tokens of a concept token."""
        for i, word in enumerate(plain):
            if word in KeywordGroundingEngine._NEGATIONS:
                window = set(plain[i + 1 : i + 4])
                if concept & window:
                    return True
        return False

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


class LLMGroundingEngine(BaseGroundingEngine):
    """Grounding via a local LLM speaking the OpenAI-compatible chat API.

    ``base_url`` should point at a local Ollama instance
    (``http://localhost:11434/v1``) or any OpenAI-compatible endpoint.  The
    model is prompted with the findings and the report and asked to return one
    JSON verdict per finding; any failure degrades to :class:`KeywordGroundingEngine`.
    """

    name = "llm"

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434/v1") -> None:
        self.model = model
        self.base_url = base_url
        self._client = None
        self._fallback = KeywordGroundingEngine()

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install the 'llm' extra (openai) to use the LLM grounding engine") from exc
            self._client = OpenAI(base_url=self.base_url, api_key="ollama")
        return self._client

    def verify(self, findings: list[Finding], report_text: str) -> list[GroundingVerdict]:
        try:
            return self._verify_llm(findings, report_text)
        except Exception:
            logger.warning("LLM grounding failed; falling back to keyword engine", exc_info=True)
            return self._fallback.verify(findings, report_text)

    def _verify_llm(self, findings: list[Finding], report_text: str) -> list[GroundingVerdict]:
        prompt = self._build_prompt(findings, report_text)
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You reconcile AI findings with a radiology report. "
                 "Return ONLY JSON: a list of {\"code_value\", \"verdict\", \"explanation\"} where verdict is "
                 "matched | unsupported | contradicted."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        import json

        parsed = json.loads(resp.choices[0].message.content)
        by_code = {f.code_value: f for f in findings}
        verdicts: list[GroundingVerdict] = []
        for item in parsed:
            f = by_code.get(str(item.get("code_value")))
            if f is None:
                continue
            verdicts.append(
                GroundingVerdict(
                    code_value=f.code_value,
                    code_meaning=f.code_meaning,
                    verdict=item.get("verdict", "unsupported"),
                    explanation=str(item.get("explanation", "")),
                )
            )
        # Ensure every finding has a verdict even if the model dropped one.
        covered = {v.code_value for v in verdicts}
        for f in findings:
            if f.code_value not in covered:
                verdicts.append(
                    GroundingVerdict(code_value=f.code_value, code_meaning=f.code_meaning, verdict="unsupported", explanation="No verdict returned by LLM.")
                )
        return verdicts

    @staticmethod
    def _build_prompt(findings: list[Finding], report_text: str) -> str:
        lines = [f"- {f.code_value} ({f.code_meaning}): value={f.value!r}" for f in findings]
        return "FINDINGS:\n" + "\n".join(lines) + f"\n\nREPORT:\n{report_text}"


def consistency_score(verdicts: list[GroundingVerdict]) -> float:
    """Fraction of findings that matched the report (0..1)."""
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v.verdict == "matched") / len(verdicts)
