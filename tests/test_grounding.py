"""Tests for the report-grounding engines."""

from __future__ import annotations

from dicomweb_ai_gateway.grounding import KeywordGroundingEngine, consistency_score
from dicomweb_ai_gateway.schemas import Finding


def _finding(code_value: str, meaning: str, value: str) -> Finding:
    return Finding(code_value=code_value, code_meaning=meaning, value=value)


def test_matched_when_report_mentions_finding():
    engine = KeywordGroundingEngine()
    findings = [_finding("24028007", "Structure of chest", "CHEST")]
    verdicts = engine.verify(findings, "The chest is unremarkable.")
    assert verdicts[0].verdict == "matched"


def test_contradicted_when_report_says_normal():
    engine = KeywordGroundingEngine()
    findings = [_finding("12738006", "Structure of brain", "BRAIN"), _finding("18803008", "Structure of imaging region is normal", "12.3")]
    verdicts = engine.verify(findings, "No evidence of acute abnormality. Unremarkable study.")
    # A "normal" finding agrees with an unremarkable report → matched.
    # A brain-structure finding has no support and the report denies findings → contradicted.
    by_code = {v.code_value: v for v in verdicts}
    assert by_code["18803008"].verdict == "matched"
    assert by_code["12738006"].verdict == "contradicted"


def test_consistency_score():
    engine = KeywordGroundingEngine()
    findings = [_finding("24028007", "Structure of chest", "CHEST")]
    matched = engine.verify(findings, "chest shows stable disease")
    assert consistency_score(matched) == 1.0
    unmatched = engine.verify(findings, "abdomen is clear")
    assert consistency_score(unmatched) == 0.0


def test_llm_engine_falls_back_without_server():
    from dicomweb_ai_gateway.grounding import LLMGroundingEngine

    engine = LLMGroundingEngine(base_url="http://127.0.0.1:1/v1")
    findings = [_finding("24028007", "Structure of chest", "CHEST")]
    # Connection refused → degrades to keyword engine, no exception.
    verdicts = engine.verify(findings, "The chest is unremarkable.")
    assert verdicts[0].verdict in {"matched", "unsupported", "contradicted"}
