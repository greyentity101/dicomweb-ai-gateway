"""Pydantic schemas shared across the gateway's HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """A single machine-produced finding for one DICOM instance."""

    code_value: str = Field(..., description="DICOM SR coded-entry code value (e.g. DCM/SNOMED code).")
    code_meaning: str = Field(..., description="Human-readable meaning of the coded entry.")
    coding_scheme: str = Field(default="DCM", description="Coding scheme designator (DCM, SCT, ...).")
    value: str = Field(..., description="Measured or qualitative value for the finding.")
    unit: str | None = Field(default=None, description="UCUM unit when the value is a measurement.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Model confidence in [0, 1].")
    provenance: str = Field(default="reference-worker", description="Which worker produced this finding.")


class InstanceResult(BaseModel):
    """Inference output for a single DICOM instance."""

    sop_instance_uid: str
    modality: str
    body_part_examined: str | None = None
    series_instance_uid: str
    study_instance_uid: str
    findings: list[Finding] = Field(default_factory=list)
    worker: str = Field(..., description="Name of the model worker that produced the result.")


class StudyInferenceResult(BaseModel):
    """Complete inference result for a study (all instances)."""

    study_instance_uid: str
    status: Literal["completed", "partial", "error"]
    instances: list[InstanceResult] = Field(default_factory=list)
    worker: str
    processed_count: int = 0
    total_count: int = 0


class InferenceRequest(BaseModel):
    """Request body for /inference/run — lets callers override the worker."""

    worker: str | None = Field(
        default=None,
        description="Worker name to use. Defaults to the gateway-configured worker.",
    )


# ---------------------------------------------------------------------------
# Report grounding
# ---------------------------------------------------------------------------


class GroundingVerdict(BaseModel):
    """Verdict reconciling one machine finding against a radiologist's report."""

    code_value: str
    code_meaning: str
    verdict: Literal["matched", "unsupported", "contradicted"]
    explanation: str = Field(..., description="Why the finding got this verdict.")


class GroundingRequest(BaseModel):
    """Request body for /grounding/verify."""

    study_instance_uid: str | None = Field(
        default=None,
        description="Study whose stored findings should be verified (either this or findings).",
    )
    report_text: str = Field(..., description="Free-text radiology report produced by the human reader.")
    findings: list[Finding] | None = Field(
        default=None,
        description="Explicit findings to verify. If omitted, the study's stored findings are used.",
    )


class GroundingResponse(BaseModel):
    """Per-finding verdicts plus an aggregate consistency score."""

    study_instance_uid: str | None = None
    verdicts: list[GroundingVerdict]
    consistency_score: float = Field(..., ge=0.0, le=1.0, description="Fraction of findings matched to the report.")


# ---------------------------------------------------------------------------
# Results (DICOM SR + FHIR)
# ---------------------------------------------------------------------------


class StudyResult(BaseModel):
    """Full structured result bundle for a study."""

    study_instance_uid: str
    json_url: str
    sr_url: str | None = None
    sr_sop_instance_uid: str | None = None
    fhir_bundle: dict[str, Any] = Field(default_factory=dict, description="FHIR Bundle of DiagnosticReport/Observation.")


class StoreSummary(BaseModel):
    """Summary of the local DICOM store for an endpoint response."""

    studies: list[dict[str, Any]]
    total_instances: int = 0
