"""FastAPI application for the DICOMweb AI Gateway.

Run with ``uvicorn dicomweb_ai_gateway.main:app`` or ``dicomweb-gateway``.

Endpoints (all under ``/api/v1``):

* ``POST /dicomweb/studies``            — STOW-RS: ingest DICOM instances.
* ``GET  /dicomweb/studies``            — QIDO-RS: list stored studies.
* ``GET  /dicomweb/studies/{study}/metadata`` — DICOMweb instance metadata.
* ``GET  /dicomweb/studies/{study}/instances/{series}/{instance}`` — WADO-RS bytes.
* ``POST /inference/studies/{study}/run`` — run the configured model worker.
* ``GET  /results/studies/{study}``     — JSON structured result.
* ``GET  /results/studies/{study}/sr``  — DICOM SR bytes.
* ``GET  /results/studies/{study}/fhir``— FHIR R4 transaction Bundle.
* ``POST /grounding/verify``            — reconcile findings with a report.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse

from . import __version__
from .grounding import (
    BaseGroundingEngine,
    KeywordGroundingEngine,
    consistency_score,
)
from .inference import InferenceOrchestrator, ReferenceMetadataWorker
from .results import build_fhir_bundle, build_sr, sr_bytes
from .schemas import (
    GroundingRequest,
    GroundingResponse,
    InferenceRequest,
    StoreSummary,
    StudyInferenceResult,
    StudyResult,
)
from .store import DICOMStore, StoreError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dicomweb_ai_gateway")

DEFAULT_STORE_DIR = Path("data/store")


class GatewayState:
    """Holds the gateway's long-lived singletons."""

    def __init__(self, store_dir: Path = DEFAULT_STORE_DIR) -> None:
        self.store = DICOMStore(store_dir)
        self.orchestrator = InferenceOrchestrator(self.store, workers=[ReferenceMetadataWorker()])
        self.grounding: BaseGroundingEngine = KeywordGroundingEngine()


def create_app(state: GatewayState | None = None) -> FastAPI:
    """Application factory. ``state`` is injectable for tests."""
    state = state or GatewayState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("DICOMweb AI Gateway v%s ready (store: %s)", __version__, state.store.root)
        yield

    app = FastAPI(
        title="DICOMweb AI Gateway",
        summary="Self-hosted, standards-compliant AI orchestration for radiology.",
        version=__version__,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "dicomweb", "description": "STOW-RS / WADO-RS / QIDO-RS operations."},
            {"name": "inference", "description": "Model-agnostic inference orchestration."},
            {"name": "results", "description": "Structured results: JSON, DICOM SR, FHIR."},
            {"name": "grounding", "description": "Report-grounding verification."},
        ],
    )

    # ------------------------------------------------------------------
    # Meta / health
    # ------------------------------------------------------------------
    @app.get("/", tags=["meta"])
    def root() -> dict:
        return {
            "service": "DICOMweb AI Gateway",
            "version": __version__,
            "workers": state.orchestrator.workers,
            "grounding_engine": state.grounding.name,
            "store_root": str(state.store.root),
            "docs": "/docs",
        }

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/v1/workers", tags=["meta"])
    def list_workers() -> dict:
        return {"workers": state.orchestrator.workers}

    # ------------------------------------------------------------------
    # DICOMweb (PS3.18)
    # ------------------------------------------------------------------
    @app.post(
        "/api/v1/dicomweb/studies",
        tags=["dicomweb"],
        status_code=status.HTTP_202_ACCEPTED,
        responses={
            202: {"description": "Instances accepted for storage."},
            400: {"description": "No valid DICOM instances in the payload."},
        },
    )
    async def stow_rs(request: Request) -> JSONResponse:
        """STOW-RS: store instances from a ``multipart/related`` body."""
        content_type = request.headers.get("content-type", "")
        body = await request.body()
        try:
            datasets = state.store.parse_multipart(body, content_type)
        except StoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not datasets:
            raise HTTPException(status_code=400, detail="No application/dicom parts found in payload")

        stored = state.store.store_instances(datasets)
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "stored_instances": len(stored),
                "referencedSOPSequence": [
                    {"00081190": {"vr": "UR", "Value": [f"/api/v1/dicomweb/studies/{s.study_uid}/instances/{s.series_uid}/{s.sop_uid}"]}}
                    for s in stored
                ],
            },
        )

    @app.post("/api/v1/dicomweb/upload", tags=["dicomweb"])
    async def dicom_upload(file: UploadFile) -> dict:
        """Convenience endpoint: ingest a single ``.dcm`` file upload."""
        payload = await file.read()
        try:
            datasets = state.store.parse_multipart(payload, "application/dicom")
        except StoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        stored = state.store.store_instances(datasets)
        return {"status": "accepted", "stored_instances": len(stored)}

    @app.get("/api/v1/dicomweb/studies", tags=["dicomweb"], response_model=StoreSummary)
    def qido_rs() -> StoreSummary:
        """QIDO-RS: lightweight list of stored studies."""
        return StoreSummary(studies=state.store.list_studies(), total_instances=sum(s["instances"] for s in state.store.list_studies()))

    @app.get("/api/v1/dicomweb/studies/{study}/metadata", tags=["dicomweb"])
    def study_metadata(study: str) -> dict:
        """Return DICOMweb-style instance metadata for a study."""
        instances = state.store.instances_for_study(study)
        if not instances:
            raise HTTPException(status_code=404, detail=f"Study {study} not found")
        return {
            "studyInstanceUID": study,
            "instances": [inst.metadata() for inst in instances],
        }

    @app.get("/api/v1/dicomweb/studies/{study}/instances/{series}/{instance}", tags=["dicomweb"])
    def wado_rs(study: str, series: str, instance: str) -> Response:
        """WADO-RS: return the original DICOM instance bytes."""
        try:
            payload, _ = state.store.get_instance_bytes(instance)
        except StoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=payload, media_type="application/dicom")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @app.post("/api/v1/inference/studies/{study}/run", tags=["inference"], response_model=StudyInferenceResult)
    def run_inference(study: str, body: InferenceRequest | None = None) -> StudyInferenceResult:
        """Run inference over every stored instance of a study."""
        try:
            return state.orchestrator.run_study(study, worker_name=body.worker if body else None)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    @app.get("/api/v1/results/studies/{study}", tags=["results"], response_model=StudyResult)
    def get_study_result(study: str) -> StudyResult:
        """Return the structured JSON result for a study."""
        _ensure_study(study, state)
        return StudyResult(
            study_instance_uid=study,
            json_url=f"/api/v1/results/studies/{study}",
            sr_url=f"/api/v1/results/studies/{study}/sr",
            fhir_bundle=build_fhir_bundle(
                state.orchestrator.run_study(study),
                base_url="",
            ),
        )

    @app.get("/api/v1/results/studies/{study}/sr", tags=["results"])
    def get_study_sr(study: str) -> Response:
        """Return a DICOM SR summarizing the study's findings."""
        _ensure_study(study, state)
        result = state.orchestrator.run_study(study)
        sr = build_sr(f"1.2.826.0.1.3680043.2.1125.{uuid.uuid4().int & (2**63 - 1)}", result)
        return Response(
            content=sr_bytes(sr),
            media_type="application/dicom",
            headers={"Content-Disposition": f'attachment; filename="sr-{study}.dcm"'},
        )

    @app.get("/api/v1/results/studies/{study}/fhir", tags=["results"])
    def get_study_fhir(study: str) -> dict:
        """Return an HL7 FHIR R4 transaction Bundle for the study."""
        _ensure_study(study, state)
        result = state.orchestrator.run_study(study)
        return build_fhir_bundle(result, base_url="")

    # ------------------------------------------------------------------
    # Grounding
    # ------------------------------------------------------------------
    @app.post("/api/v1/grounding/verify", tags=["grounding"], response_model=GroundingResponse)
    def verify_grounding(body: GroundingRequest) -> GroundingResponse:
        """Reconcile findings against a radiologist's report."""
        if body.findings is None:
            if not body.study_instance_uid:
                raise HTTPException(status_code=422, detail="Provide findings or a study_instance_uid")
            _ensure_study(body.study_instance_uid, state)
            result = state.orchestrator.run_study(body.study_instance_uid)
            findings = [f for inst in result.instances for f in inst.findings]
        else:
            findings = body.findings

        verdicts = state.grounding.verify(findings, body.report_text)
        return GroundingResponse(
            study_instance_uid=body.study_instance_uid,
            verdicts=verdicts,
            consistency_score=consistency_score(verdicts),
        )

    return app


def _ensure_study(study: str, state: GatewayState) -> None:
    if not state.store.instances_for_study(study):
        raise HTTPException(status_code=404, detail=f"Study {study} not found")


def main() -> None:  # pragma: no cover — CLI entry point
    """Run the gateway with uvicorn (``dicomweb-gateway``)."""
    import uvicorn

    uvicorn.run("dicomweb_ai_gateway.main:app", host="127.0.0.1", port=8000, reload=False)


app = create_app()
