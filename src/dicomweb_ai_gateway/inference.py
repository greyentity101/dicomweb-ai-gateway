"""Model-agnostic inference workers for the gateway.

The gateway never couples to a single model.  Every model lives behind a
``BaseModelWorker``; the orchestrator routes a study's instances to the
configured worker and normalises the output into ``StudyInferenceResult``.

Two workers ship by default:

* :class:`ReferenceMetadataWorker` — a zero-weight, offline reference worker
  that turns DICOM header metadata into coded findings.  It exists so the
  gateway is fully runnable on a CPU-only, no-dependency stack out of the box.
* :class:`TorchImageClassifierWorker` — a documented template for dropping in a
  real PyTorch model (CNN/U-Net) per the paper's "model-agnostic" contract.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from pydicom import Dataset

from .schemas import Finding, InstanceResult, StudyInferenceResult
from .store import DICOMStore

logger = logging.getLogger(__name__)


# Coding scheme designators used in generated coded entries.
class CodedEntry:
    """Small namespace of coded entries used by the reference worker."""

    # SNOMED-CT body structure codes (subset used by the reference worker).
    BODY_PART: ClassVar[dict[str, tuple[str, str, str]]] = {
        "CHEST": ("302550000", "SCT", "Computed tomography of chest"),
        "ABDOMEN": ("818981001", "SCT", "Computed tomography of abdomen"),
        "BRAIN": ("54917003", "SCT", "Computed tomography of head"),
        "SPINE": ("84028004", "SCT", "Computed tomography of spine"),
        "PELVIS": ("70028004", "SCT", "Computed tomography of pelvis"),
        "EXTREMITY": ("87443008", "SCT", "Computed tomography of extremity"),
    }
    # DICOM body part examined → SNOMED code for the region finding.
    REGION_FINDING: ClassVar[dict[str, tuple[str, str, str]]] = {
        "CHEST": ("24028007", "SCT", "Structure of chest"),
        "ABDOMEN": ("818981001", "SCT", "Structure of abdomen"),
        "BRAIN": ("12738006", "SCT", "Structure of brain"),
        "SPINE": ("421060004", "SCT", "Structure of spine"),
        "PELVIS": ("12921003", "SCT", "Structure of pelvis"),
        "EXTREMITY": ("44043000", "SCT", "Structure of extremity"),
    }
    ACQUISITION: ClassVar[tuple[str, str, str]] = ("19605001", "SCT", "Acquisition technique")


class BaseModelWorker(ABC):
    """Contract every model worker must satisfy."""

    name: str = "base"

    @abstractmethod
    def predict(self, dataset: Dataset) -> list[Finding]:
        """Return the findings for a single DICOM instance."""

    def describe(self) -> dict:
        """Metadata about the worker for the API root/debug endpoint."""
        return {"name": self.name, "kind": type(self).__name__}


class ReferenceMetadataWorker(BaseModelWorker):
    """Zero-weight reference worker: derives coded findings from DICOM headers.

    This is not a clinical model.  It produces deterministically valid coded
    entries so that the full pipeline (store → inference → SR/FHIR → grounding)
    can be exercised end to end offline.  Swap in a real model by implementing
    :class:`BaseModelWorker`.
    """

    name = "reference-metadata"

    def predict(self, dataset: Dataset) -> list[Finding]:
        findings: list[Finding] = []

        modality = str(getattr(dataset, "Modality", "OT") or "OT").upper()
        body_part = str(getattr(dataset, "BodyPartExamined", "") or "").upper()

        # Region finding derived from BodyPartExamined (if present).
        if body_part in CodedEntry.REGION_FINDING:
            cv, scheme, cm = CodedEntry.REGION_FINDING[body_part]
            findings.append(
                Finding(
                    code_value=cv,
                    code_meaning=cm,
                    coding_scheme=scheme,
                    value=body_part,
                    confidence=1.0,
                    provenance=self.name,
                )
            )

        # A deterministic, pseudo-objective "signal" so SR/FHIR values vary per
        # image without pulling in a real model: hash of the pixel data.
        pixel_signal = self._pixel_signal(dataset)

        findings.append(
            Finding(
                code_value="18803008",
                code_meaning="Structure of imaging region is normal",
                coding_scheme="SCT",
                value=f"{pixel_signal:.1f}",
                unit="{Signal}",
                confidence=0.9,
                provenance=self.name,
            )
        )

        # Acquisition technique (modality-derived).
        findings.append(
            Finding(
                code_value=CodedEntry.ACQUISITION[0],
                code_meaning=CodedEntry.ACQUISITION[2],
                coding_scheme=CodedEntry.ACQUISITION[1],
                value=modality,
                confidence=1.0,
                provenance=self.name,
            )
        )
        return findings

    @staticmethod
    def _pixel_signal(dataset: Dataset) -> float:
        """Deterministic summary statistic of the pixel data (0..255 scaled).

        Falls back to a hash of the dataset bytes when pixel data is absent so
        the worker stays total and reproducible.
        """
        try:
            import hashlib

            pixels = dataset.pixel_array
            return float(int(hashlib.sha256(pixels.tobytes()).hexdigest(), 16) % 10_000) / 100.0
        except Exception:  # noqa: BLE001 — pixel data may be missing/encoded
            return float(dataset.__len__() % 255) / 2.55


class TorchImageClassifierWorker(BaseModelWorker):
    """Template worker that runs a real PyTorch model against pixel data.

    This is a *template*: the ``checkpoint_path`` must point at a model whose
    ``forward`` accepts a batched ``NCHW`` float tensor and returns logits of
    shape ``(N, num_classes)``.  Wire your own ``preprocess``/``labels`` to the
    model's expected input domain.
    """

    name = "torch-classifier"

    def __init__(self, checkpoint_path: str | Path, labels: list[str], device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.labels = labels
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                import torch  # imported lazily so CPU-only installs work
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("PyTorch is required for the TorchImageClassifierWorker") from exc
            self._model = torch.load(self.checkpoint_path, map_location=self.device)
            self._model.eval()
        return self._model

    def predict(self, dataset: Dataset) -> list[Finding]:
        model = self._load()
        logits = model(self.preprocess(dataset))
        label_index = int(logits.argmax(dim=1).item())
        prob = float(logits.softmax(dim=1).max().item())
        return [
            Finding(
                code_value="181226008",
                code_meaning="Imaging finding",
                coding_scheme="SCT",
                value=self.labels[label_index],
                confidence=prob,
                provenance=self.name,
            )
        ]

    def preprocess(self, dataset: Dataset):
        """Convert a DICOM pixel array into a batched float tensor."""
        import torch

        arr = dataset.pixel_array.astype("float32")
        # Normalise to [0, 1] and make grayscale channels-first, batched.
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-8)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)


class InferenceOrchestrator:
    """Routes stored studies to a worker and normalises results."""

    def __init__(self, store: DICOMStore, workers: Iterable[BaseModelWorker] | None = None) -> None:
        self.store = store
        self._workers: dict[str, BaseModelWorker] = {}
        for worker in workers or [ReferenceMetadataWorker()]:
            self.register(worker)

    def register(self, worker: BaseModelWorker) -> None:
        """Register (or replace) a worker by its ``name``."""
        self._workers[worker.name] = worker

    @property
    def workers(self) -> list[dict]:
        return [w.describe() for w in self._workers.values()]

    def default_worker(self) -> BaseModelWorker:
        return next(iter(self._workers.values()))

    def run_study(self, study_uid: str, worker_name: str | None = None) -> StudyInferenceResult:
        instances = self.store.instances_for_study(study_uid)
        if not instances:
            raise KeyError(f"No stored instances for study {study_uid}")

        worker = self._workers.get(worker_name or "", self.default_worker())
        if worker_name and worker_name not in self._workers:
            raise KeyError(f"Unknown worker {worker_name!r}; available: {sorted(self._workers)}")

        results: list[InstanceResult] = []
        processed = 0
        status = "completed"
        for inst in instances:
            try:
                findings = worker.predict(inst.dataset)
                processed += 1
            except Exception:
                logger.warning("Inference failed for %s", inst.sop_uid, exc_info=True)
                status = "partial"
                findings = []
            results.append(
                InstanceResult(
                    sop_instance_uid=inst.sop_uid,
                    modality=str(getattr(inst.dataset, "Modality", "OT")),
                    body_part_examined=str(getattr(inst.dataset, "BodyPartExamined", "") or None),
                    series_instance_uid=inst.series_uid,
                    study_instance_uid=study_uid,
                    findings=findings,
                    worker=worker.name,
                )
            )
        return StudyInferenceResult(
            study_instance_uid=study_uid,
            status=status,
            instances=results,
            worker=worker.name,
            processed_count=processed,
            total_count=len(instances),
        )

    @staticmethod
    def tidy_number(x: float) -> float:
        """Round to a tidy value (used to keep generated values clean)."""
        if not math.isfinite(x):
            return 0.0
        return round(x, 2)
