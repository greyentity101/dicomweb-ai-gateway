"""Tests for inference workers and the orchestrator."""

from __future__ import annotations

import pytest
from pydicom import uid

from dicomweb_ai_gateway.inference import (
    InferenceOrchestrator,
    ReferenceMetadataWorker,
    TorchImageClassifierWorker,
)
from dicomweb_ai_gateway.store import DICOMStore
from tests.conftest import make_dcm


def test_reference_worker_emits_coded_findings():
    ds = make_dcm(modality="CT", body_part="CHEST")
    findings = ReferenceMetadataWorker().predict(ds)

    meanings = {f.code_meaning for f in findings}
    assert "Structure of chest" in meanings
    assert all(0.0 <= f.confidence <= 1.0 for f in findings)
    # Deterministic across calls.
    assert findings == ReferenceMetadataWorker().predict(ds)


def test_reference_worker_without_body_part():
    ds = make_dcm(body_part="")
    findings = ReferenceMetadataWorker().predict(ds)
    # No chest region finding when BodyPartExamined is absent.
    assert not any("Structure of chest" in f.code_meaning for f in findings)
    # Signal + acquisition findings still present.
    assert len(findings) >= 2


def test_orchestrator_runs_all_instances(tmp_path):
    store = DICOMStore(tmp_path / "store")
    study = uid.generate_uid()
    store.store_instances([make_dcm(study_uid=study) for _ in range(3)])

    orch = InferenceOrchestrator(store)
    result = orch.run_study(study)
    assert result.status == "completed"
    assert result.processed_count == 3
    assert result.total_count == 3
    assert all(i.worker == "reference-metadata" for i in result.instances)


def test_orchestrator_unknown_worker(tmp_path):
    store = DICOMStore(tmp_path / "store")
    study = uid.generate_uid()
    store.store_instances([make_dcm(study_uid=study)])
    orch = InferenceOrchestrator(store)
    with pytest.raises(KeyError, match="Unknown worker"):
        orch.run_study(study, worker_name="does-not-exist")


def test_orchestrator_empty_study(tmp_path):
    store = DICOMStore(tmp_path / "store")
    orch = InferenceOrchestrator(store)
    with pytest.raises(KeyError):
        orch.run_study(uid.generate_uid())


def test_torch_worker_requires_torch(tmp_path):
    worker = TorchImageClassifierWorker(checkpoint_path=tmp_path / "x.pt", labels=["a", "b"])
    with pytest.raises(RuntimeError, match="PyTorch"):
        worker.predict(make_dcm())
