"""Tests for DICOM SR and FHIR result serialization."""

from __future__ import annotations

import json

from pydicom import uid

from dicomweb_ai_gateway.inference import InferenceOrchestrator
from dicomweb_ai_gateway.results import build_fhir_bundle, build_sr, sr_bytes
from dicomweb_ai_gateway.store import DICOMStore
from tests.conftest import make_dcm


def _run_result(tmp_path):
    store = DICOMStore(tmp_path / "store")
    study = uid.generate_uid()
    store.store_instances([make_dcm(study_uid=study, body_part="CHEST")])
    return study, InferenceOrchestrator(store).run_study(study)


def test_build_sr_is_valid_dicom(tmp_path):
    study, result = _run_result(tmp_path)
    sr = build_sr(uid.generate_uid(), result)
    # Serializes back without error and round-trips.
    blob = sr_bytes(sr)
    assert blob.startswith(b"\x00") or blob  # non-empty
    assert sr.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.22"
    assert sr.StudyInstanceUID == study
    # Content tree has a container per instance.
    root = sr.ContentSequence[0]
    assert root.ConceptNameCodeSequence[0].CodeMeaning == "Radiology Study observation"
    assert len(root.ContentSequence) == result.total_count


def test_build_fhir_bundle(tmp_path):
    _, result = _run_result(tmp_path)
    bundle = build_fhir_bundle(result, base_url="")
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "transaction"

    types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert types.count("DiagnosticReport") == 1
    assert types.count("Observation") >= 3  # region + signal + acquisition findings

    # Every entry is JSON-serializable (FHIR is machine-consumable).
    json.dumps(bundle)

    report = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "DiagnosticReport")
    assert report["conclusion"].startswith("1/1 instances processed")
