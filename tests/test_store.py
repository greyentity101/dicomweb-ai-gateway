"""Tests for the DICOM store and STOW-RS parsing."""

from __future__ import annotations

import pytest
from pydicom import uid

from dicomweb_ai_gateway.store import DICOMStore, StoreError
from tests.conftest import dicom_bytes, make_dcm


def test_store_roundtrip(tmp_path):
    store = DICOMStore(tmp_path / "store")
    ds = make_dcm()
    stored = store.store_instances([ds])

    assert len(stored) == 1
    record = stored[0]
    assert record.study_uid == ds.StudyInstanceUID
    assert record.path.exists()

    # Retrieval by SOP UID returns the same bytes.
    payload, found = store.get_instance_bytes(ds.SOPInstanceUID)
    assert payload == dicom_bytes(ds)
    assert found.sop_uid == ds.SOPInstanceUID

    # Study listing sees the stored study.
    studies = store.list_studies()
    assert any(s["StudyInstanceUID"] == ds.StudyInstanceUID for s in studies)


def test_store_rejects_missing_required_tags(tmp_path):
    store = DICOMStore(tmp_path / "store")
    ds = make_dcm()
    del ds.StudyInstanceUID
    with pytest.raises(StoreError, match="required tags"):
        store.store_instances([ds])


def test_store_rejects_invalid_uid(tmp_path):
    store = DICOMStore(tmp_path / "store")
    ds = make_dcm()
    ds.StudyInstanceUID = "not a uid!!"
    with pytest.raises(StoreError, match="Invalid UID"):
        store.store_instances([ds])


def test_parse_multipart_roundtrip(tmp_path):
    store = DICOMStore(tmp_path / "store")
    ds1 = make_dcm()
    ds2 = make_dcm(modality="MR", body_part="BRAIN")

    boundary = "example-boundary-123"
    parts = b"".join(
        b"--" + boundary.encode() + b"\r\n"
        b"Content-Type: application/dicom\r\n"
        b"Content-Transfer-Encoding: binary\r\n\r\n"
        + dicom_bytes(d) + b"\r\n"
        for d in (ds1, ds2)
    )
    body = parts + b"--" + boundary.encode() + b"--\r\n"

    datasets = store.parse_multipart(body, f'multipart/related; type="application/dicom"; boundary={boundary}')
    assert len(datasets) == 2
    assert {d.Modality for d in datasets} == {"CT", "MR"}

    # Non-DICOM parts are skipped.
    mixed = (
        b"--" + boundary.encode() + b"\r\nContent-Type: application/dicom\r\n\r\n" + dicom_bytes(ds1) + b"\r\n"
        b"--" + boundary.encode() + b"\r\nContent-Type: text/plain\r\n\r\nignored\r\n"
        b"--" + boundary.encode() + b"--\r\n"
    )
    datasets = store.parse_multipart(mixed, f"multipart/related; boundary={boundary}")
    assert len(datasets) == 1


def test_store_multiple_instances_same_study(tmp_path):
    store = DICOMStore(tmp_path / "store")
    study = uid.generate_uid()
    series = uid.generate_uid()
    d1 = make_dcm(study_uid=study, series_uid=series)
    d2 = make_dcm(study_uid=study, series_uid=series)
    store.store_instances([d1, d2])

    instances = store.instances_for_study(study)
    assert len(instances) == 2
