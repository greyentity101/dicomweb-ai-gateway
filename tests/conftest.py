"""Shared fixtures: synthetic DICOM datasets and an in-memory gateway client."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydicom import Dataset, uid

from dicomweb_ai_gateway.main import GatewayState, create_app


def make_dcm(
    *,
    study_uid: str | None = None,
    series_uid: str | None = None,
    sop_uid: str | None = None,
    modality: str = "CT",
    body_part: str = "CHEST",
    rows: int = 16,
    cols: int = 16,
    patient_name: str = "TEST^PATIENT",
    patient_id: str = "TEST-001",
) -> Dataset:
    """Build a minimal, valid, pixel-bearing CT DICOM dataset."""
    ds = Dataset()
    ds.StudyInstanceUID = study_uid or uid.generate_uid()
    ds.SeriesInstanceUID = series_uid or uid.generate_uid()
    ds.SOPInstanceUID = sop_uid or uid.generate_uid()
    ds.SOPClassUID = uid.CTImageStorage
    ds.Modality = modality
    ds.BodyPartExamined = body_part
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.StudyDate = "20260807"
    ds.StudyDescription = "Gateway test study"

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = (b"\x00\x00" * (rows * cols))[: rows * cols * 2]

    ds.file_meta = Dataset()
    ds.file_meta.TransferSyntaxUID = uid.ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.ImplementationClassUID = uid.PYDICOM_IMPLEMENTATION_UID
    return ds


def dicom_bytes(ds: Dataset) -> bytes:
    """Serialize a dataset (with file meta) to bytes."""
    from dicomweb_ai_gateway.store import dataset_bytes

    return dataset_bytes(ds)


@pytest.fixture()
def gateway_state(tmp_path: Path) -> GatewayState:
    """A gateway state backed by a temp store directory."""
    return GatewayState(store_dir=tmp_path / "store")


@pytest.fixture()
def client(gateway_state: GatewayState) -> TestClient:
    """FastAPI TestClient wired to the temp gateway state."""
    return TestClient(create_app(gateway_state))
