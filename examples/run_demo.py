"""End-to-end demo: synthesize a DICOM study and run the whole gateway pipeline.

Exercises the HTTP API exactly as a remote client would:
STOW-RS ingest → inference → JSON/SR/FHIR results → report grounding.

Usage::

    python examples/run_demo.py

Run from the repository root (the gateway is imported in-process).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from pydicom import Dataset
from pydicom import uid as pydicom_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dicomweb_ai_gateway.main import create_app


def make_dcm(body_part: str, modality: str = "CT", signal: int = 512, study_uid: str | None = None) -> Dataset:
    """A tiny synthetic grayscale DICOM instance (all share one study by default)."""
    ds = Dataset()
    ds.StudyInstanceUID = study_uid or pydicom_uid.generate_uid()
    ds.SeriesInstanceUID = pydicom_uid.generate_uid()
    ds.SOPInstanceUID = pydicom_uid.generate_uid()
    ds.SOPClassUID = pydicom_uid.CTImageStorage
    ds.Modality = modality
    ds.BodyPartExamined = body_part
    ds.PatientName = "DEMO^PATIENT"
    ds.PatientID = "DEMO-001"
    ds.StudyDate = "20260807"
    ds.StudyDescription = "Gateway demo study"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = ds.Columns = 32
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = (signal.to_bytes(2, "little") * (32 * 32))
    ds.file_meta = Dataset()
    ds.file_meta.TransferSyntaxUID = pydicom_uid.ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    return ds


def stow_body(datasets: list[Dataset], boundary: str = "demo-boundary") -> tuple[bytes, str]:
    """Build a multipart/related STOW-RS payload from the datasets."""
    parts = []
    for ds in datasets:
        buf = io.BytesIO()
        from pydicom import dcmwrite

        dcmwrite(buf, ds, enforce_file_format=True)
        parts.append(
            f"--{boundary}\r\nContent-Type: application/dicom\r\n\r\n".encode()
            + buf.getvalue()
            + b"\r\n"
        )
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f'multipart/related; type="application/dicom"; boundary={boundary}'


def main() -> int:
    client = TestClient(create_app())
    print("== DICOMweb AI Gateway demo ==")
    print(f"  service: {client.get('/').json()['service']}\n")

    # 1) Synthesize a 3-instance chest CT study and STOW it.
    study_uid = pydicom_uid.generate_uid()
    studies = [make_dcm("CHEST", signal=700 + i * 17, study_uid=study_uid) for i in range(3)]
    body, content_type = stow_body(studies)
    resp = client.post("/api/v1/dicomweb/studies", content=body, headers={"Content-Type": content_type})
    print(f"[1] STOW-RS ingest      -> {resp.status_code}, {resp.json()['stored_instances']} instance(s)")

    # 2) Inference over the study.
    result = client.post(f"/api/v1/inference/studies/{study_uid}/run").json()
    print(f"[2] Inference            -> status={result['status']}, {result['processed_count']}/{result['total_count']} instances, worker='{result['worker']}'")
    print(f"      findings/instance: {len(result['instances'][0]['findings'])}")

    # 3) Structured results.
    client.get(f"/api/v1/results/studies/{study_uid}").raise_for_status()
    sr_res = client.get(f"/api/v1/results/studies/{study_uid}/sr")
    fhir = client.get(f"/api/v1/results/studies/{study_uid}/fhir").json()
    print(f"[3] Results              -> JSON ok | DICOM SR {len(sr_res.content)} bytes | FHIR bundle ({fhir['type']}, {len(fhir['entry'])} entries)")

    # 4) Grounding: machine findings vs. a radiologist-style report.
    grounding = client.post(
        "/api/v1/grounding/verify",
        json={"study_instance_uid": study_uid, "report_text": "No acute abnormality. The chest is unremarkable."},
    ).json()
    print(f"[4] Grounding            -> consistency score {grounding['consistency_score']:.2f}")
    for v in grounding["verdicts"][:3]:
        print(f"      {v['verdict']:>12}  {v['code_meaning']}")

    print("\nDone. Serve the API with `dicomweb-gateway` and open http://127.0.0.1:8000/docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
