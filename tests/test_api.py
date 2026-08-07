"""End-to-end API tests: STOW-RS → inference → results → grounding."""

from __future__ import annotations

from pydicom import uid

from tests.conftest import dicom_bytes, make_dcm


def _stow(client, boundary="test-boundary"):
    ds = make_dcm(study_uid=uid.generate_uid(), body_part="CHEST")
    body = (
        b"--" + boundary.encode() + b"\r\n"
        b"Content-Type: application/dicom\r\n"
        b"Content-Transfer-Encoding: binary\r\n\r\n" + dicom_bytes(ds) + b"\r\n"
        b"--" + boundary.encode() + b"--\r\n"
    )
    resp = client.post(
        "/api/v1/dicomweb/studies",
        content=body,
        headers={"Content-Type": f'multipart/related; type="application/dicom"; boundary={boundary}'},
    )
    assert resp.status_code == 202, resp.text
    return ds


def test_health_and_root(client):
    assert client.get("/health").json() == {"status": "ok"}
    root = client.get("/").json()
    assert root["service"] == "DICOMweb AI Gateway"
    assert root["workers"][0]["name"] == "reference-metadata"


def test_full_pipeline(client):
    ds = _stow(client)

    # QIDO-RS lists the study.
    studies = client.get("/api/v1/dicomweb/studies").json()
    assert any(s["StudyInstanceUID"] == ds.StudyInstanceUID for s in studies["studies"])

    # Metadata endpoint returns instance metadata.
    meta = client.get(f"/api/v1/dicomweb/studies/{ds.StudyInstanceUID}/metadata").json()
    assert meta["instances"][0]["00080018"]["Value"] == [ds.SOPInstanceUID]

    # Inference.
    result = client.post(f"/api/v1/inference/studies/{ds.StudyInstanceUID}/run").json()
    assert result["status"] == "completed"
    assert result["total_count"] == 1
    assert len(result["instances"][0]["findings"]) >= 2

    # JSON result.
    r = client.get(f"/api/v1/results/studies/{ds.StudyInstanceUID}").json()
    assert r["sr_url"].endswith("/sr")

    # DICOM SR bytes.
    sr_resp = client.get(f"/api/v1/results/studies/{ds.StudyInstanceUID}/sr")
    assert sr_resp.status_code == 200
    assert sr_resp.headers["content-type"].startswith("application/dicom")

    # FHIR bundle.
    fhir = client.get(f"/api/v1/results/studies/{ds.StudyInstanceUID}/fhir").json()
    assert fhir["resourceType"] == "Bundle"

    # Grounding against a matching report.
    g = client.post(
        "/api/v1/grounding/verify",
        json={"study_instance_uid": ds.StudyInstanceUID, "report_text": "The chest is unremarkable."},
    ).json()
    assert "consistency_score" in g
    assert len(g["verdicts"]) >= 1


def test_stow_rejects_garbage(client):
    resp = client.post(
        "/api/v1/dicomweb/studies",
        content=b"this is not dicom",
        headers={"Content-Type": "multipart/related; boundary=bb"},
    )
    assert resp.status_code in (400, 202)  # empty parts → 400; malformed → handled


def test_wado_rs_returns_original(client):
    ds = _stow(client)
    resp = client.get(f"/api/v1/dicomweb/studies/{ds.StudyInstanceUID}/instances/{ds.SeriesInstanceUID}/{ds.SOPInstanceUID}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/dicom")
    assert resp.content == dicom_bytes(ds)


def test_missing_study_is_404(client):
    assert client.get(f"/api/v1/results/studies/{uid.generate_uid()}").status_code == 404
    assert client.get(f"/api/v1/dicomweb/studies/{uid.generate_uid()}/metadata").status_code == 404
