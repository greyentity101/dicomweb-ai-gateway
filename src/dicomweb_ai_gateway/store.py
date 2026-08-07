"""Local DICOM store with DICOMweb (PS3.18) ingestion/retrieval semantics.

Implements the subset of DICOMweb needed for a self-hosted AI gateway:

* **STOW-RS** — store DICOM instances delivered as ``multipart/related`` with
  ``application/dicom`` parts (``POST /studies``).
* **WADO-RS** — retrieve instance bytes and instance metadata (``GET /studies/...``).
* **QIDO-RS** — lightweight study search over stored studies.

Storage is deliberately dumb and file-system based so the gateway runs on
commodity hardware with zero cloud dependency, mirroring the reference
architecture's deployment constraint.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pydicom import Dataset
from pydicom.errors import InvalidDicomError

#: Values required for a stored instance to be minimally useful downstream.
_REQUIRED_KEYWORDS = ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SOPClassUID")


def dataset_bytes(ds: Dataset) -> bytes:
    """Serialize a dataset to a complete DICOM file (preamble + file meta + data).

    ``dcmwrite(..., enforce_file_format=True)`` guarantees the ``DICM`` header
    is present so the bytes can be re-read by any DICOM tool, not just pydicom.
    """
    from io import BytesIO

    from pydicom import dcmwrite

    buf = BytesIO()
    dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()


class StoreError(RuntimeError):
    """Raised when a DICOM instance cannot be stored or retrieved."""


def _coerce_uid(uid: str) -> str:
    """Return the canonical UID string, rejecting obviously invalid input."""
    if not uid or not re.fullmatch(r"[0-9.]{1,64}", uid):
        raise StoreError(f"Invalid UID: {uid!r}")
    return uid


@dataclass
class StoredInstance:
    """A DICOM instance persisted by the store."""

    path: Path
    study_uid: str
    series_uid: str
    sop_uid: str
    sop_class_uid: str
    sha256: str
    bytes: int
    dataset: Dataset = field(repr=False, default_factory=Dataset)

    def metadata(self) -> dict:
        """DICOMweb-style instance metadata (PS3.18 Table 10.4-1 subset)."""
        return {
            "00080018": {"vr": "UI", "Value": [self.sop_uid]},  # SOPInstanceUID
            "00080016": {"vr": "UI", "Value": [self.sop_class_uid]},  # SOPClassUID
            "0020000D": {"vr": "UI", "Value": [self.study_uid]},  # StudyInstanceUID
            "0020000E": {"vr": "UI", "Value": [self.series_uid]},  # SeriesInstanceUID
            "00080060": _tag(self.dataset, "Modality", "OT"),  # Modality
            "00080008": {"vr": "CS", "Value": ["ORIGINAL", "PRIMARY"]},
        }


def _tag(dataset: Dataset, keyword: str, default: str | None = None) -> dict:
    value = getattr(dataset, keyword, None)
    if value is None or value == "":
        value = default
    if isinstance(value, (bytes, bytearray)):
        value = value.decode(errors="replace")
    if value is None:
        return {"vr": "UN"}
    if isinstance(value, (tuple, list)):
        value = list(value)
    else:
        value = [str(value)]
    return {"vr": "SH", "Value": value}


class DICOMStore:
    """Filesystem-backed DICOM store exposing DICOMweb-style operations."""

    def __init__(self, root: Path | str = "data/store") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # STOW-RS
    # ------------------------------------------------------------------
    def store_instances(self, datasets: Iterable[Dataset]) -> list[StoredInstance]:
        """Persist one or more parsed DICOM datasets; return the stored records."""
        stored: list[StoredInstance] = []
        for ds in datasets:
            missing = [k for k in _REQUIRED_KEYWORDS if not getattr(ds, k, None)]
            if missing:
                raise StoreError(f"Instance missing required tags: {', '.join(missing)}")
            study = _coerce_uid(ds.StudyInstanceUID)
            series = _coerce_uid(ds.SeriesInstanceUID)
            sop = _coerce_uid(ds.SOPInstanceUID)
            payload = dataset_bytes(ds)
            digest = hashlib.sha256(payload).hexdigest()

            with self._lock:
                dest = self.root / study / series
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / f"{sop}.dcm"
                path.write_bytes(payload)

            stored.append(
                StoredInstance(
                    path=path,
                    study_uid=study,
                    series_uid=series,
                    sop_uid=sop,
                    sop_class_uid=ds.SOPClassUID,
                    sha256=digest,
                    bytes=len(payload),
                    dataset=ds,
                )
            )
        return stored

    def parse_multipart(self, body: bytes, content_type: str) -> list[Dataset]:
        """Parse a DICOMweb ``multipart/related`` payload into pydicom datasets.

        Each part with ``Content-Type: application/dicom`` is decoded; parts with
        any other media type are skipped (per PS3.18 §6.6.1, mixed content is legal).
        """
        boundary = self._boundary(content_type)
        if not boundary:
            # Fallback: assume the whole body is a single DICOM file.
            return [self._decode(body)]
        datasets: list[Dataset] = []
        for part in self._split_multipart(body, boundary):
            payload = self._part_body(part)
            if not payload:
                continue
            headers = self._part_headers(part)
            media = headers.get("content-type", "application/dicom").split(";")[0].strip()
            if media != "application/dicom":
                continue
            datasets.append(self._decode(payload))
        return datasets

    # ------------------------------------------------------------------
    # WADO-RS / QIDO-RS
    # ------------------------------------------------------------------
    def get_instance_bytes(self, sop_uid: str) -> tuple[bytes, StoredInstance]:
        """Return (bytes, record) for a stored SOP instance UID."""
        record = self.find_by_sop(sop_uid)
        return record.path.read_bytes(), record

    def find_by_sop(self, sop_uid: str) -> StoredInstance:
        """Locate a stored instance by SOP instance UID."""
        needle = _coerce_uid(sop_uid)
        for path in self.root.rglob("*.dcm"):
            try:
                ds = pydicom_dcmread(path, stop_before_pixels=True)
            except InvalidDicomError:
                continue
            if ds.SOPInstanceUID == needle:
                return self._record(path, ds)
        raise StoreError(f"No stored instance with SOPInstanceUID {sop_uid}")

    def list_studies(self) -> list[dict]:
        """QIDO-RS-style study list with minimal demographic metadata."""
        studies: dict[str, dict] = {}
        for path in sorted(self.root.rglob("*.dcm")):
            try:
                ds = pydicom_dcmread(path, stop_before_pixels=True)
            except InvalidDicomError:
                continue
            uid = ds.StudyInstanceUID
            entry = studies.setdefault(
                uid,
                {
                    "StudyInstanceUID": uid,
                    "StudyDate": str(getattr(ds, "StudyDate", "")),
                    "StudyDescription": str(getattr(ds, "StudyDescription", "")),
                    "PatientName": str(getattr(ds, "PatientName", "")),
                    "PatientID": str(getattr(ds, "PatientID", "")),
                    "Modalities": set(),
                    "instances": 0,
                },
            )
            entry["Modalities"].add(str(getattr(ds, "Modality", "OT")))
            entry["instances"] += 1
        for entry in studies.values():
            entry["Modalities"] = sorted(entry["Modalities"])
        return list(studies.values())

    def instances_for_study(self, study_uid: str) -> list[StoredInstance]:
        """Return all stored instances belonging to a study."""
        study = _coerce_uid(study_uid)
        out: list[StoredInstance] = []
        for path in (self.root / study).rglob("*.dcm"):
            try:
                ds = pydicom_dcmread(path)
            except InvalidDicomError:
                continue
            out.append(self._record(path, ds))
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _boundary(content_type: str) -> str | None:
        match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", content_type, re.IGNORECASE)
        if not match:
            return None
        return (match.group(1) or match.group(2)).strip().strip('"')

    @staticmethod
    def _split_multipart(body: bytes, boundary: str) -> list[bytes]:
        delim = f"--{boundary}".encode()
        parts: list[bytes] = []
        for chunk in body.split(delim):
            chunk = chunk.strip(b"\r\n")
            if not chunk or chunk == b"--":
                continue
            parts.append(chunk)
        return parts

    @staticmethod
    def _part_headers(part: bytes) -> dict[str, str]:
        head, _, _ = part.partition(b"\r\n\r\n")
        headers: dict[str, str] = {}
        for line in head.split(b"\r\n"):
            if b":" in line:
                k, _, v = line.partition(b":")
                headers[k.strip().decode(errors="replace").lower()] = v.strip().decode(errors="replace")
        return headers

    @staticmethod
    def _part_body(part: bytes) -> bytes:
        _, _, body = part.partition(b"\r\n\r\n")
        # Tolerate the common single-\n variant.
        if not body:
            _, _, body = part.partition(b"\n\n")
        return body.rstrip(b"\r\n")

    @staticmethod
    def _decode(payload: bytes) -> Dataset:
        try:
            return pydicom_dcmread_from_bytes(payload)
        except (InvalidDicomError, EOFError, ValueError) as exc:
            raise StoreError(f"Failed to decode DICOM payload: {exc}") from exc

    def _record(self, path: Path, ds: Dataset) -> StoredInstance:
        payload = path.read_bytes()
        return StoredInstance(
            path=path,
            study_uid=ds.StudyInstanceUID,
            series_uid=ds.SeriesInstanceUID,
            sop_uid=ds.SOPInstanceUID,
            sop_class_uid=ds.SOPClassUID,
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
            dataset=ds,
        )


def pydicom_dcmread(path: Path, stop_before_pixels: bool = False) -> Dataset:
    """Thin wrapper so the store can be swapped for a custom reader in tests."""
    from pydicom import dcmread

    return dcmread(path, stop_before_pixels=stop_before_pixels)


def pydicom_dcmread_from_bytes(payload: bytes) -> Dataset:
    """Decode a DICOM dataset from a byte buffer."""
    from pydicom import dcmread
    from pydicom.filebase import DicomBytesIO

    return dcmread(DicomBytesIO(payload))
