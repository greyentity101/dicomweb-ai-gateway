"""Structured result serialization: DICOM SR (PS3.3) and HL7 FHIR (R4).

Given a :class:`StudyInferenceResult`, the gateway can emit two standards-compliant
representations so downstream systems can consume machine output:

* **DICOM Structured Report** — a real DICOM SR instance built with pydicom
  (SOP class ``1.2.840.10008.5.1.4.1.1.88.22``, Enhanced SR), one ``CONTAINER``
  with one ``CONTAINER`` per instance and coded findings as ``CODE``/``NUM``
  content items.  The serialized bytes can be STOW-RS'd back into a PACS.
* **FHIR R4 resources** — a ``DiagnosticReport`` plus one ``Observation`` per
  finding, wrapped in a ``Bundle`` (transaction).  This is the machine-consumable
  form for hospital integration layers.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydicom import Dataset
from pydicom.dataset import FileMetaDataset
from pydicom.uid import (  # 1.2.840.10008.5.1.4.1.1.88.22
    PYDICOM_IMPLEMENTATION_UID,
    EnhancedSRStorage,
    ExplicitVRLittleEndian,
)
from pydicom.valuerep import PersonName

from .schemas import Finding, StudyInferenceResult

#: FHIR release identifier used in resource meta.
FHIR_VERSION = "4.0.1"


# ---------------------------------------------------------------------------
# DICOM SR
# ---------------------------------------------------------------------------

def build_sr(dataset_sop_uid: str, result: StudyInferenceResult, institution: str = "") -> Dataset:
    """Build an Enhanced SR dataset summarizing a study's inference findings."""
    sr = Dataset()
    now = datetime.now(timezone.utc)
    sr.SOPClassUID = EnhancedSRStorage
    sr.SOPInstanceUID = dataset_sop_uid
    sr.StudyInstanceUID = result.study_instance_uid
    sr.SeriesInstanceUID = f"1.2.826.0.1.3680043.2.1125.{uuid.uuid4().int & (2**32 - 1)}"
    sr.Modality = "SR"
    sr.SeriesNumber = 1
    sr.InstanceNumber = 1
    sr.StudyDate = now.strftime("%Y%m%d")
    sr.StudyTime = now.strftime("%H%M%S.%f")[:13]
    sr.StudyDescription = "AI gateway structured report"
    if institution:
        sr.InstitutionName = institution
    sr.ReferringPhysicianName = PersonName("")
    sr.PatientName = PersonName("GATEWAY^AI")
    sr.PatientID = ""

    # Complete file meta so the SR serializes as a valid standalone DICOM file.
    sr.file_meta = FileMetaDataset()
    sr.file_meta.MediaStorageSOPClassUID = EnhancedSRStorage
    sr.file_meta.MediaStorageSOPInstanceUID = sr.SOPInstanceUID
    sr.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    sr.file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    # --- DICOM SR content tree (TID 1500 root container) ---
    _add_content_item(sr, "CONTAINER", code_value="18782-3", code_scheme="LN", code_meaning="Radiology Study observation")
    sr.ContentSequence[0].ContinuityOfContent = "SEPARATE"

    # One container per instance.
    instance_items = []
    for inst in result.instances:
        instance_items.append(_instance_container(inst.sop_instance_uid, inst.findings))
    sr.ContentSequence[0].ContentSequence = instance_items
    return sr


def _instance_container(sop_uid: str, findings: list[Finding]) -> Dataset:
    """Build a CONTAINER content item describing one instance's findings."""
    container = Dataset()
    _set_code_item(container, "CONTAINER", code_value="111028", code_scheme="DCM", code_meaning="Image Library")
    container.ContinuityOfContent = "SEPARATE"
    sub_items: list[Dataset] = []

    # Which instance this container refers to.
    ref = Dataset()
    ref.RelationshipType = "CONTAINS"
    ref.ValueType = "UIDREF"
    ref.ConceptNameCodeSequence = [_code("121191", "DCM", "Referenced SOP Instance UID")]
    ref.UID = sop_uid
    sub_items.append(ref)

    for f in findings:
        item = Dataset()
        item.RelationshipType = "CONTAINS"
        if f.unit:
            item.ValueType = "NUM"
            item.MeasuredValueSequence = [_measured_value(f)]
        else:
            item.ValueType = "CODE"
            item.ConceptCodeSequence = [_code(f.code_value, f.coding_scheme, f.code_meaning)]
        item.ConceptNameCodeSequence = [_code(f.code_value, f.coding_scheme, f.code_meaning)]
        if f.value and f.unit is None:
            # Text value carried as the code's meaning; keep the coded entry too.
            item.ConceptCodeSequence[0].CodeMeaning = f.value
        sub_items.append(item)

    container.ContentSequence = sub_items
    return container


def _measured_value(finding: Finding) -> Dataset:
    mv = Dataset()
    mv.NumericValue = float(finding.value or 0.0)
    mv.MeasurementUnitsCodeSequence = [_code("255604002", "SCT", "Percent") if not finding.unit else _code("", "UCUM", finding.unit)]
    if not finding.unit:
        mv.MeasurementUnitsCodeSequence[0].CodeValue = "255604002"
        mv.MeasurementUnitsCodeSequence[0].CodingSchemeDesignator = "SCT"
        mv.MeasurementUnitsCodeSequence[0].CodeMeaning = "Percent"
    return mv


def _add_content_item(sr: Dataset, value_type: str, *, code_value: str, code_scheme: str, code_meaning: str) -> Dataset:
    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = value_type
    item.ConceptNameCodeSequence = [_code(code_value, code_scheme, code_meaning)]
    sr.ContentSequence = [item]
    return item


def _set_code_item(container: Dataset, value_type: str, *, code_value: str, code_scheme: str, code_meaning: str) -> None:
    container.ValueType = value_type
    container.ConceptNameCodeSequence = [_code(code_value, code_scheme, code_meaning)]


def _code(value: str, scheme: str, meaning: str) -> Dataset:
    code = Dataset()
    code.CodeValue = value or "127001"
    code.CodingSchemeDesignator = scheme or "DCM"
    code.CodeMeaning = meaning or ""
    return code


def sr_bytes(dataset: Dataset) -> bytes:
    """Serialize a SR dataset to a DICOM byte stream."""
    from .store import dataset_bytes

    return dataset_bytes(dataset)


# ---------------------------------------------------------------------------
# FHIR R4
# ---------------------------------------------------------------------------

def build_fhir_bundle(result: StudyInferenceResult, base_url: str) -> dict:
    """Build a FHIR R4 transaction Bundle of DiagnosticReport + Observations."""
    report_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    entry = [
        _fhir_entry(
            "DiagnosticReport",
            report_id,
            {
                "status": "final",
                "code": {"coding": [{"system": "http://loinc.org", "code": "18782-3", "display": "Radiology Study observation"}]},
                "subject": {"reference": "Patient/__study__"},
                "effectiveDateTime": now,
                "performer": [{"display": "DICOMweb AI Gateway"}],
                "conclusion": f"{result.processed_count}/{result.total_count} instances processed by worker '{result.worker}'.",
                "result": [
                    {"reference": f"Observation/{uuid.uuid4().hex}"}
                    for _ in _all_findings(result)
                ],
            },
        )
    ]

    for finding in _all_findings(result):
        entry.append(
            _fhir_entry(
                "Observation",
                uuid.uuid4().hex,
                {
                    "status": "final",
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct" if finding.coding_scheme != "DCM" else "http://dicom.nema.org/resources/ontology/DCM",
                                "code": finding.code_value,
                                "display": finding.code_meaning,
                            }
                        ]
                    },
                    "subject": {"reference": "Patient/__study__"},
                    "valueString": finding.value,
                    "interpretation": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "N"}]}],
                },
            )
        )

    return {
        "resourceType": "Bundle",
        "id": uuid.uuid4().hex,
        "meta": {"lastUpdated": now, "profile": ["http://hl7.org/fhir/R4/Bundle"]},
        "type": "transaction",
        "entry": entry,
    }


def _all_findings(result: StudyInferenceResult) -> list[Finding]:
    return [f for inst in result.instances for f in inst.findings]


def _fhir_entry(resource_type: str, resource_id: str, resource: dict) -> dict:
    return {
        "fullUrl": f"urn:uuid:{resource_id}",
        "resource": {"resourceType": resource_type, "id": resource_id, **resource},
        "request": {"method": "POST", "url": f"{resource_type}"},
    }
