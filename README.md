# DICOMweb AI Gateway

Self-hosted, standards-compliant AI orchestration for radiology.

```
DICOM instances ──▶ STOW-RS ──▶ ┌──────────────────────────────┐ ──▶ JSON result
  (from a PACS,               │    DICOMweb AI Gateway       │ ──▶ DICOM Structured Report
   modality, disk)            │                              │ ──▶ HL7 FHIR R4 Bundle
                              │  store → inference → results │
                              │  ───── + report grounding ─── │ ──▶ consistency verdicts
                              └──────────────────────────────┘
                                        ▲  ▲         ▲
                              reference │  │         │ local LLM (optional)
                              worker    │  └─────────┴─ your model (torch, onnx, …)
```

Clinical-grade AI models exist; *deployment* into radiology practice does not
scale. The bottleneck is integration — moving a study from a PACS to the right
model, and getting the result back in a standards-compliant, reviewable form.
This gateway addresses that bottleneck with an open, model-agnostic pipeline
built entirely on DICOMweb (DICOM PS3.18) and HL7 FHIR, designed to run on a
commodity machine with **no cloud dependency**.

> This repository is the reference *implementation* of the architecture and
> evaluation protocol described in the accompanying research paper
> **"A DICOMweb-Native AI Orchestration Gateway for Radiology"**.

## Features

| Layer | What it does | Standards |
|---|---|---|
| **Ingestion** | Accepts DICOM instances via `multipart/related` (STOW-RS) or plain upload; validates required tags; persists to a filesystem store. | DICOMweb PS3.18 §6.6 / §10.4 |
| **Inference** | Routes instances to a pluggable model worker and normalises output. Ships a zero-weight `reference-metadata` worker so the pipeline runs offline out of the box. | — |
| **Results** | Serializes findings to JSON, a real **DICOM SR** instance, and an **FHIR R4** transaction `Bundle`. | PS3.3 (TID 1500), FHIR R4 |
| **Grounding** | Reconciles machine findings against the radiologist's free-text report and flags `matched` / `unsupported` / `contradicted`. Default engine is offline; an optional local-LLM engine (Ollama/OpenAI-compatible) gives nuanced verdicts. | — |

## Quickstart

Requires Python ≥ 3.10.

```bash
git clone https://github.com/greyentity101/dicomweb-ai-gateway.git
cd dicomweb-ai-gateway
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[test]"

# Run the test suite (22 tests)
pytest

# Start the API on http://127.0.0.1:8000
dicomweb-gateway          # or: uvicorn dicomweb_ai_gateway.main:app
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

### End-to-end with curl

**1. Ingest a study** (STOW-RS) — send one or more `application/dicom` parts:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/dicomweb/studies \
     -H 'Content-Type: multipart/related; type="application/dicom"; boundary=abc' \
     --data-binary @- <<'EOF'
--abc
Content-Type: application/dicom

<your-instance.dcm bytes>
--abc--
EOF
```

**2. Run inference:**

```bash
curl -X POST http://127.0.0.1:8000/api/v1/inference/studies/<StudyInstanceUID>/run
```

**3. Pull the structured results:**

```bash
curl http://127.0.0.1:8000/api/v1/results/studies/<StudyInstanceUID>       # JSON
curl http://127.0.0.1:8000/api/v1/results/studies/<StudyInstanceUID>/sr    # DICOM SR (.dcm)
curl http://127.0.0.1:8000/api/v1/results/studies/<StudyInstanceUID>/fhir  # FHIR Bundle
```

**4. Verify the report** (grounding):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/grounding/verify \
     -H 'Content-Type: application/json' \
     -d '{"study_instance_uid":"<StudyInstanceUID>","report_text":"The chest is unremarkable."}'
```

See [`examples/run_demo.py`](examples/run_demo.py) for a self-contained script
that synthesizes a DICOM study and exercises the whole pipeline.

## API reference

All routes live under `/api/v1`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` · `/health` | Service info, health check |
| `POST` | `/dicomweb/studies` | **STOW-RS** — ingest `multipart/related` DICOM parts |
| `POST` | `/dicomweb/upload` | Convenience single-file `.dcm` upload |
| `GET` | `/dicomweb/studies` | **QIDO-RS** — list stored studies |
| `GET` | `/dicomweb/studies/{study}/metadata` | DICOMweb instance metadata |
| `GET` | `/dicomweb/studies/{study}/instances/{series}/{instance}` | **WADO-RS** — original instance bytes |
| `POST` | `/inference/studies/{study}/run` | Run the configured model worker |
| `GET` | `/results/studies/{study}` | JSON structured result |
| `GET` | `/results/studies/{study}/sr` | DICOM SR instance (downloadable `.dcm`) |
| `GET` | `/results/studies/{study}/fhir` | FHIR R4 transaction Bundle |
| `POST` | `/grounding/verify` | Reconcile findings vs. a free-text report |

## Bringing your own model

Implement the [`BaseModelWorker`](src/dicomweb_ai_gateway/inference.py) contract
and register it on the orchestrator:

```python
from dicomweb_ai_gateway.inference import BaseModelWorker, InferenceOrchestrator
from dicomweb_ai_gateway.schemas import Finding

class MyUNetWorker(BaseModelWorker):
    name = "my-unet"

    def predict(self, dataset):
        seg = my_network(dataset.pixel_array)          # your inference
        return [Finding(code_value="24642003",
                        code_meaning="Pulmonary mass",
                        coding_scheme="SCT",
                        value=f"{seg.area_px} px", confidence=seg.p, provenance=self.name)]

orch = InferenceOrchestrator(store)
orch.register(MyUNetWorker())
```

## Report grounding

`/grounding/verify` takes either explicit findings or a stored study, plus a
free-text radiology report, and returns one verdict per finding plus an overall
consistency score. Swap the offline engine for a local LLM (e.g. Ollama):

```python
from dicomweb_ai_gateway.grounding import LLMGroundingEngine
state.grounding = LLMGroundingEngine(model="llama3.2", base_url="http://localhost:11434/v1")
```

The LLM engine degrades gracefully to the keyword engine if the endpoint is
unreachable.

## Deployment

```bash
docker build -t dicomweb-ai-gateway .
docker run -p 8000:8000 -v $PWD/data:/app/data dicomweb-ai-gateway
```

A `docker-compose.yml` is included for a gateway + local Ollama stack.

## Project layout

```
src/dicomweb_ai_gateway/
  main.py         FastAPI app, routes, lifespan wiring
  store.py        DICOM store — STOW-RS/WADO-RS/QIDO-RS semantics
  inference.py    BaseModelWorker + reference/torch workers + orchestrator
  results.py      DICOM SR + FHIR R4 serializers
  grounding.py    report-grounding engines + consistency scoring
  schemas.py      pydantic request/response models
tests/            pytest suite (store, inference, results, grounding, e2e API)
examples/         run_demo.py — synthesize a study, run the full pipeline
```

## Roadmap

- [x] Reference architecture implementation (this repository)
- [x] STOW-RS / WADO-RS / QIDO-RS core
- [x] DICOM SR + FHIR output
- [x] Offline + local-LLM report grounding
- [ ] E1–E4 evaluation runs on a CPU-only stack (per the paper's protocol)
- [ ] DICOMweb QIDO-RS full search semantics + C-STORE import bridge

## License

[MIT](LICENSE) © 2026 Mohit Kumar. Built on `pydicom`, `FastAPI`, and friends —
medical imaging interoperability should be open.
