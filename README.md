# Protein–Ligand MVP

Minimal FastAPI + RDKit backend with a Streamlit UI for heuristic ligand scoring and simple improvements.

## Repository layout (2 lines per file)

- services/__init__.py
  - Marks `services` as a Python package. No runtime logic.

- services/api/__init__.py
  - Marks `services.api` as a package. No runtime logic.

- services/api/main.py
  - FastAPI app with endpoints: POST /predict and POST /improve. Wires to the scoring stubs and returns JSON.

- services/api/schemas.py
  - Pydantic v2 models for requests/responses. Includes minimal models for /predict and stub improver.

- services/api/utils.py
  - Helper to validate ligand inputs. Currently supports SMILES-only.

- services/ml/__init__.py
  - Marks `services.ml` as a package. No runtime logic.

- services/ml/scoring.py
  - RDKit-based deterministic heuristic scorer + SDF conformer generation for viewer. Also contains stub improver.

- services/ml/explain.py
  - Extra heuristics for polarity/flexibility warnings. Not used by minimal endpoints, kept for expansion.

- services/ml/improve.py
  - RDKit mutation-based improver prototype (not used by stub endpoint). Left for future extension.

- frontend/app.py
  - Streamlit UI: inputs for Protein ID + SMILES, calls /predict, shows score/explanations, and renders 3D via py3Dmol.

- requirements.txt
  - Core Python dependencies (FastAPI, Uvicorn, Pydantic). Install RDKit via Conda on Windows.

- data/, checkpoints/
  - Empty folders reserved for inputs/outputs or future artifacts.

## Run the backend

1) Create/activate an environment with RDKit (recommended on Windows):
   - Using conda/mamba (example):
     - `mamba create -n plmvp -y python=3.10 rdkit`
     - `mamba activate plmvp`
2) Install Python dependencies:
   - `pip install -r requirements.txt`
   - Also install UI deps if needed: `pip install streamlit py3Dmol requests`
3) Start FastAPI:
   - From project root: `uvicorn services.api.main:app --reload --port 8001`
4) Open API docs:
   - http://127.0.0.1:8001/docs

## Run the Streamlit UI

1) From project root: `streamlit run frontend/app.py`
2) In the sidebar, set `BASE_URL` to your backend (e.g., `http://127.0.0.1:8001`).
3) Enter Protein ID and SMILES, click Predict; score/explanations display and the 3D viewer renders the ligand.

## Notes

- /predict returns: `score`, `calibration_info`, `explanations`, and `viewer_payload` with an SDF block for 3D.
- /improve is a stub that returns mocked variants if the original score < target.
