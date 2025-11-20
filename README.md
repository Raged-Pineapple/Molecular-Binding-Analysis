# Protein–Ligand MVP

This repository contains a minimal-but-complete protein–ligand scoring stack: a FastAPI backend powered by RDKit plus optional ML/docking backends, and a Streamlit UI (single-page and multipage) for chemists to score, dock, batch process, and iteratively improve ligands.  
This README expands on the earlier two-lines-per-file summary to explain every feature, how to run it, and how to interact with the system.

> All project code lives under `protein-ligand-mvp/` inside this repo.

---

## Repository Layout

```
protein-ligand-mvp/
├── services/                 # FastAPI + scoring/docking/improvement logic
│   ├── api/
│   │   ├── main.py           # Routes: /predict, /dock_predict, /batch_predict, /upload_protein, /improve
│   │   ├── schemas.py        # Pydantic v2 models shared by backend + UI
│   │   └── utils.py          # SMILES validation, PDB helpers, file IO
│   └── ml/
│       ├── scoring.py        # Deterministic heuristic scorer + 3D conformers
│       ├── explain.py        # Polarity/flexibility heuristics + recommendations
│       ├── model.py          # Optional torch MLP fallback for “ml” mode
│       ├── docking.py        # AutoDock Vina + Open Babel pipeline
│       ├── mutation.py       # Catalog of chemistry mutations
│       └── improve.py        # GA-style improvers (legacy + advanced)
├── frontend/
│   ├── app.py                # Single-page Streamlit app (predict + upload)
│   ├── Home.py               # Multipage entry point
│   └── pages/                # Dashboard, Predict, UploadProtein, BatchPredict
├── data/
│   └── proteins/             # Uploaded PDBs (UUID-named); sample files included
├── checkpoints/              # Reserved for future model weights
└── requirements.txt          # FastAPI + Pydantic core deps (install RDKit via Conda)
```

Auxiliary files in the repo root (`protein.pdbqt`, `test.sdf`, `test_log.txt`, `venv/`) are local artifacts and not required to run the MVP.

---

## Features at a Glance

- **Scoring modes**
  - `heuristic` (default): RDKit descriptors + deterministic explanations (`services/ml/scoring.py`).
  - `ml`: optional torch MLP fallback (`services/ml/model.py`) with feature dump.
  - `docking`: AutoDock Vina + Open Babel pipeline (`services/ml/docking.py`) with pose conversion to SDF.
- **Batch workflows**: `/batch_predict` scores multiple ligands in a single request and surfaces per-ligand explanations.
- **Protein ingestion**: `/upload_protein` accepts `.pdb`, persists in `data/proteins/<uuid>.pdb`, and returns atom/residue counts from `services/api/utils.parse_pdb_preview`.
- **Ligand improvement**:
  - Legacy GA (`improve.genetic_optimize`) for quick heuristic refinement.
  - Advanced GA (`improve.advanced_genetic_optimize`) with diversity control, optional docking/ML modes, debug trace, and metadata.
- **Frontend experiences**:
  - `frontend/app.py`: quick single-page input + 3D viewer + protein uploader.
  - `frontend/Home.py` multipage suite:
    - Dashboard tab for scoring, docking, improvement, and protein browsing.
    - Predict / UploadProtein single-purpose pages.
    - BatchPredict page with CSV export.
  - All pages allow configuring the backend URL via sidebar, env (`PLMVP_BASE_URL`), or Streamlit secrets.

---

## Requirements

| Component             | Notes                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------|
| Python 3.10           | Recommended via Conda/Mamba on Windows                                               |
| RDKit                 | Install through conda (`mamba install -c conda-forge rdkit`)                          |
| FastAPI stack         | From `requirements.txt` (`fastapi`, `uvicorn`, `pydantic`)                            |
| Streamlit UI extras   | `streamlit`, `py3Dmol`, `requests`, `pandas`, `graphviz`, (optional) `rdkit` in UI    |
| Docking dependencies  | AutoDock Vina (`vina` binary or set `VINA_BIN`) and Open Babel (`obabel` or `OBABEL_BIN`) |

Optional environment variables:

- `PLMVP_BASE_URL`: default backend URL for Streamlit.
- `VINA_BIN`: override AutoDock Vina path (Windows default hardcoded in `services/ml/docking.py`).
- `OBABEL_BIN`: override Open Babel binary path.
- `DOCK_KEEP_TMP=1`: persist docking intermediates under `data/dock_debug/` for debugging.

---

## Setup

```bash
# 1. Create a conda env with RDKit (recommended on Windows)
mamba create -n plmvp -y python=3.10 rdkit
conda activate plmvp

# 2. Install backend deps
cd protein-ligand-mvp
pip install -r requirements.txt

# 3. Install UI + optional tooling
pip install streamlit py3Dmol requests pandas graphviz
```

If you plan to use docking, ensure AutoDock Vina and Open Babel binaries are installed and discoverable (PATH or env vars above).

---

## Running the Backend

```bash
cd protein-ligand-mvp
uvicorn services.api.main:app --reload --port 8001
```

- Swagger UI: http://127.0.0.1:8001/docs  
- Uploaded proteins are stored under `data/proteins/` (UUID filenames).  
- Docking mode looks for `<protein_id>.pdb` in that folder—upload via API/UI first.  
- Logs print `VINA_BIN` and `which(vina)` on startup to help diagnose docking availability.

---

## Running the Streamlit UI

Single-page demo (predict + upload):

```bash
cd protein-ligand-mvp
streamlit run frontend/app.py
```

Full multipage experience:

```bash
cd protein-ligand-mvp
streamlit run frontend/Home.py
```

In the sidebar (or via `PLMVP_BASE_URL`/Streamlit secrets), point the UI to your FastAPI base URL. Each page remembers its own `BASE_URL` in `st.session_state`.

---

## API Reference & Usage

All endpoints live in `services/api/main.py`. Request/response models are under `services/api/schemas.py`.

### `POST /predict`
- Body: `{"protein_id": "1ABC", "ligand": "CCO", "mode": "heuristic|ml|docking"}`.
- Returns score (1–100 for heuristic/ML, scaled docking score), calibration tag, explanations, and optional `viewer_payload` containing an SDF for the ligand.
- Docking mode requires the protein PDB to exist under `data/proteins/<protein_id>.pdb`.

### `POST /dock_predict`
- Dedicated docking run using AutoDock Vina, returning `{score, affinity, pose}` where `pose` is an SDF string if conversion succeeds.
- Useful when you only care about docking and want the raw affinity (kcal/mol).

### `POST /batch_predict`
- Body: `{"protein_id": "...", "ligands": ["CCO", "CCCBr"]}`.
- Returns `results: [{ligand, score, explanation}]` with per-ligand failures captured as `score=0` plus the error message.

### `POST /upload_protein`
- `multipart/form-data` with a `.pdb` file.
- Persists the file, returns `{protein_id, message, preview_info: {num_atoms, num_residues}}`.
- The `protein_id` (UUID) is what docking mode expects later.

### `POST /improve`
- Body includes `protein_id`, `ligand_smiles`, `target_score`, plus optional GA controls (`mode`, `n_iters`, `pop_size`, `mutate_rate`, docking box `center/size`, `persist_debug`).
- Response: base score, list of improved molecules (with operations, parent relationships, docking metadata), and a trace array capturing every generation’s candidate.
- Modes:
  - No optional params → legacy heuristic GA.
  - Any tuning param or `mode` set → advanced GA path. When `mode="docking"`, scores are raw affinities (more negative is better); set `size` to shrink the docking box for “quick” mode.

Sample `curl`:

```bash
curl -X POST http://127.0.0.1:8001/predict \
     -H "Content-Type: application/json" \
     -d '{"protein_id": "1ABC", "ligand": "CCO", "mode": "heuristic"}'
```

---

## Streamlit Functionality

- **Dashboard (pages/0_Dashboard.py)**  
  Unified console for scoring, docking (with combined protein+pose rendering), running improvements, browsing uploaded proteins, and downloading cached history.

- **Predict page**  
  Mirrors `/predict` with the same viewer logic as the single-page app.

- **UploadProtein page**  
  Wraps `/upload_protein`, displays preview metrics, and renders the uploaded structure via py3Dmol.

- **BatchPredict page**  
  Multiligand input, uses `/batch_predict`, and provides a CSV download.

The UI caches recent API responses (`st.cache_data`) and, when RDKit/Graphviz are installed client-side, generates 2D SVG depictions, mutation trees, and highlight diffs for improved molecules.

---

## Data & Artifacts

- `data/proteins/`: PDB uploads saved as `<uuid>.pdb`. Several sample PDBs are already committed for convenience.
- `data/improve_debug/`: Created automatically when `/improve` is called with `persist_debug=true`.
- `data/dock_debug/`: Optional docking artifacts when `DOCK_KEEP_TMP=1`.
- `checkpoints/`: Reserved placeholder for future learned models.

`test.sdf`, `protein.pdbqt`, and `test_log.txt` in the repo root are example files for manual experimentation; they are not consumed automatically.

---

## Troubleshooting

- **RDKit import errors**: Ensure you installed RDKit via Conda; pip wheels are not provided for Windows.
- **Docking fails with “Vina not found”**: Install AutoDock Vina and set `VINA_BIN` to the executable path (Windows default is `C:/Program Files (x86)/The Scripps Research Institute/Vina/vina.exe`).
- **Pose conversion errors**: Install Open Babel (`obabel`) and ensure it is on PATH or specify `OBABEL_BIN`.
- **py3Dmol missing**: Install with `pip install py3Dmol`; without it, the UI will show instructional warnings.
- **Front-end cannot reach backend**: Check sidebar BASE_URL, confirm uvicorn port, and verify CORS (backend enables `allow_origins=["*"]` for MVP).

---

## Next Steps

- Swap the placeholder ML model with a trained checkpoint under `checkpoints/`.
- Extend `/improve` to persist best candidates or expose async jobs.
- Integrate real docking box selection (instead of heuristic center) via uploaded grids.


