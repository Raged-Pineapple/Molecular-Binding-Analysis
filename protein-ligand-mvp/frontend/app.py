import os
import json
from typing import Any, Dict

import requests
import streamlit as st
from streamlit.components.v1 import html as st_html

try:
    import py3Dmol  # type: ignore
except Exception:  # pragma: no cover
    py3Dmol = None

# Page config MUST be the first Streamlit command
st.set_page_config(page_title="Protein-Ligand MVP", page_icon="🧪", layout="centered")

# Base URL for the FastAPI backend. Configure via env var or Streamlit secrets.
def _resolve_base_url() -> str:
    # 1) Environment variable wins if present
    env_val = os.getenv("PLMVP_BASE_URL")
    if env_val:
        return env_val

    # 2) Only consult st.secrets if a secrets.toml file exists to avoid UI error banners
    try:
        from pathlib import Path

        user_secrets = Path(os.path.expanduser("~")) / ".streamlit" / "secrets.toml"
        proj_secrets = Path(".streamlit") / "secrets.toml"
        if user_secrets.exists() or proj_secrets.exists():
            # Access st.secrets only when file exists
            if "BASE_URL" in st.secrets:
                return str(st.secrets["BASE_URL"]).strip()
    except Exception:
        pass

    # 3) Fallback default
    return "http://127.0.0.1:8001"

BASE_URL: str = _resolve_base_url()
st.title("Protein–Ligand Scoring (MVP)")

# Hardcoded benzene SDF (3D-ish placeholder) to guarantee visible rendering
BENZENE_SDF = """
  Mrv2014 07272015012D          

 12 12  0  0  0  0            999 V2000
    1.3960    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6980    1.2091    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6980    1.2091    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3960    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6980   -1.2091    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6980   -1.2091    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.4785    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2393    2.1467    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2393    2.1467    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4785    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2393   -2.1467    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2393   -2.1467    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  4  5  1  0  0  0  0
  5  6  2  0  0  0  0
  6  1  1  0  0  0  0
  1  7  1  0  0  0  0
  2  8  1  0  0  0  0
  3  9  1  0  0  0  0
  4 10  1  0  0  0  0
  5 11  1  0  0  0  0
  6 12  1  0  0  0  0
M  END
"""

with st.sidebar:
    st.header("Backend Settings")
    # Initialize session state for the input to avoid KeyError on reruns
    if "base_url" not in st.session_state:
        st.session_state["base_url"] = BASE_URL
    st.text_input(
        "Backend BASE_URL",
        key="base_url",
        value=st.session_state["base_url"],
        help="FastAPI base URL (e.g., http://127.0.0.1:8001)",
    )
    BASE_URL = str(st.session_state.get("base_url", BASE_URL)).strip()

st.subheader("Input")
col1, col2 = st.columns(2)
with col1:
    protein_id = st.text_input("Protein ID (PDB)", value="1ABC", placeholder="e.g., 1ABC")
with col2:
    ligand_smiles = st.text_input("Ligand SMILES", value="CCO", placeholder="e.g., CCO")

submit = st.button("Predict", type="primary")

if submit:
    if not protein_id.strip() or not ligand_smiles.strip():
        st.error("Please provide both Protein ID and Ligand SMILES.")
    else:
        payload: Dict[str, Any] = {
            "protein_id": protein_id.strip(),
            "ligand": ligand_smiles.strip(),
        }
        url = f"{BASE_URL.rstrip('/')}/predict"
        with st.spinner("Scoring ligand..."):
            try:
                resp = requests.post(url, json=payload, timeout=30)
            except Exception as e:
                st.error(f"Failed to reach backend at {url}: {e}")
            else:
                if resp.status_code != 200:
                    detail = resp.text
                    try:
                        data = resp.json()
                        detail = data.get("detail", detail)
                    except Exception:
                        pass
                    st.error(f"Backend error ({resp.status_code}): {detail}")
                else:
                    try:
                        data = resp.json()
                    except Exception as e:
                        st.error(f"Invalid JSON response: {e}\nRaw: {resp.text[:500]}")
                    else:
                        st.subheader("Result")
                        score = data.get("score")
                        cal = data.get("calibration_info")
                        expl = data.get("explanations", []) or []

                        st.metric(label="Score", value=f"{score}")
                        if cal:
                            st.caption(f"Calibration: {cal}")

                        if expl:
                            st.markdown("**Explanations**")
                            for item in expl:
                                st.markdown(f"- {item}")

                        # 3D Viewer section
                        st.markdown("**3D Viewer**")

                        def _extract_structure(payload: Dict[str, Any]) -> tuple[str, str]:
                            # Priority: viewer_payload.ligand_sdf -> ligand_sdf -> pdb -> sdf -> placeholder benzene SMILES
                            vp = payload.get("viewer_payload") or {}
                            if isinstance(vp, dict) and vp.get("ligand_sdf"):
                                return str(vp["ligand_sdf"]), "sdf"
                            if payload.get("ligand_sdf"):
                                return str(payload["ligand_sdf"]), "sdf"
                            if payload.get("pdb"):
                                return str(payload["pdb"]), "pdb"
                            if payload.get("sdf"):
                                return str(payload["sdf"]), "sdf"
                            # Fallback: benzene SDF with coordinates
                            return BENZENE_SDF, "sdf"

                        def _render_3d(model_str: str, fmt: str) -> None:
                            if py3Dmol is None:
                                st.info("py3Dmol is not installed. Run: pip install py3Dmol")
                                return
                            w, h = 700, 450
                            view = py3Dmol.view(width=w, height=h)
                            view.setBackgroundColor('0x00000000')  # transparent to match theme
                            try:
                                # Convert smiles fallback into SDF to guarantee visibility
                                if fmt == "smi":
                                    model_str = BENZENE_SDF
                                    fmt = "sdf"
                                view.addModel(model_str, fmt)
                            except Exception:
                                # final fallback to benzene SDF
                                view.addModel(BENZENE_SDF, "sdf")
                            if fmt == "pdb":
                                view.setStyle({"cartoon": {"color": "spectrum"}})
                                view.addStyle({"hetflag": True}, {"stick": {"colorscheme": "cyanCarbon"}})
                            else:
                                view.setStyle({"stick": {"radius": 0.2}})
                            view.zoomTo()
                            try:
                                html_str = view.show()  # more reliable than _make_html in some envs
                            except Exception:
                                html_str = view._make_html()
                            st_html(html_str, height=h, scrolling=False)

                        try:
                            mstr, mfmt = _extract_structure(data)
                        except Exception:
                            mstr, mfmt = ("c1ccccc1", "smi")
                        _render_3d(mstr, mfmt)

                        with st.expander("Raw response"):
                            st.code(json.dumps(data, indent=2), language="json")

# ----------------------
# Upload protein (PDB)
# ----------------------
st.divider()
st.subheader("Upload protein (PDB)")
uploaded_pdb = st.file_uploader("Select a PDB file", type=["pdb"], key="pdb_uploader")
upload_clicked = st.button("Upload Protein")

def _render_protein_cartoon(pdb_text: str) -> None:
    if py3Dmol is None:
        st.info("py3Dmol is not installed. Run: pip install py3Dmol")
        return
    w, h = 700, 450
    view = py3Dmol.view(width=w, height=h)
    view.setBackgroundColor('0x00000000')
    try:
        view.addModel(pdb_text, 'pdb')
    except Exception:
        st.warning("Failed to parse PDB for 3D rendering.")
        return
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    try:
        html_str = view.show()
    except Exception:
        html_str = view._make_html()
    st_html(html_str, height=h, scrolling=False)

if upload_clicked:
    if not uploaded_pdb:
        st.error("Please select a .pdb file before uploading.")
    else:
        url = f"{BASE_URL.rstrip('/')}/upload_protein"
        try:
            files = {
                'file': (uploaded_pdb.name, uploaded_pdb.getvalue(), 'chemical/x-pdb')
            }
            with st.spinner("Uploading protein..."):
                resp = requests.post(url, files=files, timeout=60)
        except Exception as e:
            st.error(f"Failed to upload to {url}: {e}")
        else:
            if resp.status_code != 200:
                detail = resp.text
                try:
                    data = resp.json()
                    detail = data.get("detail", detail)
                except Exception:
                    pass
                st.error(f"Backend error ({resp.status_code}): {detail}")
            else:
                try:
                    res = resp.json()
                except Exception as e:
                    st.error(f"Invalid JSON response: {e}\nRaw: {resp.text[:500]}")
                else:
                    st.success(f"Protein uploaded: ID {res.get('protein_id')}")
                    preview = res.get('preview_info', {}) or {}
                    st.write({
                        'num_atoms': preview.get('num_atoms'),
                        'num_residues': preview.get('num_residues'),
                    })
                    # Render uploaded protein locally from the file content
                    try:
                        pdb_text = uploaded_pdb.getvalue().decode('utf-8', errors='ignore')
                    except Exception:
                        pdb_text = ''
                    if pdb_text:
                        _render_protein_cartoon(pdb_text)
                    with st.expander("Raw upload response"):
                        st.code(json.dumps(res, indent=2), language="json")

st.caption("Configure BASE_URL in the sidebar or via Streamlit secrets (BASE_URL) or env var PLMVP_BASE_URL.")
