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

# Do NOT call st.set_page_config here to avoid conflicts; it's in Home.py
st.title("Predict")

# Reuse BASE_URL resolution

def _resolve_base_url() -> str:
    env_val = os.getenv("PLMVP_BASE_URL")
    if env_val:
        return env_val
    try:
        from pathlib import Path
        user_secrets = Path(os.path.expanduser("~")) / ".streamlit" / "secrets.toml"
        proj_secrets = Path(".streamlit") / "secrets.toml"
        if user_secrets.exists() or proj_secrets.exists():
            if "BASE_URL" in st.secrets:
                return str(st.secrets["BASE_URL"]).strip()
    except Exception:
        pass
    return "http://127.0.0.1:8001"

BASE_URL: str = _resolve_base_url()

with st.sidebar:
    st.header("Backend Settings")
    if "base_url_predict" not in st.session_state:
        st.session_state["base_url_predict"] = BASE_URL
    st.text_input(
        "Backend BASE_URL",
        key="base_url_predict",
        value=st.session_state["base_url_predict"],
        help="FastAPI base URL (e.g., http://127.0.0.1:8001)",
    )
    BASE_URL = str(st.session_state.get("base_url_predict", BASE_URL)).strip()

# Inputs
st.subheader("Input")
col1, col2 = st.columns(2)
with col1:
    protein_id = st.text_input("Protein ID (PDB)", value="1ABC", placeholder="e.g., 1ABC")
with col2:
    ligand_smiles = st.text_input("Ligand SMILES", value="CCO", placeholder="e.g., CCO")

submit = st.button("Predict", type="primary")

# 3D helpers copied from single-page app
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


def _extract_structure(payload: Dict[str, Any]) -> tuple[str, str]:
    vp = payload.get("viewer_payload") or {}
    if isinstance(vp, dict) and vp.get("ligand_sdf"):
        return str(vp["ligand_sdf"]), "sdf"
    if payload.get("ligand_sdf"):
        return str(payload["ligand_sdf"]), "sdf"
    if payload.get("pdb"):
        return str(payload["pdb"]), "pdb"
    if payload.get("sdf"):
        return str(payload["sdf"]), "sdf"
    return BENZENE_SDF, "sdf"


def _render_3d(model_str: str, fmt: str) -> None:
    if py3Dmol is None:
        st.info("py3Dmol is not installed. Run: pip install py3Dmol")
        return
    w, h = 700, 450
    view = py3Dmol.view(width=w, height=h)
    view.setBackgroundColor('0x00000000')
    try:
        if fmt == "smi":
            model_str = BENZENE_SDF
            fmt = "sdf"
        view.addModel(model_str, fmt)
    except Exception:
        view.addModel(BENZENE_SDF, "sdf")
    if fmt == "pdb":
        view.setStyle({"cartoon": {"color": "spectrum"}})
        view.addStyle({"hetflag": True}, {"stick": {"colorscheme": "cyanCarbon"}})
    else:
        view.setStyle({"stick": {"radius": 0.2}})
    view.zoomTo()
    try:
        html_str = view.show()
    except Exception:
        html_str = view._make_html()
    st_html(html_str, height=h, scrolling=False)

# Predict action
if submit:
    if not protein_id.strip() or not ligand_smiles.strip():
        st.error("Please provide both Protein ID and Ligand SMILES.")
    else:
        payload: Dict[str, Any] = {"protein_id": protein_id.strip(), "ligand": ligand_smiles.strip()}
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
                        st.markdown("**3D Viewer**")
                        mstr, mfmt = _extract_structure(data)
                        _render_3d(mstr, mfmt)
                        with st.expander("Raw response"):
                            st.code(json.dumps(data, indent=2), language="json")

st.caption("Set BASE_URL via sidebar, env PLMVP_BASE_URL, or Streamlit secrets.")
