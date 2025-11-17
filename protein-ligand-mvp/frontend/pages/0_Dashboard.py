import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from streamlit.components.v1 import html as st_html

try:
    import py3Dmol  # type: ignore
except Exception:  # pragma: no cover
    py3Dmol = None

st.title("Dashboard")

# ----------------------
# Session login (local only)
# ----------------------
if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None

with st.sidebar:
    st.header("Session")
    if st.session_state["auth_user"]:
        st.success(f"Logged in as {st.session_state['auth_user']}")
        if st.button("Logout"):
            st.session_state["auth_user"] = None
    else:
        user = st.text_input("Username", key="login_user")
        if st.button("Login"):
            if user.strip():
                st.session_state["auth_user"] = user.strip()
            else:
                st.warning("Enter a username to start a session")

# ----------------------
# Backend config + health check
# ----------------------

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
    if "base_url_dash" not in st.session_state:
        st.session_state["base_url_dash"] = BASE_URL
    st.text_input(
        "Backend BASE_URL",
        key="base_url_dash",
        value=st.session_state["base_url_dash"],
        help="FastAPI base URL (e.g., http://127.0.0.1:8001)",
    )
    BASE_URL = str(st.session_state.get("base_url_dash", BASE_URL)).strip()

health_placeholder = st.empty()

def _check_backend(url: str) -> Tuple[bool, str]:
    try:
        r = requests.get(url.rstrip("/") + "/docs", timeout=5)
        return (r.status_code == 200, f"HTTP {r.status_code}")
    except Exception as e:
        return (False, str(e))

ok, msg = _check_backend(BASE_URL)
if ok:
    health_placeholder.success(f"Backend reachable: {msg}")
else:
    health_placeholder.error(f"Backend not reachable: {msg}")

# ----------------------
# Helpers: history + viewers
# ----------------------
if "result_history" not in st.session_state:
    st.session_state["result_history"] = []  # list of dicts

@st.cache_data(ttl=300)
def cached_predict(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(base_url.rstrip("/") + "/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=300)
def cached_dock(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(base_url.rstrip("/") + "/dock_predict", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=300)
def cached_improve(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(base_url.rstrip("/") + "/improve", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

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


def render_mol(model_str: str, fmt: str = "sdf", style: str = "stick", bg: str = "0x00000000", height: int = 450):
    if py3Dmol is None:
        st.info("py3Dmol is not installed. Run: pip install py3Dmol")
        return
    view = py3Dmol.view(width=700, height=height)
    view.setBackgroundColor(bg)
    try:
        if fmt == "smi":
            model_str, fmt = BENZENE_SDF, "sdf"
        view.addModel(model_str or BENZENE_SDF, fmt)
    except Exception:
        view.addModel(BENZENE_SDF, "sdf")
    if style == "cartoon":
        view.setStyle({"cartoon": {"color": "spectrum"}})
    elif style == "surface":
        view.addSurface(py3Dmol.VDW, {"opacity": 0.85, "colorscheme": "default"})
    else:
        view.setStyle({"stick": {"radius": 0.2}})
    view.zoomTo()
    try:
        html_str = view.show()
    except Exception:
        html_str = view._make_html()
    st_html(html_str, height=height, scrolling=False)

# ----------------------
# Tabs
# ----------------------
scoring_tab, docking_tab, improvement_tab, proteins_tab = st.tabs(["Scoring", "Docking", "Improvement", "Protein Browser"])

with scoring_tab:
    st.subheader("Ligand Scoring")
    col1, col2 = st.columns(2)
    with col1:
        protein_id = st.text_input("Protein ID", value="1ABC", key="dash_score_pid")
        ligand = st.text_input("Ligand SMILES", value="CCO", key="dash_score_smi")
    with col2:
        mode = st.selectbox("Mode", ["heuristic", "ml"], index=0, key="dash_score_mode")
        style = st.selectbox("Viewer style", ["stick", "cartoon", "surface"], index=0)
    if st.button("Score", type="primary", key="dash_score_btn"):
        payload = {"protein_id": protein_id.strip(), "ligand": ligand.strip(), "mode": mode}
        try:
            with st.spinner("Scoring..."):
                data = cached_predict(BASE_URL, payload)
            st.metric("Score", f"{data.get('score')}")
            exps = data.get("explanations", [])
            if exps:
                st.markdown("**Explanations**")
                for e in exps:
                    st.markdown(f"- {e}")
            # Viewer from viewer_payload or fallback
            vp = data.get("viewer_payload") or {}
            sdf = vp.get("ligand_sdf") if isinstance(vp, dict) else None
            render_mol(sdf or BENZENE_SDF, fmt="sdf", style=style)
            st.session_state["result_history"].append({"type": "score", "payload": payload, "result": data})
        except Exception as e:
            st.error(f"Scoring failed: {e}")

with docking_tab:
    st.subheader("Docking")
    col1, col2 = st.columns(2)
    with col1:
        protein_id = st.text_input("Protein ID (uploaded UUID)", value="", key="dash_dock_pid")
        ligand = st.text_input("Ligand SMILES", value="CCO", key="dash_dock_smi")
    with col2:
        style = st.selectbox("Pose style", ["stick", "cartoon", "surface"], index=0, key="dash_dock_style")
        spin = st.checkbox("Spin animation", value=False, key="dash_dock_spin")
    # Optional protein PDB for visualization only (does not affect backend)
    prot_pdb_for_view = st.file_uploader("Optional: load protein PDB for visualization only", type=["pdb"], key="dash_dock_prot_view")
    if st.button("Dock", type="primary", key="dash_dock_btn"):
        payload = {"protein_id": protein_id.strip(), "ligand": ligand.strip()}
        try:
            with st.spinner("Docking with Vina..."):
                data = cached_dock(BASE_URL, payload)
            st.metric("Dock score", f"{data.get('score')}")
            aff = data.get("affinity")
            if aff is not None:
                st.caption(f"Affinity: {aff} kcal/mol (more negative is better)")
            pose = data.get("pose")
            if pose or prot_pdb_for_view is not None:
                # Render combined scene if possible: protein (cartoon) + ligand pose (sticks)
                if py3Dmol is None:
                    st.info("py3Dmol is not installed. Run: pip install py3Dmol")
                else:
                    view = py3Dmol.view(width=700, height=500)
                    view.setBackgroundColor('0x00000000')
                    # Protein first (if provided)
                    if prot_pdb_for_view is not None:
                        try:
                            pdb_text = prot_pdb_for_view.getvalue().decode('utf-8', errors='ignore')
                            view.addModel(pdb_text, 'pdb')
                            view.setStyle({"cartoon": {"color": "spectrum"}})
                        except Exception:
                            st.warning("Failed to parse uploaded protein PDB for visualization.")
                    # Ligand pose next (if available)
                    if pose:
                        try:
                            view.addModel(pose, 'sdf')
                            view.addStyle({}, {"stick": {"radius": 0.2}})
                        except Exception:
                            st.warning("Failed to parse ligand pose SDF for visualization.")
                    view.zoomTo()
                    if spin:
                        try:
                            view.spin(True)
                        except Exception:
                            pass
                    try:
                        html_str = view.show()
                    except Exception:
                        html_str = view._make_html()
                    st_html(html_str, height=500, scrolling=False)
            else:
                st.info("No pose SDF available (tooling may be missing or conversion failed).")
            st.session_state["result_history"].append({"type": "dock", "payload": payload, "result": data})
        except Exception as e:
            st.error(f"Docking failed: {e}")

with improvement_tab:
    st.subheader("Improvement")
    col1, col2 = st.columns(2)
    with col1:
        protein_id = st.text_input("Protein ID", value="1ABC", key="dash_imp_pid")
        ligand = st.text_input("Ligand SMILES", value="CCO", key="dash_imp_smi")
    with col2:
        target = st.number_input("Target score", value=80, min_value=1, max_value=100, step=1, key="dash_imp_target")
    if st.button("Run improvement", type="primary", key="dash_imp_btn"):
        payload = {"protein_id": protein_id.strip(), "ligand_smiles": ligand.strip(), "target_score": int(target)}
        try:
            with st.spinner("Optimizing..."):
                data = cached_improve(BASE_URL, payload)
            st.json({"base_score": data.get("base_score")})
            imps = data.get("improvements", [])
            if imps:
                st.markdown("**Top candidates**")
                st.dataframe(imps, use_container_width=True)
            trace = data.get("trace", [])
            with st.expander("Trace"):
                st.dataframe(trace, use_container_width=True)
            st.session_state["result_history"].append({"type": "improve", "payload": payload, "result": data})
        except Exception as e:
            st.error(f"Improvement failed: {e}")

with proteins_tab:
    st.subheader("Protein Browser")
    # Prefer project root data/proteins; fallback to frontend/data/proteins
    proj_root = Path(__file__).resolve().parents[2]
    dir_primary = proj_root / "data" / "proteins"
    dir_fallback = Path(__file__).resolve().parents[1] / "data" / "proteins"
    proteins_dir = dir_primary if dir_primary.exists() else dir_fallback
    if proteins_dir.exists():
        files = sorted([p.name for p in proteins_dir.glob("*.pdb")])
        if files:
            st.write(f"Found {len(files)} proteins")
            st.dataframe({"file": files, "protein_id": [f.split(".")[0] for f in files]}, use_container_width=True)
        else:
            st.info("No proteins uploaded yet. Use the UploadProtein page to add one.")
    else:
        st.info("Proteins directory not found. It will be created on first upload.")

st.divider()
st.subheader("History (cached)")
if st.session_state["result_history"]:
    st.dataframe(st.session_state["result_history"], use_container_width=True)
    dl = st.download_button(
        "Download history JSON",
        data=json.dumps(st.session_state["result_history"], indent=2).encode("utf-8"),
        file_name="history.json",
        mime="application/json",
    )
else:
    st.caption("No results yet.")
