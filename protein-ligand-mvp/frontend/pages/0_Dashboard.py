import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import time
import streamlit as st
from streamlit.components.v1 import html as st_html
import pandas as pd

try:
    import py3Dmol  # type: ignore
except Exception:  # pragma: no cover
    py3Dmol = None

# Optional RDKit for 2D SVGs
try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Draw  # type: ignore
    from rdkit.Chem import rdFMCS  # type: ignore
    RD_OK = True
except Exception:
    RD_OK = False

# Optional Graphviz for tree rendering
try:
    import graphviz  # type: ignore
    GV_OK = True
except Exception:
    GV_OK = False

st.title("Dashboard")

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

def _post_json_with_retry(url: str, payload: Dict[str, Any], timeout: int = 60, retries: int = 3, backoff: float = 0.8) -> Dict[str, Any]:
    last_err = None
    for i in range(max(1, retries)):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if i < retries - 1:
                time.sleep(backoff * (2 ** i))
            else:
                raise

@st.cache_data(ttl=300)
def cached_predict(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return _post_json_with_retry(base_url.rstrip("/") + "/predict", payload, timeout=30)

@st.cache_data(ttl=300)
def cached_dock(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return _post_json_with_retry(base_url.rstrip("/") + "/dock_predict", payload, timeout=180)

@st.cache_data(ttl=300)
def cached_improve(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return _post_json_with_retry(base_url.rstrip("/") + "/improve", payload, timeout=240)

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
        mode = st.selectbox("Mode", ["heuristic"], index=0, key="dash_score_mode")
        style = st.selectbox("Viewer style", ["stick", "cartoon", "surface"], index=0)
    if st.button("Score", type="primary", key="dash_score_btn"):
        payload = {"protein_id": protein_id.strip(), "ligand": ligand.strip(), "mode": mode}
        try:
            with st.spinner("Analyzing Compatibility..."):
                data = cached_predict(BASE_URL, payload)
            
            sc1, sc2 = st.columns([1, 2])
            with sc1:
                st.metric("Binding Compatibility", f"{data.get('binding_score')}%")
            
            with sc2:
                # Property table
                props = data.get("ligand_properties") or {}
                if props:
                    st.markdown("**Ligand Properties**")
                    st.dataframe(pd.DataFrame([props]), hide_index=True)

            col_ex, col_rec = st.columns(2)
            with col_ex:
                exps = data.get("explanations", [])
                if exps:
                    st.markdown("**Explanations**")
                    for e in exps:
                        st.markdown(f"- {e}")
            
            with col_rec:
                recs = data.get("recommendations", [])
                if recs:
                    st.markdown("**Recommendations**")
                    for r in recs:
                        st.markdown(f"- {r}")
            
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
    if st.button("Dock", type="primary", key="dash_dock_btn"):
        payload = {"protein_id": protein_id.strip(), "ligand": ligand.strip()}
        try:
            with st.spinner("Docking with Vina..."):
                data = cached_dock(BASE_URL, payload)
            st.metric("Dock score", f"{data.get('score')}")
            aff = data.get("affinity")
            if aff is not None:
                st.subheader("Best docking result")
                st.markdown(f"**Affinity:** `{aff} kcal/mol`  (more negative is better)")
            pose = data.get("pose")
            if pose:
                # Render ligand pose (sticks)
                if py3Dmol is None:
                    st.info("py3Dmol is not installed. Run: pip install py3Dmol")
                else:
                    view = py3Dmol.view(width=700, height=500)
                    view.setBackgroundColor('0x00000000') # Transparent background
                    
                    # 1. Add Protein (Background - Muted)
                    clean_pid = protein_id.strip()
                    proj_root = Path(__file__).resolve().parents[2]
                    prot_path = proj_root / "data" / "proteins" / f"{clean_pid}.pdb"
                    if not prot_path.exists():
                         prot_path = Path("data") / "proteins" / f"{clean_pid}.pdb"
                    
                    if prot_path.exists():
                        try:
                            view.addModel(prot_path.read_text(), "pdb")
                            # Style: Muted Slate/Teal Cartoon, transparent
                            view.setStyle({"model": -1}, {"cartoon": {"color": "#607d8b", "opacity": 0.6}})
                        except Exception as e:
                            st.warning(f"Could not load protein structure: {e}")
                    else:
                        st.warning(f"Protein file not found at {prot_path}")

                    # 2. Add Ligand (Hero - High Contrast)
                    try:
                        view.addModel(pose, 'sdf')
                        # Style: Thick sticks, CPK colors (Jmol), Carbon=Grayish
                        view.setStyle({"model": -1}, {"stick": {"colorscheme": "Jmol", "radius": 0.3}})
                    except Exception:
                        st.warning("Failed to parse ligand pose SDF for visualization.")

                    # 3. Highlight Binding Pocket (Midground - Context)
                    # Select residues within 5 Angstroms of the ligand (model -1)
                    ligand_sel = {"model": -1}
                    pocket_sel = {"within": {"distance": 5, "sel": ligand_sel}}
                    
                    # Style 3.1: Semi-transparent surface for the pocket
                    view.addSurface(py3Dmol.MS, {"opacity": 0.4, "color": "#81d4fa"}, pocket_sel)
                    
                    # Style 3.2: Detailed sticks for pocket residues (thinner than ligand)
                    view.addStyle(pocket_sel, {"stick": {"radius": 0.1, "colorscheme": "chain"}})

                    # 4. Interactions (Hydrogen Bonds)
                    # Auto-calculate and draw H-bonds between ligand and everything else
                    # view.addHydrogenBonds might be tricky in pure python-wrapper without JS callback, 
                    # but simple invocation often works for standard residues.
                    # We will rely on the pocket visual context as primary interaction cue.

                    # 5. Camera & Animation
                    view.zoomTo({"model": -1}) # Focus on Ligand/Pocket
                    
                    if spin:
                         view.spin(True)

                    try:
                        html_str = view.show()
                    except Exception:
                        html_str = view._make_html()
                    st_html(html_str, height=500, scrolling=False)
                st.download_button(
                    "Download pose SDF",
                    data=pose.encode("utf-8"),
                    file_name="pose.sdf",
                    mime="chemical/x-mdl-sdfile",
                )
            else:
                st.info("No pose SDF available (tooling may be missing or conversion failed).")
        except Exception as e:
            st.error(f"Docking failed: {e}")

with improvement_tab:
    st.subheader("Improvement")
    col1, col2 = st.columns(2)
    with col1:
        protein_id = st.text_input("Protein ID", value="1ABC", key="dash_imp_pid")
        ligand = st.text_input("Ligand SMILES", value="CCO", key="dash_imp_smi")
    with col2:
        mode = st.selectbox("Mode", ["heuristic", "docking"], index=1, key="dash_imp_mode", help="Heuristic uses fast descriptors; Docking uses AutoDock Vina for precise affinity.")
    with col2:
        if mode == "docking":
            target = st.number_input("Target affinity (kcal/mol)", value=-8.0, max_value=0.0, step=0.1, key="dash_imp_target", help="Lower is better. Optimization stops if reached.")
        else:
            target = st.number_input("Target score (0-100)", value=80, min_value=1, max_value=100, step=1, key="dash_imp_target", help="Higher is better.")
    if st.button("Run improvement", type="primary", key="dash_imp_btn"):
        payload = {"protein_id": protein_id.strip(), "ligand_smiles": ligand.strip(), "target_score": int(target)}
        # Use advanced optimizer when user selects a mode or quick
        if mode:
            payload.update({"mode": mode})
        try:
            with st.spinner("Optimizing..."):
                data = cached_improve(BASE_URL, payload)
            base_score = data.get("base_score")
            unit = " kcal/mol" if mode == "docking" else ""
            st.markdown(f"**Base score:** `{base_score}`{unit}")

            imps = data.get("improvements", [])
            trace = data.get("trace", [])

            # Helper: SMILES -> score map from trace and final
            score_by_smiles = {}
            for t in trace or []:
                s = t.get("smiles"); sc = t.get("score")
                if s is not None and sc is not None:
                    score_by_smiles[s] = float(sc)
            for it in imps:
                s = it.get("smiles"); sc = it.get("score")
                if s is not None and sc is not None:
                    score_by_smiles[s] = float(sc)

            def mol_svg(smiles: str, size=(180, 140), highlight=None) -> Optional[str]:
                if not RD_OK or not smiles:
                    return None
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        return None
                    if highlight:
                        drawer = Draw.MolDraw2DSVG(size[0], size[1])
                        Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=list(highlight))
                        drawer.FinishDrawing()
                        return drawer.GetDrawingText()
                    return Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=size, useSVG=True)
                except Exception:
                    return None

            def diff_svg(parent: Optional[str], child: str) -> Optional[str]:
                if not RD_OK or not child:
                    return None
                try:
                    cm = Chem.MolFromSmiles(child)
                    if cm is None:
                        return None
                    if not parent:
                        return mol_svg(child)
                    pm = Chem.MolFromSmiles(parent)
                    if pm is None:
                        return mol_svg(child)
                    mcs = rdFMCS.FindMCS([pm, cm])
                    patt = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
                    highlight = []
                    if patt is not None:
                        match = cm.GetSubstructMatch(patt)
                        in_mcs = set(match)
                        highlight = [i for i in range(cm.GetNumAtoms()) if i not in in_mcs]
                    drawer = Draw.MolDraw2DSVG(220, 160)
                    if highlight:
                        Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, cm, highlightAtoms=highlight)
                    else:
                        Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, cm)
                    drawer.FinishDrawing()
                    return drawer.GetDrawingText()
                except Exception:
                    return None

            # Table with previews and colored delta
            st.markdown("**Top candidates**")
            if "debug_info" in data and data["debug_info"]:
                with st.expander("Debug Info (Developer)"):
                    st.json(data["debug_info"])
                    st.write("Raw Base Score:", data.get("base_score"))
            if not imps:
                st.info("No improvements found.")
            else:
                for it in imps:
                    s = it.get("smiles"); sc = float(it.get("score", 0.0))
                    # Priority: mutation_description -> op_name -> "-"
                    op = it.get("mutation_description") or it.get("op_name") or "-"
                    parent = it.get("parent_smiles")
                    psc = score_by_smiles.get(parent, base_score)
                    
                    # Determine improvement direction by mode
                    run_mode = (data.get("run_metadata", {}) or {}).get("mode", mode)
                    better = (sc < psc) if run_mode == "docking" else (sc > psc)
                    
                    # Priority: delta_affinity from backend -> calculated delta
                    delta = it.get("delta_affinity")
                    if delta is None and psc is not None:
                        delta = sc - psc

                    c1, c2, c3, c4 = st.columns([2, 2, 2, 1.2])
                    with c1:
                        svg = diff_svg(parent, s) or mol_svg(s)
                        if svg:
                            st.write(svg, unsafe_allow_html=True)
                        else:
                            st.code(s)
                    with c2:
                        st.markdown(f"**Operation**: {op}")
                        if parent:
                            psvg = mol_svg(parent)
                            if psvg:
                                st.write(psvg, unsafe_allow_html=True)
                            else:
                                st.code(parent)
                        else:
                            st.caption("Seed")
                    with c3:
                        st.markdown(f"**Parent**: `{parent or 'seed'}`")
                        unit = " kcal/mol" if run_mode == "docking" else ""
                        st.markdown(f"**Score**: `{sc}`{unit}")
                    with c4:
                        if delta is not None:
                            color = "#2e7d32" if better else "#c62828"
                            # For docking, negative delta is improvement.
                            # For heuristic, positive delta is improvement.
                            # We show '+' if it's an 'increase' in heuristic or 'worse' in docking?
                            # Actually, just show the delta sign from backend.
                            sign = "+" if (float(delta) > 0) else ""
                            st.markdown(f"<div style='font-weight:700;color:{color}'>Δ {sign}{delta:.2f}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("Δ n/a")
                    st.divider()

            # Trace table (compact)
            with st.expander("Trace (table)"):
                # Annotate operation by mapping from improvements (if available)
                op_map = { (it.get("smiles") or ""): (it.get("op_name") or "") for it in imps if it.get("smiles") }
                rows = []
                for t in trace or []:
                    rows.append({
                        "step": int(t.get("step", 0)),
                        "smiles": t.get("smiles"),
                        "score": float(t.get("score", 0.0)) if t.get("score") is not None else None,
                        "operation": op_map.get(t.get("smiles") or "", "-"),
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), width='stretch')
                else:
                    st.caption("No trace returned by backend.")

            # Mutation tree using Graphviz
            with st.expander("Mutation tree"):
                if not GV_OK:
                    st.info("graphviz not installed. Run: pip install graphviz")
                else:
                    dot = graphviz.Digraph()
                    # Build node ids by index
                    def node_label(sm: str) -> str:
                        sc = score_by_smiles.get(sm, None)
                        if sc is None:
                            return sm[:12] + ("…" if len(sm) > 12 else "")
                        return f"{sm[:12]}…\n{sc:.3f}"
                    # Seed node
                    seed = ligand.strip()
                    dot.node("seed", node_label(seed), shape="box")
                    added = {seed}
                    for it in imps:
                        s = it.get("smiles"); p = it.get("parent_smiles"); op = it.get("op_name") or ""
                        if not s:
                            continue
                        sid = f"n{abs(hash(s))%10**8}"
                        if s not in added:
                            dot.node(sid, node_label(s))
                            added.add(s)
                        if p:
                            pid = "seed" if p == seed else f"n{abs(hash(p))%10**8}"
                            if p not in added:
                                dot.node(pid, node_label(p))
                                added.add(p)
                            dot.edge(pid, sid, label=op)
                        else:
                            dot.edge("seed", sid, label=op)
                    st.graphviz_chart(dot)
                    st.download_button(
                        "Download Mutation Tree (DOT)",
                        data=dot.source,
                        file_name="mutation_tree.dot",
                        mime="text/vnd.graphviz",
                        key="dash_imp_dot_dl"
                    )
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
            st.dataframe({"file": files, "protein_id": [f.split(".")[0] for f in files]}, width='stretch')
        else:
            st.info("No proteins uploaded yet. Use the UploadProtein page to add one.")
    else:
        st.info("Proteins directory not found. It will be created on first upload.")

st.divider()
st.subheader("History (cached)")
if st.session_state["result_history"]:
    st.dataframe(st.session_state["result_history"], width='stretch')
    dl = st.download_button(
        "Download history JSON",
        data=json.dumps(st.session_state["result_history"], indent=2).encode("utf-8"),
        file_name="history.json",
        mime="application/json",
    )
else:
    st.caption("No results yet.")
