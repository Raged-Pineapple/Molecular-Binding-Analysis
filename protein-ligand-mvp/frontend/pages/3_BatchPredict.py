import os
import csv
import io
from typing import List, Dict

import requests
import streamlit as st

st.title("Batch Predict")


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
    if "base_url_batch" not in st.session_state:
        st.session_state["base_url_batch"] = BASE_URL
    st.text_input(
        "Backend BASE_URL",
        key="base_url_batch",
        value=st.session_state["base_url_batch"],
        help="FastAPI base URL (e.g., http://127.0.0.1:8001)",
    )
    BASE_URL = str(st.session_state.get("base_url_batch", BASE_URL)).strip()

st.subheader("Inputs")
protein_id = st.text_input("Protein ID (PDB)", value="1ABC", placeholder="e.g., 1ABC")
ligands_text = st.text_area(
    "Ligands (one SMILES per line)",
    value="CCO\nCCCBr\nc1ccccc1",
    height=150,
)
run = st.button("Run batch", type="primary")

results: List[Dict] = []

if run:
    ligands = [ln.strip() for ln in (ligands_text or "").splitlines() if ln.strip()]
    if not protein_id.strip() or not ligands:
        st.error("Provide a Protein ID and at least one ligand SMILES.")
    else:
        url = f"{BASE_URL.rstrip('/')}/batch_predict"
        payload = {"protein_id": protein_id.strip(), "ligands": ligands}
        with st.spinner("Running batch predictions..."):
            try:
                resp = requests.post(url, json=payload, timeout=120)
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
                        results = data.get("results", []) or []
                        if results:
                            st.subheader("Results")
                            st.dataframe(results, use_container_width=True)

                            # Prepare CSV
                            output = io.StringIO()
                            writer = csv.DictWriter(output, fieldnames=["ligand", "score", "explanation"])
                            writer.writeheader()
                            for row in results:
                                writer.writerow({
                                    "ligand": row.get("ligand", ""),
                                    "score": row.get("score", ""),
                                    "explanation": row.get("explanation", ""),
                                })
                            csv_bytes = output.getvalue().encode("utf-8")
                            st.download_button(
                                label="Download CSV",
                                data=csv_bytes,
                                file_name="batch_predict.csv",
                                mime="text/csv",
                            )

                        with st.expander("Raw response"):
                            st.code(resp.text, language="json")
