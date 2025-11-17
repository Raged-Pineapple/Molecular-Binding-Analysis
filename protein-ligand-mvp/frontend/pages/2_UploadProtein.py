import os
import json
import requests
import streamlit as st
from streamlit.components.v1 import html as st_html

try:
    import py3Dmol  # type: ignore
except Exception:  # pragma: no cover
    py3Dmol = None

st.title("UploadProtein")


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
    if "base_url_upload" not in st.session_state:
        st.session_state["base_url_upload"] = BASE_URL
    st.text_input(
        "Backend BASE_URL",
        key="base_url_upload",
        value=st.session_state["base_url_upload"],
        help="FastAPI base URL (e.g., http://127.0.0.1:8001)",
    )
    BASE_URL = str(st.session_state.get("base_url_upload", BASE_URL)).strip()

st.subheader("Upload protein (PDB)")
uploaded_pdb = st.file_uploader("Select a PDB file", type=["pdb"], key="pdb_uploader_page")
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
                    try:
                        pdb_text = uploaded_pdb.getvalue().decode('utf-8', errors='ignore')
                    except Exception:
                        pdb_text = ''
                    if pdb_text:
                        _render_protein_cartoon(pdb_text)
                    with st.expander("Raw upload response"):
                        st.code(json.dumps(res, indent=2), language="json")

st.caption("Set BASE_URL via sidebar, env PLMVP_BASE_URL, or Streamlit secrets.")
