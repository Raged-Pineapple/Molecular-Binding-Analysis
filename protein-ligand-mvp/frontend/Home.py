import os
import streamlit as st

# Page config should be called only once in a multipage app
st.set_page_config(page_title="Protein–Ligand MVP", page_icon="🧪", layout="centered")

st.title("Protein–Ligand Scoring (MVP)")

st.markdown(
    """
Welcome to the Protein–Ligand MVP.

Use the pages in the left sidebar:
- Predict: Score a ligand (SMILES) against a protein ID and view 3D ligand.
- UploadProtein: Upload a PDB file and preview protein in 3D.

Configure your backend BASE_URL in each page's sidebar. Default is `http://127.0.0.1:8001`.
    """
)
