import os
import streamlit as st
import base64

# Page config
st.set_page_config(
    page_title="Molecular Binding Analysis", 
    page_icon="🧪", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e, #0f3460);
    }

    .stApp {
        background: transparent;
    }

    /* Hero Section */
    .hero-container {
        padding: 4rem 2rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 3rem;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    .hero-subtitle {
        font-size: 1.5rem;
        color: #ccd6f6;
        margin-bottom: 2rem;
    }

    /* Feature Cards */
    .feature-card {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: transform 0.3s ease, background 0.3s ease;
        height: 100%;
        text-align: center;
    }

    .feature-card:hover {
        transform: translateY(-10px);
        background: rgba(255, 255, 255, 0.08);
        border-color: #4facfe;
    }

    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
    }

    .feature-desc {
        color: #8892b0;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Gradient Button */
    .stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        font-weight: 700;
        transition: opacity 0.3s;
    }

    .stButton > button:hover {
        opacity: 0.9;
        color: white;
    }
    
    /* Hide Streamlit Menu/Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

local_css()

# Hero Section
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div style="padding-top: 50px;"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Molecular Binding Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle"> AI-powered protein-ligand scoring,docking,improvement and visualization platform.</p>', unsafe_allow_html=True)

with col2:
    if os.path.exists("hero.png"):
        st.image("hero.png", use_container_width=True)
    else:
        st.markdown('<div style="height: 300px; background: rgba(255,255,255,0.1); border-radius: 20px;"></div>', unsafe_allow_html=True)

st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)

# Feature Cards
st.markdown('<h2 style="text-align: center; color: white; margin-bottom: 2rem;">Core Capabilities</h2>', unsafe_allow_html=True)

f_col1, f_col2, f_col3 = st.columns(3)

with f_col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧬</div>
        <div class="feature-title">Fast Prediction</div>
        <div class="feature-desc">Predict binding scores for any ligand (SMILES) against specific protein targets in seconds.</div>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚓</div>
        <div class="feature-title">Molecular Docking</div>
        <div class="feature-desc">Predict Molecular Docking scores for any ligand with protien molecules to check the actual fit.</div>
    </div>
    """, unsafe_allow_html=True)

with f_col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔂</div>
        <div class="feature-title">Iterative Refinement</div>
        <div class="feature-desc">Modify the ligand and refine binding poses iteratively to improve binding affinity and molecular docking</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
backend_url = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8001")
st.sidebar.markdown("---")
st.sidebar.info("Select a page from the sidebar to begin analysis.")
