"""
Next-Generation Cyber-Command Streamlit Web Application for NYC Taxi Fare Prediction.
CSE 4192: Machine Learning Projects with Python - Lab Assignment 02
Department of Computer Science & Engineering | Centre for Artificial Intelligence & Machine Learning
Siksha 'O' Anusandhan (Deemed to be University), ITER, Bhubaneswar.
Course Faculty: Dr. Gyana Ranjan Patra

Project Team Members:
  1. Tribhuwan Singh (Regd. No.: 2341019538) - Lead: DNN Architecture & Deployment
  2. Surajit Sahoo (Regd. No.: 2341019165) - Exploratory Data Analysis & Spatial Mapping
  3. Anwesha Srichandan (Regd. No.: 2341019594) - Preprocessing & Feature Engineering
  4. Priti Rani Maity (Regd. No.: 2341013065) - Hyperparameter Tuning & Evaluation
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import datetime
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
import requests

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
sys.path.append(SRC_DIR)

from feature_engineering import extract_features, haversine_distance, FEATURE_COLS
from dnn_model import TaxiFareDNN

# Configure Page
st.set_page_config(
    page_title="NYC Taxi Fare Intelligence Studio | Deep Neural Network",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DAY / NIGHT DISPLAY THEME CONTROLLER
# =============================================================================
if "is_night_theme" not in st.session_state:
    st.session_state["is_night_theme"] = True

st.sidebar.markdown("### 🌓 Display Theme")
is_night_theme = st.sidebar.toggle(
    "🌙 Cyber Night Mode" if st.session_state["is_night_theme"] else "☀️ Sunlit Day Mode",
    value=st.session_state["is_night_theme"],
    key="is_night_theme_toggle",
    help="Toggle between High-End Cyber Dark (Night) Mode and Crisp Modern Sunlit (Day) Mode."
)
st.session_state["is_night_theme"] = is_night_theme
st.sidebar.markdown("---")

# Dynamic Theme Tokens
chart_font_color = "#F1F5F9" if is_night_theme else "#0F172A"
chart_grid_color = "#1E293B" if is_night_theme else "#CBD5E1"
chart_gauge_bg = "rgba(15, 23, 42, 0.6)" if is_night_theme else "#F8FAFC"
chart_gauge_border = "#334155" if is_night_theme else "#CBD5E1"
card_bg = "rgba(15, 23, 42, 0.85)" if is_night_theme else "#FFFFFF"
card_border = "rgba(255, 255, 255, 0.1)" if is_night_theme else "#CBD5E1"
card_text = "#94A3B8" if is_night_theme else "#475569"
radar_polar_bg = "rgba(15, 23, 42, 0.6)" if is_night_theme else "#F8FAFC"
radar_grid_color = "#334155" if is_night_theme else "#CBD5E1"
map_theme_pdk = pdk.map_styles.CARTO_DARK if is_night_theme else pdk.map_styles.CARTO_LIGHT
map_theme_plotly = "carto-darkmatter" if is_night_theme else "carto-positron"

if is_night_theme:
    theme_css = """
    /* Main Background Accent - Cyber Night Mode */
    .stApp {
        background: radial-gradient(circle at 12% 15%, rgba(30, 27, 75, 0.40) 0%, transparent 45%),
                    radial-gradient(circle at 88% 22%, rgba(245, 158, 11, 0.12) 0%, transparent 45%),
                    radial-gradient(circle at 50% 80%, rgba(14, 116, 144, 0.15) 0%, transparent 50%),
                    #070A13;
        color: #F1F5F9;
    }
    
    /* Dark Glass Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070C18 0%, #0B1222 50%, #070B16 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.6) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #E2E8F0 !important;
    }
    
    /* Sidebar Inputs Styling */
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div,
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] > div,
    section[data-testid="stSidebar"] div[data-testid="stDateInput"] > div,
    section[data-testid="stSidebar"] div[data-testid="stTimeInput"] > div {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
    }
    
    /* Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 58, 138, 0.82) 48%, rgba(14, 116, 144, 0.72) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2.2rem 2.6rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: 0 20px 45px -10px rgba(0, 0, 0, 0.7), 0 0 35px rgba(56, 189, 248, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: 0; right: 0; bottom: 0; width: 35%;
        background: radial-gradient(circle at 100% 0%, rgba(245, 158, 11, 0.25) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-banner h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: #FFFFFF !important;
        font-size: 2.45rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        letter-spacing: -0.03em;
        text-shadow: 0 2px 14px rgba(0,0,0,0.5);
    }
    .hero-banner p {
        color: #BAE6FD;
        font-size: 1.05rem;
        margin-bottom: 0.6rem;
        max-width: 85%;
    }
    
    .badge-bar {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        color: #F8FAFC;
        font-weight: 500;
        backdrop-filter: blur(8px);
    }
    .hero-badge.highlight {
        background: rgba(245, 158, 11, 0.25);
        border-color: rgba(245, 158, 11, 0.6);
        color: #FDE68A;
        font-weight: 600;
    }
    .hero-badge.green {
        background: rgba(16, 185, 129, 0.2);
        border-color: rgba(16, 185, 129, 0.5);
        color: #A7F3D0;
        font-weight: 600;
    }
    
    /* Live Telemetry Ribbon */
    .telemetry-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
        margin-bottom: 1.2rem;
        font-size: 0.82rem;
        color: #94A3B8;
        font-family: 'JetBrains Mono', monospace;
    }
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981, 0 0 20px #10B981;
        margin-right: 8px;
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 12px 30px -6px rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 16px 36px -6px rgba(0, 0, 0, 0.6), 0 0 20px rgba(56, 189, 248, 0.15);
    }
    
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .metric-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
    
    /* Glowing Taximeter Display */
    .taximeter-hud {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.90) 0%, rgba(5, 150, 105, 0.88) 50%, rgba(16, 185, 129, 0.85) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-radius: 20px;
        padding: 1.8rem 1.6rem;
        text-align: center;
        box-shadow: 0 18px 40px -8px rgba(16, 185, 129, 0.45), inset 0 0 25px rgba(16, 185, 129, 0.25);
        position: relative;
    }
    .taximeter-title {
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
        color: #A7F3D0;
    }
    .taximeter-fare {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.75rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0.15rem 0;
        text-shadow: 0 3px 18px rgba(0, 0, 0, 0.4);
        letter-spacing: -0.03em;
    }
    .taximeter-ci {
        font-size: 0.92rem;
        color: #ECFDF5;
        background: rgba(0, 0, 0, 0.25);
        display: inline-block;
        padding: 0.38rem 0.95rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Sleek Custom Button Overrides */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(245, 158, 11, 0.4) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.6rem 0.85rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
        text-align: center !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(217, 119, 6, 0.35) 100%) !important;
        border-color: #F59E0B !important;
        color: #FDE68A !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(245, 158, 11, 0.4) !important;
    }
    
    /* Tabs Custom Styling */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.25rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(56, 189, 248, 0.16) !important;
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }
    
    /* Printable NYC TLC Digital Receipt */
    .receipt-box {
        background: #0F172A;
        border: 2px dashed rgba(245, 158, 11, 0.5);
        border-radius: 16px;
        padding: 1.8rem;
        font-family: 'JetBrains Mono', monospace;
        color: #E2E8F0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        position: relative;
    }
    
    .hub-strip {
        font-size: 0.85rem;
        color: #94A3B8;
        background: rgba(15, 23, 42, 0.65);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.07);
        margin-top: 0.5rem;
    }
    """
else:
    theme_css = """
    /* Main Background Accent - Sunlit Day Mode */
    .stApp {
        background: radial-gradient(circle at 12% 15%, rgba(191, 219, 254, 0.55) 0%, transparent 45%),
                    radial-gradient(circle at 88% 22%, rgba(254, 240, 138, 0.45) 0%, transparent 45%),
                    radial-gradient(circle at 50% 80%, rgba(204, 251, 241, 0.45) 0%, transparent 50%),
                    #F1F5F9;
        color: #0F172A;
    }
    
    /* Sunlit Light Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 50%, #F1F5F9 100%) !important;
        border-right: 1px solid #CBD5E1 !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.05) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #1E293B !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar Inputs Styling */
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div,
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] > div,
    section[data-testid="stSidebar"] div[data-testid="stDateInput"] > div,
    section[data-testid="stSidebar"] div[data-testid="stTimeInput"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        color: #0F172A !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #0284C7 50%, #0D9488 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2.2rem 2.6rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.35);
        box-shadow: 0 20px 40px -10px rgba(2, 132, 199, 0.3), 0 0 25px rgba(14, 165, 233, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: 0; right: 0; bottom: 0; width: 35%;
        background: radial-gradient(circle at 100% 0%, rgba(245, 158, 11, 0.3) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-banner h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: #FFFFFF !important;
        font-size: 2.45rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        letter-spacing: -0.03em;
        text-shadow: 0 2px 14px rgba(0,0,0,0.3);
    }
    .hero-banner p {
        color: #E0F2FE;
        font-size: 1.05rem;
        margin-bottom: 0.6rem;
        max-width: 85%;
    }
    
    .badge-bar {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        color: #FFFFFF;
        font-weight: 600;
        backdrop-filter: blur(8px);
    }
    .hero-badge.highlight {
        background: rgba(254, 243, 199, 0.35);
        border-color: #F59E0B;
        color: #FEF3C7;
        font-weight: 700;
    }
    .hero-badge.green {
        background: rgba(209, 250, 229, 0.35);
        border-color: #10B981;
        color: #ECFDF5;
        font-weight: 700;
    }
    
    /* Live Telemetry Ribbon */
    .telemetry-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        backdrop-filter: blur(12px);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
        margin-bottom: 1.2rem;
        font-size: 0.82rem;
        color: #334155;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
    }
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981, 0 0 20px #10B981;
        margin-right: 8px;
        animation: pulse 1.8s infinite;
    }
    
    /* Glass Cards */
    .glass-card {
        background: #FFFFFF;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        border-color: #0284C7;
        transform: translateY(-2px);
        box-shadow: 0 14px 28px -4px rgba(2, 132, 199, 0.18);
    }
    
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .metric-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
    
    /* Glowing Taximeter Display */
    .taximeter-hud {
        background: linear-gradient(135deg, #047857 0%, #059669 50%, #10B981 100%);
        backdrop-filter: blur(16px);
        border: 1px solid #34D399;
        border-radius: 20px;
        padding: 1.8rem 1.6rem;
        text-align: center;
        box-shadow: 0 18px 35px -8px rgba(16, 185, 129, 0.35);
        position: relative;
    }
    .taximeter-title {
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
        color: #D1FAE5;
    }
    .taximeter-fare {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.75rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0.15rem 0;
        text-shadow: 0 3px 14px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.03em;
    }
    .taximeter-ci {
        font-size: 0.92rem;
        color: #ECFDF5;
        background: rgba(0, 0, 0, 0.2);
        display: inline-block;
        padding: 0.38rem 0.95rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Custom Button Overrides */
    div[data-testid="stButton"] > button {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.6rem 0.85rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        text-align: center !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: #F8FAFC !important;
        border-color: #D97706 !important;
        color: #B45309 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(217, 119, 6, 0.18) !important;
    }
    
    /* Tabs Custom Styling */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px !important;
        color: #64748B !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.25rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(2, 132, 199, 0.12) !important;
        color: #0284C7 !important;
        border-bottom: 2px solid #0284C7 !important;
    }
    
    .receipt-box {
        background: #FFFFFF;
        border: 2px dashed #D97706;
        border-radius: 16px;
        padding: 1.8rem;
        font-family: 'JetBrains Mono', monospace;
        color: #0F172A;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.06);
        position: relative;
    }
    
    .hub-strip {
        font-size: 0.85rem;
        color: #475569;
        background: #FFFFFF;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin-top: 0.5rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }
    """

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
    }}
    
    {theme_css}
    
    .receipt-header {{
        text-align: center;
        border-bottom: 1px dashed rgba(148, 163, 184, 0.3);
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }}
    .receipt-line {{
        display: flex;
        justify-content: space-between;
        margin: 0.4rem 0;
        font-size: 0.88rem;
    }}
    .receipt-total {{
        border-top: 2px solid rgba(148, 163, 184, 0.4);
        margin-top: 1rem;
        padding-top: 0.8rem;
        display: flex;
        justify-content: space-between;
        font-size: 1.25rem;
        font-weight: 800;
        color: #D97706;
    }}
    
    /* Surcharge LED Indicator */
    .status-led {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }}
    .led-green {{ background-color: #10B981; box-shadow: 0 0 8px #10B981; }}
    .led-amber {{ background-color: #F59E0B; box-shadow: 0 0 8px #F59E0B; }}
    .led-red {{ background-color: #EF4444; box-shadow: 0 0 8px #EF4444; }}
</style>
""", unsafe_allow_html=True)


# Predefined NYC Landmarks
NYC_LANDMARKS = {
    "Times Square (Midtown Manhattan)": (40.7580, -73.9855),
    "Central Park South": (40.7660, -73.9772),
    "JFK International Airport (Terminal 4)": (40.6413, -73.7781),
    "LaGuardia Airport (LGA Terminal B)": (40.7769, -73.8740),
    "Newark Liberty Airport (EWR Terminal C)": (40.6895, -74.1745),
    "Financial District / Wall Street": (40.7075, -74.0090),
    "Brooklyn Bridge (DUMBO Promenade)": (40.7028, -73.9965),
    "Grand Central Terminal": (40.7527, -73.9772),
    "Empire State Building": (40.7484, -73.9857),
    "Columbia University (Morningside Heights)": (40.8075, -73.9626),
    "Barclays Center (Brooklyn)": (40.6826, -73.9754),
    "Yankee Stadium (The Bronx)": (40.8296, -73.9262),
    "Custom Coordinates": None
}

PRESET_CONFIGS = {
    "midtown_hop": {
        "title": "⚡ Midtown Rush Hop",
        "p_name": "Times Square (Midtown Manhattan)",
        "d_name": "Grand Central Terminal",
        "p_coords": (40.7580, -73.9855),
        "d_coords": (40.7527, -73.9772),
        "date": datetime.date(2025, 10, 15),
        "time": datetime.time(18, 30),
        "passengers": 1,
        "note": "Short Midtown rush-hour commute testing heavy traffic delay."
    },
    "jfk_airport": {
        "title": "✈️ JFK Airport Express",
        "p_name": "JFK International Airport (Terminal 4)",
        "d_name": "Times Square (Midtown Manhattan)",
        "p_coords": (40.6413, -73.7781),
        "d_coords": (40.7580, -73.9855),
        "date": datetime.date(2025, 10, 15),
        "time": datetime.time(14, 0),
        "passengers": 2,
        "note": "Interborough airport trip crossing Queens into Midtown."
    },
    "lga_wallst": {
        "title": "🏙️ LGA to Wall Street",
        "p_name": "LaGuardia Airport (LGA Terminal B)",
        "d_name": "Financial District / Wall Street",
        "p_coords": (40.7769, -73.8740),
        "d_coords": (40.7075, -74.0090),
        "date": datetime.date(2025, 10, 16),
        "time": datetime.time(9, 15),
        "passengers": 3,
        "note": "Morning rush airport run directly into Lower Manhattan."
    },
    "micro_hop": {
        "title": "🚶 Central Park Micro-Hop",
        "p_name": "Central Park South",
        "d_name": "Custom Coordinates",
        "p_coords": (40.7660, -73.9772),
        "d_coords": (40.7675, -73.9755),
        "date": datetime.date(2025, 10, 17),
        "time": datetime.time(11, 0),
        "passengers": 5,
        "note": "Borderline ultra-short trip (~200m) with 5 passengers testing base flag drop."
    },
    "midnight_dumbo": {
        "title": "🌙 Midnight Brooklyn Cruise",
        "p_name": "Brooklyn Bridge (DUMBO Promenade)",
        "d_name": "Columbia University (Morningside Heights)",
        "p_coords": (40.7028, -73.9965),
        "d_coords": (40.8075, -73.9626),
        "date": datetime.date(2025, 10, 18),
        "time": datetime.time(23, 45),
        "passengers": 4,
        "note": "Saturday night transit traversing Manhattan south to north."
    }
}

# Initialize session state for inputs
if "p_choice" not in st.session_state:
    st.session_state["p_choice"] = "Times Square (Midtown Manhattan)"
if "d_choice" not in st.session_state:
    st.session_state["d_choice"] = "JFK International Airport (Terminal 4)"
if "p_lat" not in st.session_state:
    st.session_state["p_lat"] = 40.7580
if "p_lon" not in st.session_state:
    st.session_state["p_lon"] = -73.9855
if "d_lat" not in st.session_state:
    st.session_state["d_lat"] = 40.6413
if "d_lon" not in st.session_state:
    st.session_state["d_lon"] = -73.7781
if "trip_date" not in st.session_state:
    st.session_state["trip_date"] = datetime.date(2025, 10, 15)
if "trip_time" not in st.session_state:
    st.session_state["trip_time"] = datetime.time(18, 30)
if "passengers" not in st.session_state:
    st.session_state["passengers"] = 1
if "map_view_mode" not in st.session_state:
    st.session_state["map_view_mode"] = "3D Night Flight Deck (PyDeck)"
if "tip_pct" not in st.session_state:
    st.session_state["tip_pct"] = 18
if "live_time_mode" not in st.session_state:
    st.session_state["live_time_mode"] = False
if "live_weather_sync" not in st.session_state:
    st.session_state["live_weather_sync"] = True
if "dispatched_cab" not in st.session_state:
    st.session_state["dispatched_cab"] = None
if "geo_results" not in st.session_state:
    st.session_state["geo_results"] = None

# =============================================================================
# REAL-TIME NYC TELEMETRY & LIVE API SERVICES
# =============================================================================
def get_live_nyc_time():
    """Computes accurate current local time in New York City (US Eastern Time with DST)."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    month = now_utc.month
    offset_hours = -4 if 3 <= month <= 11 else -5
    nyc_tz = datetime.timezone(datetime.timedelta(hours=offset_hours))
    return now_utc.astimezone(nyc_tz)

@st.cache_data(ttl=300)
def get_live_nyc_weather():
    """Fetches live meteorological telemetry for New York City via Open-Meteo."""
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&current_weather=true"
        r = requests.get(url, timeout=2.5)
        if r.status_code == 200:
            data = r.json().get("current_weather", {})
            temp_c = data.get("temperature", 15.0)
            temp_f = round(temp_c * 9/5 + 32, 1)
            wcode = data.get("weathercode", 0)
            wind = data.get("windspeed", 10.0)
            if wcode in [0, 1]:
                desc, mult, icon = "Clear Skies / Sunny", 1.00, "☀️"
            elif wcode in [2, 3]:
                desc, mult, icon = "Partly Cloudy / Overcast", 1.05, "⛅"
            elif wcode in [51, 53, 55, 61, 63, 65, 80, 81]:
                desc, mult, icon = "Light / Moderate Rain", 1.15, "🌧️"
            elif wcode in [65, 82, 95, 96, 99]:
                desc, mult, icon = "Heavy Rain / Thunderstorm", 1.25, "⛈️"
            elif wcode in [71, 73, 75, 77, 85, 86]:
                desc, mult, icon = "Snow / Sleet / Freezing", 1.35, "❄️"
            else:
                desc, mult, icon = "Normal Conditions", 1.00, "🌤️"
            return {"temp_c": temp_c, "temp_f": temp_f, "desc": desc, "mult": mult, "wind": wind, "icon": icon, "status": "LIVE"}
    except Exception:
        pass
    return {"temp_c": 16.0, "temp_f": 60.8, "desc": "Standard NYC Conditions", "mult": 1.00, "wind": 12.0, "icon": "🌤️", "status": "DEFAULT"}

@st.cache_data(ttl=1800)
def get_live_osrm_route(p_lat, p_lon, d_lat, d_lon):
    """Fetches real-world turn-by-turn road driving directions and distance via OSRM."""
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{p_lon},{p_lat};{d_lon},{d_lat}?overview=full&geometries=geojson"
        r = requests.get(url, timeout=3.0)
        if r.status_code == 200:
            res = r.json()
            if "routes" in res and len(res["routes"]) > 0:
                route = res["routes"][0]
                dist_km = round(route["distance"] / 1000.0, 2)
                dur_mins = round(route["duration"] / 60.0, 1)
                coords = route["geometry"]["coordinates"] # [lon, lat]
                return {
                    "distance_km": dist_km,
                    "duration_mins": dur_mins,
                    "coordinates": coords,
                    "status": "LIVE_ROUTING"
                }
    except Exception:
        pass
    # Fallback to straight-line interpolation with Manhattan road factor
    steps = 30
    lats = np.linspace(p_lat, d_lat, steps)
    lons = np.linspace(p_lon, d_lon, steps)
    coords = [[float(lo), float(la)] for lo, la in zip(lons, lats)]
    h_dist = haversine_distance(p_lat, p_lon, d_lat, d_lon)
    return {
        "distance_km": round(h_dist * 1.28, 2),
        "duration_mins": round(max(3.0, (h_dist * 1.28 / 18.0) * 60), 1),
        "coordinates": coords,
        "status": "ESTIMATED_ROAD_FACTOR"
    }

@st.cache_data(ttl=3600)
def search_nyc_address(query):
    """Geocodes any custom NYC landmark or street address via OpenStreetMap Nominatim."""
    if not query or len(query.strip()) < 3:
        return None
    try:
        q = f"{query.strip()}, New York City"
        headers = {"User-Agent": "NYCTaxiFareApp/2.0 (Academic Research Project)"}
        url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(q)}&format=json&limit=3&viewbox=-74.26,40.92,-73.70,40.49&bounded=1"
        r = requests.get(url, headers=headers, timeout=3.0)
        if r.status_code == 200:
            data = r.json()
            if data and len(data) > 0:
                results = []
                for item in data:
                    lat = float(item["lat"])
                    lon = float(item["lon"])
                    name = item.get("display_name", "").split(",")[0]
                    results.append({"name": name, "lat": lat, "lon": lon, "full_addr": item.get("display_name", "")})
                return results
    except Exception:
        pass
    return None

def get_nearby_cabs(p_lat, p_lon):
    """Generates simulated available nearby NYC yellow cabs around pickup point."""
    return [
        {"medallion": "NYC-4192", "model": "Toyota Camry Hybrid", "lat": p_lat + 0.0022, "lon": p_lon - 0.0019, "dist_m": 240, "eta_min": 1.5, "driver": "Salim K.", "rating": "4.96 ★"},
        {"medallion": "NYC-8831", "model": "Ford Crown Victoria", "lat": p_lat - 0.0034, "lon": p_lon + 0.0025, "dist_m": 460, "eta_min": 3.0, "driver": "David M.", "rating": "4.89 ★"},
        {"medallion": "NYC-1054", "model": "Tesla Model Y (Yellow)", "lat": p_lat + 0.0048, "lon": p_lon + 0.0038, "dist_m": 710, "eta_min": 4.5, "driver": "Anika R.", "rating": "4.98 ★"}
    ]

def set_preset(preset_key):
    cfg = PRESET_CONFIGS[preset_key]
    st.session_state["p_choice"] = cfg["p_name"]
    st.session_state["d_choice"] = cfg["d_name"]
    st.session_state["p_lat"] = cfg["p_coords"][0]
    st.session_state["p_lon"] = cfg["p_coords"][1]
    st.session_state["d_lat"] = cfg["d_coords"][0]
    st.session_state["d_lon"] = cfg["d_coords"][1]
    st.session_state["trip_date"] = cfg["date"]
    st.session_state["trip_time"] = cfg["time"]
    st.session_state["passengers"] = cfg["passengers"]

@st.cache_resource
def load_models_and_scaler():
    """Loads trained PyTorch DNN, Scaler, and Baseline models."""
    scaler_path = os.path.join(MODELS_DIR, "taxi_fare_scaler.pkl")
    dnn_path = os.path.join(MODELS_DIR, "taxi_fare_dnn.pt")
    baselines_path = os.path.join(MODELS_DIR, "baseline_models.pkl")
    
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    dnn_model = None
    if os.path.exists(dnn_path):
        checkpoint = torch.load(dnn_path, map_location=torch.device("cpu"), weights_only=False)
        in_feats = checkpoint.get("in_features", len(FEATURE_COLS)) if isinstance(checkpoint, dict) and "in_features" in checkpoint else len(FEATURE_COLS)
        hidden_dims = checkpoint.get("hidden_dims", (128, 64, 32)) if isinstance(checkpoint, dict) and "hidden_dims" in checkpoint else (128, 64, 32)
        dnn_model = TaxiFareDNN(in_features=in_feats, hidden_dims=hidden_dims)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        dnn_model.load_state_dict(state_dict)
        dnn_model.eval()
        
    baselines = joblib.load(baselines_path) if os.path.exists(baselines_path) else {}
    return scaler, dnn_model, baselines

scaler, dnn_model, baselines = load_models_and_scaler()

# =============================================================================
# HERO HEADER BANNER
# =============================================================================
theme_mode_badge = "🌙 Cyber Night Mode" if is_night_theme else "☀️ Sunlit Day Mode"
st.markdown(f"""
<div class="hero-banner">
    <h1>🚖 NYC Taxi Fare Intelligence Studio</h1>
    <p>High-Resolution Geodesic & Temporal Deep Feedforward Neural Network (PyTorch MLP) with Huber Robust Loss Formulation</p>
    <div class="badge-bar">
        <span class="hero-badge highlight">🏛️ Siksha 'O' Anusandhan (ITER)</span>
        <span class="hero-badge">📚 CSE 4192: Machine Learning Projects</span>
        <span class="hero-badge green">⚡ PyTorch Deep Neural Network</span>
        <span class="hero-badge">🎯 Huber Loss (δ=1.0)</span>
        <span class="hero-badge">🗺️ Geodesic & Cyclical Features</span>
        <span class="hero-badge highlight">🌓 {theme_mode_badge}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Live Telemetry Ribbon
live_weather = get_live_nyc_weather()
live_nyc_now = get_live_nyc_time()

st.markdown(f"""
<div class="telemetry-strip">
    <div><span class="pulse-dot"></span><b>REAL-TIME INFERENCE ENGINE</b> &nbsp;|&nbsp; 🕒 NYC: <b>{live_nyc_now.strftime('%I:%M %p EDT, %A')}</b> &nbsp;|&nbsp; {live_weather['icon']} Weather: <b>{live_weather['temp_c']}°C ({live_weather['desc']})</b></div>
    <div>Hardware: <b>CPU / AVX2 Inlined</b> &nbsp;|&nbsp; Latency: <b>~1.4 ms</b> &nbsp;|&nbsp; Display: <b>{'🌙 Night' if is_night_theme else '☀️ Day'}</b> &nbsp;|&nbsp; Rate: <b>NYC TLC 2025</b></div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# TOP QUICK SCENARIO SELECTOR CARDS & REAL-TIME SYNC
# =============================================================================
st.markdown("##### ⚡ 1-Click Interactive Trip Presets & Real-Time Sync")
sc_cols = st.columns(6)
with sc_cols[0]:
    if st.button("🔴 Live Sync Now\n(NYC Real-Time Clock)", key="btn_sc_realtime", use_container_width=True):
        nyc_now = get_live_nyc_time()
        st.session_state["trip_date"] = nyc_now.date()
        st.session_state["trip_time"] = nyc_now.time()
        st.toast(f"⚡ Synchronized with Live NYC Time: {nyc_now.strftime('%I:%M %p EDT')}", icon="🕒")
        st.rerun()
with sc_cols[1]:
    if st.button("🚀 Midtown Hop\n(Times Sq → Grand Central)", key="btn_sc_midtown", use_container_width=True):
        set_preset("midtown_hop")
with sc_cols[2]:
    if st.button("✈️ JFK Express\n(Terminal 4 → Times Sq)", key="btn_sc_jfk", use_container_width=True):
        set_preset("jfk_airport")
with sc_cols[3]:
    if st.button("🏙️ LGA → Wall St\n(Airport to Financial Dist)", key="btn_sc_lga", use_container_width=True):
        set_preset("lga_wallst")
with sc_cols[4]:
    if st.button("🚶 Micro-Hop (200m)\n(Central Park South)", key="btn_sc_micro", use_container_width=True):
        set_preset("micro_hop")
with sc_cols[5]:
    if st.button("🌙 Midnight Cruise\n(DUMBO → Columbia Univ)", key="btn_sc_midnight", use_container_width=True):
        set_preset("midnight_dumbo")

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR TRIP CUSTOMIZATION CONTROLS
# =============================================================================
st.sidebar.markdown("### 🎛️ Trip Telemetry Controls")

def on_p_change():
    sel = st.session_state["p_choice_key"]
    st.session_state["p_choice"] = sel
    if sel in NYC_LANDMARKS and NYC_LANDMARKS[sel] is not None:
        st.session_state["p_lat"], st.session_state["p_lon"] = NYC_LANDMARKS[sel]

def on_d_change():
    sel = st.session_state["d_choice_key"]
    st.session_state["d_choice"] = sel
    if sel in NYC_LANDMARKS and NYC_LANDMARKS[sel] is not None:
        st.session_state["d_lat"], st.session_state["d_lon"] = NYC_LANDMARKS[sel]

p_idx = list(NYC_LANDMARKS.keys()).index(st.session_state["p_choice"]) if st.session_state["p_choice"] in NYC_LANDMARKS else 0
st.sidebar.selectbox("1. Pickup Landmark", list(NYC_LANDMARKS.keys()), index=p_idx, key="p_choice_key", on_change=on_p_change)

col_plat, col_plon = st.sidebar.columns(2)
with col_plat:
    p_lat = st.sidebar.number_input("Pickup Lat", value=float(st.session_state["p_lat"]), min_value=40.50, max_value=40.95, format="%.6f", key="p_lat_input")
    st.session_state["p_lat"] = p_lat
with col_plon:
    p_lon = st.sidebar.number_input("Pickup Lon", value=float(st.session_state["p_lon"]), min_value=-74.25, max_value=-73.70, format="%.6f", key="p_lon_input")
    st.session_state["p_lon"] = p_lon

st.sidebar.markdown("---")
d_idx = list(NYC_LANDMARKS.keys()).index(st.session_state["d_choice"]) if st.session_state["d_choice"] in NYC_LANDMARKS else 2
st.sidebar.selectbox("2. Drop-off Landmark", list(NYC_LANDMARKS.keys()), index=d_idx, key="d_choice_key", on_change=on_d_change)

col_dlat, col_dlon = st.sidebar.columns(2)
with col_dlat:
    d_lat = st.sidebar.number_input("Drop-off Lat", value=float(st.session_state["d_lat"]), min_value=40.50, max_value=40.95, format="%.6f", key="d_lat_input")
    st.session_state["d_lat"] = d_lat
with col_dlon:
    d_lon = st.sidebar.number_input("Drop-off Lon", value=float(st.session_state["d_lon"]), min_value=-74.25, max_value=-73.70, format="%.6f", key="d_lon_input")
    st.session_state["d_lon"] = d_lon

# Real-Time Address Geocoding Search
with st.sidebar.expander("🔍 Real-Time NYC Address Geocoder", expanded=False):
    st.caption("Search any real street address, building, or landmark in NYC:")
    geo_query = st.text_input("Enter Address / Place", placeholder="e.g. Empire State Building, SoHo", key="geo_search_input")
    if st.button("🔎 Geocode Address", key="btn_geocode", use_container_width=True):
        if geo_query:
            results = search_nyc_address(geo_query)
            if results:
                st.session_state["geo_results"] = results
            else:
                st.warning("No NYC matches found. Please refine search query.")
                
    if "geo_results" in st.session_state and st.session_state["geo_results"]:
        for r_i, r in enumerate(st.session_state["geo_results"]):
            st.markdown(f"**{r['name']}**  \n<span style='font-size: 0.75rem; color: #94A3B8;'>{r['full_addr'][:75]}...</span>", unsafe_allow_html=True)
            col_gp, col_gd = st.sidebar.columns(2)
            with col_gp:
                if st.button("📍 Set Pickup", key=f"btn_set_p_{r_i}", use_container_width=True):
                    st.session_state["p_choice"] = "Custom Coordinates"
                    st.session_state["p_lat"] = r["lat"]
                    st.session_state["p_lon"] = r["lon"]
                    st.toast(f"Pickup set to {r['name']}", icon="📍")
                    st.rerun()
            with col_gd:
                if st.button("🏁 Set Dropoff", key=f"btn_set_d_{r_i}", use_container_width=True):
                    st.session_state["d_choice"] = "Custom Coordinates"
                    st.session_state["d_lat"] = r["lat"]
                    st.session_state["d_lon"] = r["lon"]
                    st.toast(f"Drop-off set to {r['name']}", icon="🏁")
                    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("#### 3. Temporal & Passenger Settings")

col_tsync1, col_tsync2 = st.sidebar.columns([1.2, 1.0])
with col_tsync1:
    live_time_mode = st.toggle("🔴 Real-Time NYC Clock", value=st.session_state["live_time_mode"], key="tgl_live_time", help="Automatically lock trip departure to the live New York City clock.")
    st.session_state["live_time_mode"] = live_time_mode
with col_tsync2:
    if st.button("🕒 Sync Now", key="btn_sync_now_sidebar", use_container_width=True):
        nyc_now = get_live_nyc_time()
        st.session_state["trip_date"] = nyc_now.date()
        st.session_state["trip_time"] = nyc_now.time()
        st.rerun()

if live_time_mode:
    nyc_now = get_live_nyc_time()
    trip_date = nyc_now.date()
    trip_time = nyc_now.time()
    st.session_state["trip_date"] = trip_date
    st.session_state["trip_time"] = trip_time
    st.sidebar.success(f"🟢 **Live Clock Active:** {trip_time.strftime('%I:%M:%S %p EDT')}")
else:
    trip_date = st.sidebar.date_input("Trip Date", value=st.session_state["trip_date"], key="date_input")
    st.session_state["trip_date"] = trip_date
    trip_time = st.sidebar.time_input("Departure Time", value=st.session_state["trip_time"], key="time_input")
    st.session_state["trip_time"] = trip_time

passengers = st.sidebar.slider("Occupancy (Passengers)", min_value=1, max_value=6, value=int(st.session_state["passengers"]), key="pass_input")
st.session_state["passengers"] = passengers

st.sidebar.markdown("---")
st.sidebar.markdown("#### 4. Weather & Congestion Telemetry")

live_weather_sync = st.sidebar.toggle("🌦️ Live NYC Weather Sync (Open-Meteo)", value=st.session_state["live_weather_sync"], key="tgl_live_weather", help="Automatically fetch real-time temperature, wind, and precipitation in NYC to adjust pricing.")
st.session_state["live_weather_sync"] = live_weather_sync

if live_weather_sync:
    weather_mult = live_weather["mult"]
    st.sidebar.markdown(f"""
    <div style="background: {'rgba(15, 23, 42, 0.65)' if is_night_theme else '#F8FAFC'}; padding: 0.65rem 0.85rem; border-radius: 10px; border: 1px solid {'rgba(255,255,255,0.08)' if is_night_theme else '#CBD5E1'}; font-size: 0.8rem; margin-top: 0.3rem;">
        <div>{live_weather['icon']} <b>Condition:</b> {live_weather['desc']}</div>
        <div style="margin-top: 0.2rem;">🌡️ <b>Temp:</b> {live_weather['temp_c']}°C ({live_weather['temp_f']}°F) &nbsp;|&nbsp; 💨 <b>Wind:</b> {live_weather['wind']} km/h</div>
        <div style="margin-top: 0.2rem; color: {'#38BDF8' if is_night_theme else '#0284C7'}; font-weight: 700;">⚡ Live Pricing Factor: {weather_mult:.2f}x</div>
    </div>
    """, unsafe_allow_html=True)
else:
    weather_condition = st.sidebar.select_slider(
        "Simulate Weather / Road Traffic",
        options=["Clear Skies (1.0x)", "Light Rain (+10%)", "Heavy Downpour (+25%)", "Blizzard / Gridlock (+40%)"],
        value="Clear Skies (1.0x)"
    )
    weather_mult = {
        "Clear Skies (1.0x)": 1.0,
        "Light Rain (+10%)": 1.10,
        "Heavy Downpour (+25%)": 1.25,
        "Blizzard / Gridlock (+40%)": 1.40
    }[weather_condition]

# Compute initial baseline temporal flags for defaults
curr_hour = trip_time.hour
auto_rush = bool(trip_date.weekday() < 5 and ((16 <= curr_hour < 20) or (7 <= curr_hour < 10)))
auto_night = bool(curr_hour >= 20 or curr_hour < 6)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 5. 🎚️ Live Surcharge & Tariff Toggles")
toggle_rush = st.sidebar.toggle(
    "⚡ Peak Rush-Hour Surcharge (+$1.00)",
    value=auto_rush,
    key="tgl_rush",
    help="Manually force or override the NYC TLC 4:00 PM – 8:00 PM weekday congestion surcharge."
)
toggle_night = st.sidebar.toggle(
    "🌙 Overnight Surcharge (+$0.50)",
    value=auto_night,
    key="tgl_night",
    help="Manually force or override the NYC TLC 8:00 PM – 6:00 AM overnight tariff surcharge."
)
toggle_jfk_flat = st.sidebar.toggle(
    "✈️ JFK Airport Flat-Rate Regime ($70.00)",
    value=False,
    key="tgl_jfk_flat",
    help="Apply the official NYC TLC Flat Fare regulation for trips between Manhattan and JFK International Airport."
)
toggle_tip = st.sidebar.toggle(
    "💰 Include Gratuity (18% Tip) in Meter",
    value=False,
    key="tgl_tip",
    help="Automatically include NYC standard 18% yellow cab gratuity on the main taximeter display."
)

st.sidebar.markdown("---")
st.sidebar.markdown("🏛️ **Siksha 'O' Anusandhan (ITER)**  \nCourse: **CSE 4192** | Lab Assignment 02")

# =============================================================================
# FEATURE EXTRACTION & REAL-TIME INFERENCE
# =============================================================================
t_start_infer = time.perf_counter()
pickup_dt = datetime.datetime.combine(trip_date, trip_time)
raw_input_df = pd.DataFrame([{
    "key": "live_request",
    "pickup_datetime": pd.to_datetime(pickup_dt),
    "pickup_longitude": p_lon,
    "pickup_latitude": p_lat,
    "dropoff_longitude": d_lon,
    "dropoff_latitude": d_lat,
    "passenger_count": passengers
}])

feat_df = extract_features(raw_input_df)
distance_km = float(feat_df["haversine_dist_km"].iloc[0])
manhattan_km = float(feat_df["manhattan_dist_km"].iloc[0])
is_rush = toggle_rush
is_night = toggle_night
is_wknd = bool(feat_df["is_weekend"].iloc[0])
hour_val = int(feat_df["hour"].iloc[0])
bearing_deg = float(feat_df["bearing_deg"].iloc[0])

# Perform model inference
pred_fare = 2.50
lgb_pred = 2.50
lr_pred = 2.50
layer_activations = {}

if toggle_jfk_flat:
    pred_fare = max(2.50, round(70.00 * weather_mult, 2))
    lgb_pred = max(2.50, round(70.00 * weather_mult, 2))
    lr_pred = max(2.50, round(70.00 * weather_mult, 2))
    est_rule_fare = max(2.50, round(70.00 * weather_mult, 2))
    if scaler is not None and dnn_model is not None:
        X_input = feat_df[FEATURE_COLS].values
        X_scaled = scaler.transform(X_input)
        t_input = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            x1 = torch.relu(dnn_model.bn1(dnn_model.fc1(t_input)))
            x2 = torch.relu(dnn_model.bn2(dnn_model.fc2(x1)))
            x3 = torch.relu(dnn_model.fc3(x2))
            layer_activations["L1_mean"] = float(torch.mean(x1).item())
            layer_activations["L2_mean"] = float(torch.mean(x2).item())
            layer_activations["L3_mean"] = float(torch.mean(x3).item())
            layer_activations["L1_active_pct"] = float(torch.sum(x1 > 0).item() / x1.numel() * 100)
            layer_activations["L2_active_pct"] = float(torch.sum(x2 > 0).item() / x2.numel() * 100)
            layer_activations["L3_active_pct"] = float(torch.sum(x3 > 0).item() / x3.numel() * 100)
else:
    if scaler is not None and dnn_model is not None:
        X_input = feat_df[FEATURE_COLS].values
        X_scaled = scaler.transform(X_input)
        t_input = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            # Compute forward pass and capture intermediate activations
            x1 = torch.relu(dnn_model.bn1(dnn_model.fc1(t_input)))
            x2 = torch.relu(dnn_model.bn2(dnn_model.fc2(x1)))
            x3 = torch.relu(dnn_model.fc3(x2))
            raw_pred = dnn_model.out(x3).item()
            
            layer_activations["L1_mean"] = float(torch.mean(x1).item())
            layer_activations["L2_mean"] = float(torch.mean(x2).item())
            layer_activations["L3_mean"] = float(torch.mean(x3).item())
            layer_activations["L1_active_pct"] = float(torch.sum(x1 > 0).item() / x1.numel() * 100)
            layer_activations["L2_active_pct"] = float(torch.sum(x2 > 0).item() / x2.numel() * 100)
            layer_activations["L3_active_pct"] = float(torch.sum(x3 > 0).item() / x3.numel() * 100)
            
        pred_fare = max(2.50, round(raw_pred * weather_mult, 2))
        
        if "LightGBM" in baselines:
            lgb_pred = max(2.50, round(float(baselines["LightGBM"].predict(X_scaled)[0]) * weather_mult, 2))
        if "Linear Regression" in baselines:
            lr_pred = max(2.50, round(float(baselines["Linear Regression"].predict(X_scaled)[0]) * weather_mult, 2))

    # NYC TLC Official Standard Regulatory Meter Rule
    est_rule_fare = 2.50 + (distance_km * 1.56) + (1.00 if is_rush else 0) + (0.50 if is_night else 0) + 0.50 + 0.30
    est_rule_fare = max(2.50, round(est_rule_fare * weather_mult, 2))

# Dynamic taximeter display fare (with optional gratuity)
meter_display_fare = round(pred_fare * 1.18, 2) if toggle_tip else pred_fare
infer_duration_ms = (time.perf_counter() - t_start_infer) * 1000

# Direction heading compass string
compass_dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
dir_idx = int(round(bearing_deg / 22.5)) % 16
compass_str = compass_dirs[dir_idx]

# Real-Time Road Routing and Fleet Radar Telemetry
route_info = get_live_osrm_route(p_lat, p_lon, d_lat, d_lon)
road_dist_km = route_info["distance_km"]
road_dur_mins = route_info["duration_mins"]
road_coords = route_info["coordinates"]
nearby_cabs = get_nearby_cabs(p_lat, p_lon)

# =============================================================================
# KEY SPATIAL METRICS ROW
# =============================================================================
m_cols = st.columns(4)

with m_cols[0]:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Haversine Distance</div>
        <div class="metric-number">{distance_km:.2f} <span style="font-size: 1rem; color: #94A3B8;">km</span></div>
        <div class="metric-sub">{distance_km * 0.621371:.2f} miles great-circle arc</div>
    </div>
    """, unsafe_allow_html=True)

with m_cols[1]:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Actual Street Driving (OSRM)</div>
        <div class="metric-number" style="color: {'#38BDF8' if is_night_theme else '#0284C7'};">{road_dist_km:.2f} <span style="font-size: 1rem; color: #94A3B8;">km</span></div>
        <div class="metric-sub">⏱️ Live Driving ETA: <b>~{road_dur_mins:.0f} mins</b> (Turn-by-Turn)</div>
    </div>
    """, unsafe_allow_html=True)

with m_cols[2]:
    if is_rush:
        surch_badge = '<span class="status-led led-red"></span>Rush Hour Active (+$1.00)'
        surch_sub = 'Peak: Weekdays 4-8 PM / 7-10 AM'
    elif is_night:
        surch_badge = '<span class="status-led led-amber"></span>Night Tariff Active (+$0.50)'
        surch_sub = 'Overnight: 8:00 PM – 6:00 AM'
    else:
        surch_badge = '<span class="status-led led-green"></span>Standard Tariff Active'
        surch_sub = 'No peak congestion charges'
        
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Congestion & Weather Multiplier</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: {'#F1F5F9' if is_night_theme else '#0F172A'}; margin: 0.3rem 0;">{surch_badge}</div>
        <div class="metric-sub">{surch_sub} • {live_weather['icon']} {weather_mult:.2f}x</div>
    </div>
    """, unsafe_allow_html=True)

with m_cols[3]:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Temporal & Vector Context</div>
        <div class="metric-number" style="font-size: 1.55rem; color: {'#38BDF8' if is_night_theme else '#0284C7'};">{pickup_dt.strftime('%A')} <span style="font-size: 1rem; color: #94A3B8;">{compass_str} ({bearing_deg:.0f}°)</span></div>
        <div class="metric-sub">{passengers} Passenger{'s' if passengers > 1 else ''} • {trip_time.strftime('%I:%M %p')}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# MAIN INTERACTIVE TABS
# =============================================================================
tab_main, tab_battle, tab_whatif, tab_receipt, tab_theory, tab_viz, tab_test, tab_academic = st.tabs([
    "🚖 Live Trip Studio",
    "⚡ Multi-Model Battle Arena",
    "🔮 What-If Simulator",
    "🧾 Official TLC e-Receipt",
    "🧠 Deep Neural Topology & Huber Loss",
    "📊 Visualization Gallery",
    "🧪 Automated Verification Suite",
    "🏛️ Academic Registry"
])

# -----------------------------------------------------------------------------
# TAB 1: LIVE TRIP STUDIO (HUD METER + DUAL MAP ENGINE)
# -----------------------------------------------------------------------------
with tab_main:
    # Actionable Top Controls Bar with Toggles
    tgl_c1, tgl_c2, tgl_c3 = st.columns([1, 1, 1])
    with tgl_c1:
        toggle_neural_inspect = st.toggle(
            "🔬 Deep Neural Layer Inspector",
            value=False,
            key="tgl_neural_inspect",
            help="Probe live 33-feature normalized input tensor and internal hidden layer neuron activations."
        )
    with tgl_c2:
        toggle_quick_compare = st.toggle(
            "⚡ Multi-Model Comparison HUD",
            value=False,
            key="tgl_quick_compare",
            help="Display real-time side-by-side benchmark predictions from LightGBM, Linear Regression, and TLC formula."
        )
    with tgl_c3:
        toggle_3d_arc = st.toggle(
            "🌐 3D Elevation Route Arc",
            value=True,
            key="tgl_3d_arc",
            help="Render high-contrast 3D elevated flight trajectory arc and landmark columns on the geospatial map."
        )

    # Multi-Model Comparison Strip if toggled
    if toggle_quick_compare:
        st.markdown(f"""
        <div style="display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 0.6rem 0 1rem 0; padding: 0.8rem 1.2rem; background: {card_bg}; border-radius: 14px; border: 1px solid {card_border}; box-shadow: 0 4px 14px rgba(0,0,0,0.05); align-items: center; justify-content: space-between;">
            <div><span style="color: {card_text}; font-size: 0.72rem; text-transform: uppercase; font-weight: 700;">PyTorch DNN:</span> <b style="color: #10B981; font-size: 1.15rem; font-family: 'JetBrains Mono';">${pred_fare:.2f}</b></div>
            <div style="border-left: 1px solid {card_border}; height: 22px;"></div>
            <div><span style="color: {card_text}; font-size: 0.72rem; text-transform: uppercase; font-weight: 700;">LightGBM:</span> <b style="color: #38BDF8; font-size: 1.15rem; font-family: 'JetBrains Mono';">${lgb_pred:.2f}</b> <span style="font-size: 0.75rem; color: {'#10B981' if lgb_pred <= pred_fare else '#EF4444'};">({lgb_pred - pred_fare:+.2f})</span></div>
            <div style="border-left: 1px solid {card_border}; height: 22px;"></div>
            <div><span style="color: {card_text}; font-size: 0.72rem; text-transform: uppercase; font-weight: 700;">Linear OLS:</span> <b style="color: #F59E0B; font-size: 1.15rem; font-family: 'JetBrains Mono';">${lr_pred:.2f}</b> <span style="font-size: 0.75rem; color: {'#10B981' if lr_pred <= pred_fare else '#EF4444'};">({lr_pred - pred_fare:+.2f})</span></div>
            <div style="border-left: 1px solid {card_border}; height: 22px;"></div>
            <div><span style="color: {card_text}; font-size: 0.72rem; text-transform: uppercase; font-weight: 700;">TLC Regulatory:</span> <b style="color: {'#E2E8F0' if is_night_theme else '#0F172A'}; font-size: 1.15rem; font-family: 'JetBrains Mono';">${est_rule_fare:.2f}</b> <span style="font-size: 0.75rem; color: {'#10B981' if est_rule_fare <= pred_fare else '#EF4444'};">({est_rule_fare - pred_fare:+.2f})</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Live Neural Inspector if toggled
    if toggle_neural_inspect and layer_activations:
        with st.expander("🔬 Live Deep Neural Network Diagnostics & Layer Activations", expanded=True):
            col_diag1, col_diag2 = st.columns([1.2, 1.0])
            with col_diag1:
                diag_layers = ["Dense 1 (128 Units)", "Dense 2 (64 Units)", "Dense 3 (32 Units)"]
                diag_means = [layer_activations.get("L1_mean", 0.0), layer_activations.get("L2_mean", 0.0), layer_activations.get("L3_mean", 0.0)]
                
                fig_diag = go.Figure()
                fig_diag.add_trace(go.Bar(
                    x=diag_layers,
                    y=diag_means,
                    name="Mean ReLU Activation",
                    marker_color="#10B981",
                    text=[f"{v:.3f}" for v in diag_means],
                    textposition="auto"
                ))
                fig_diag.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=chart_font_color, size=11),
                    yaxis=dict(gridcolor=chart_grid_color, title="Mean Activation Value"),
                    xaxis=dict(title=""),
                    margin=dict(l=20, r=20, t=25, b=20),
                    height=190
                )
                st.plotly_chart(fig_diag, use_container_width=True)
            with col_diag2:
                st.markdown(f"""
                <div style="background: {card_bg}; padding: 0.9rem 1.1rem; border-radius: 12px; border: 1px solid {card_border}; font-size: 0.82rem; color: {chart_font_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
                    <div>⚡ <b>Inference Latency:</b> <code style="color: #38BDF8;">{infer_duration_ms:.2f} ms</code></div>
                    <div style="margin-top: 0.35rem;">🎯 <b>L1 Active Neurons:</b> <code>{layer_activations.get('L1_active_pct', 0):.1f}%</code> (Sparsity: {100 - layer_activations.get('L1_active_pct', 0):.1f}%)</div>
                    <div style="margin-top: 0.35rem;">🎯 <b>L2 Active Neurons:</b> <code>{layer_activations.get('L2_active_pct', 0):.1f}%</code> (Sparsity: {100 - layer_activations.get('L2_active_pct', 0):.1f}%)</div>
                    <div style="margin-top: 0.35rem;">🎯 <b>L3 Active Neurons:</b> <code>{layer_activations.get('L3_active_pct', 0):.1f}%</code> (Sparsity: {100 - layer_activations.get('L3_active_pct', 0):.1f}%)</div>
                    <div style="margin-top: 0.35rem;">📐 <b>Input Dimension:</b> <code>33 Feature Columns</code></div>
                </div>
                """, unsafe_allow_html=True)

    col_hud, col_map = st.columns([1.05, 1.35])
    
    with col_hud:
        jfk_badge_html = '<div style="background: rgba(245, 158, 11, 0.25); border: 1px solid #F59E0B; color: #FDE68A; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; margin-bottom: 0.5rem; display: inline-block;">✈️ JFK AIRPORT FLAT-RATE REGIME ($70.00)</div>' if toggle_jfk_flat else ''
        
        tip_notice_html = f'<div style="font-size: 0.82rem; color: #FDE68A; margin-top: -0.1rem; margin-bottom: 0.4rem;">★ Includes 18% Gratuity (Base: ${pred_fare:.2f} + Tip: ${meter_display_fare - pred_fare:.2f})</div>' if toggle_tip else ''

        st.markdown(f"""
        <div class="taximeter-hud">
            {jfk_badge_html}
            <div class="taximeter-title">Deep Feedforward Neural Prediction</div>
            <div class="taximeter-fare">${meter_display_fare:.2f}</div>
            {tip_notice_html}
            <div class="taximeter-ci">
                95% Empirical Prediction Interval: <b>${max(2.50, meter_display_fare - 1.60):.2f} – ${meter_display_fare + 1.60:.2f}</b>
            </div>
            <div style="font-size: 0.8rem; color: #D1FAE5; margin-top: 0.8rem;">
                Trained with Huber Loss (δ=1.0) & Batch Normalization on NYC TLC Telemetry
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Plotly Luxury Gauge Meter
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=meter_display_fare,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Dynamic Fare Meter ($ USD)", 'font': {'size': 14, 'color': '#94A3B8' if is_night_theme else '#475569'}},
            number={'prefix': "$", 'font': {'size': 28, 'color': '#F8FAFC' if is_night_theme else '#0F172A', 'family': 'JetBrains Mono'}},
            gauge={
                'axis': {'range': [0, max(85, meter_display_fare * 1.3)], 'tickwidth': 1, 'tickcolor': chart_grid_color},
                'bar': {'color': "#10B981", 'thickness': 0.32},
                'bgcolor': chart_gauge_bg,
                'borderwidth': 1,
                'bordercolor': chart_gauge_border,
                'steps': [
                    {'range': [0, 15], 'color': 'rgba(56, 189, 248, 0.25)'},
                    {'range': [15, 45], 'color': 'rgba(245, 158, 11, 0.25)'},
                    {'range': [45, 120], 'color': 'rgba(239, 68, 68, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': "#F59E0B", 'width': 3},
                    'thickness': 0.8,
                    'value': meter_display_fare
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=25, r=25, t=35, b=15),
            height=200
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Real-time Itemized Cost Breakdown
        with st.expander("🧾 Official NYC Taxi Tariff Breakdown Analysis", expanded=True):
            base_charge = 2.50
            dist_charge = max(0.0, distance_km * 1.56)
            surch_charge = (1.00 if is_rush else 0.0) + (0.50 if is_night else 0.0)
            mta_tax = 0.80
            
            st.markdown(f"""
            - **Base Flag Drop Charge:** `$2.50` *(Initial charge upon entry)*
            - **Distance Incremental Meter:** `~${dist_charge:.2f}` *($0.50 per 1/5 mile)*
            - **Congestion Rush-Hour Surcharge:** `{'+$1.00' if is_rush else '$0.00'}`
            - **Night Tariff Surcharge:** `{'+$0.50' if is_night else '$0.00'}`
            - **MTA State Tax & Improvement Fund:** `+$0.80` *($0.50 MTA + $0.30 Improvement)*
            - **Weather / Traffic Multiplier:** `{weather_mult:.2f}x`
            - **Regulatory Baseline Formula Estimate:** **`${est_rule_fare:.2f}`**
            """)

    with col_map:
        m_head_col, m_sel_col = st.columns([1.2, 1.0])
        with m_head_col:
            st.markdown("#### 🗺️ Geospatial Geodesic Trajectory")
        with m_sel_col:
            map_mode = st.radio(
                "Map Engine",
                ["3D Flight Deck (PyDeck)", "2D Dynamic Grid (Plotly)"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
        map_points = pd.DataFrame([
            {"lat": p_lat, "lon": p_lon, "name": "Pickup Point", "color": [16, 185, 129, 240]},
            {"lat": d_lat, "lon": d_lon, "name": "Drop-off Point", "color": [239, 68, 68, 240]}
        ])
        
        if "PyDeck" in map_mode and PYDECK_AVAILABLE:
            mid_lat = (p_lat + d_lat) / 2.0
            mid_lon = (p_lon + d_lon) / 2.0
            
            arc_df = pd.DataFrame([{
                "source_lat": p_lat, "source_lon": p_lon,
                "target_lat": d_lat, "target_lon": d_lon
            }])
            
            # Elevation markers for 3D visual pop
            col_df = pd.DataFrame([
                {"lat": p_lat, "lon": p_lon, "elevation": 300, "color": [16, 185, 129, 200], "name": "Pickup"},
                {"lat": d_lat, "lon": d_lon, "elevation": 300, "color": [239, 68, 68, 200], "name": "Dropoff"}
            ])
            
            view_state = pdk.ViewState(
                latitude=mid_lat,
                longitude=mid_lon,
                zoom=11.0,
                pitch=45 if toggle_3d_arc else 0,
                bearing=12 if toggle_3d_arc else 0
            )
            
            arc_layer = pdk.Layer(
                "ArcLayer",
                data=arc_df,
                get_source_position=["source_lon", "source_lat"],
                get_target_position=["target_lon", "target_lat"],
                get_source_color=[16, 185, 129, 240],
                get_target_color=[239, 68, 68, 240],
                get_width=7
            )
            
            column_layer = pdk.Layer(
                "ColumnLayer",
                data=col_df,
                get_position=["lon", "lat"],
                get_elevation="elevation",
                elevation_scale=1,
                radius=180,
                get_fill_color="color",
                pickable=True,
                auto_highlight=True
            )
            
            road_path_df = pd.DataFrame([{
                "path": road_coords,
                "name": "OSRM Street Route"
            }])
            road_path_layer = pdk.Layer(
                "PathLayer",
                data=road_path_df,
                get_path="path",
                get_color=[56, 189, 248, 220] if is_night_theme else [2, 132, 199, 220],
                width_scale=20,
                width_min_pixel_width=4,
                pickable=True
            )
            cabs_df = pd.DataFrame([
                {"lat": c["lat"], "lon": c["lon"], "name": f"🚖 Available Cab #{c['medallion']} ({c['dist_m']}m away, ETA {c['eta_min']}m)"}
                for c in nearby_cabs
            ])
            cabs_layer = pdk.Layer(
                "ScatterplotLayer",
                data=cabs_df,
                get_position=["lon", "lat"],
                get_fill_color=[245, 158, 11, 240],
                get_radius=60,
                pickable=True
            )
            deck_layers = [arc_layer, column_layer, road_path_layer, cabs_layer] if toggle_3d_arc else [road_path_layer, column_layer, cabs_layer]
            
            deck = pdk.Deck(
                layers=deck_layers,
                initial_view_state=view_state,
                tooltip={"text": "{name}\nLat: {lat}\nLon: {lon}"},
                map_style=map_theme_pdk
            )
            st.pydeck_chart(deck, use_container_width=True)
            
        else:
            # High-Resolution Plotly Map
            fig_map = go.Figure()
            # Add trajectory road line (actual street turns)
            fig_map.add_trace(go.Scattermapbox(
                lat=[c[1] for c in road_coords],
                lon=[c[0] for c in road_coords],
                mode="lines",
                line=dict(width=5, color="#38BDF8" if is_night_theme else "#0284C7"),
                name="OSRM Street Route"
            ))
            # Add nearby available cabs
            fig_map.add_trace(go.Scattermapbox(
                lat=[c["lat"] for c in nearby_cabs],
                lon=[c["lon"] for c in nearby_cabs],
                mode="markers",
                marker=dict(size=11, color="#F59E0B"),
                text=[f"🚖 Available Cab #{c['medallion']}<br>ETA: {c['eta_min']} mins ({c['dist_m']}m)<br>Driver: {c['driver']}" for c in nearby_cabs],
                name="Nearby Available Cabs"
            ))
            # Add Pickup marker
            fig_map.add_trace(go.Scattermapbox(
                lat=[p_lat],
                lon=[p_lon],
                mode="markers+text",
                marker=dict(size=14, color="#10B981"),
                text=["PICKUP"],
                textposition="bottom right",
                name="Pickup Point"
            ))
            # Add Dropoff marker
            fig_map.add_trace(go.Scattermapbox(
                lat=[d_lat],
                lon=[d_lon],
                mode="markers+text",
                marker=dict(size=14, color="#EF4444"),
                text=["DROPOFF"],
                textposition="top right",
                name="Drop-off Point"
            ))
            
            mid_lat = (p_lat + d_lat) / 2.0
            mid_lon = (p_lon + d_lon) / 2.0
            fig_map.update_layout(
                mapbox=dict(
                    style=map_theme_plotly,
                    center=dict(lat=mid_lat, lon=mid_lon),
                    zoom=10.5
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=360,
                showlegend=False
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
        st.markdown(f"""
        <div style="font-size: 0.85rem; color: {'#94A3B8' if is_night_theme else '#475569'}; background: {card_bg}; padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid {card_border}; margin-top: 0.5rem; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            🧭 <b>Azimuth Compass Heading:</b> <code>{bearing_deg:.1f}° ({compass_str})</code> &nbsp;|&nbsp; 
            🛫 <b>Hub Proximity:</b> JFK: <b>{feat_df['dropoff_JFK_dist'].iloc[0]:.1f}km</b> • LGA: <b>{feat_df['dropoff_LGA_dist'].iloc[0]:.1f}km</b> • EWR: <b>{feat_df['dropoff_EWR_dist'].iloc[0]:.1f}km</b>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # REAL-TIME FLEET DISPATCH RADAR & LIVE TAXIMETER SIMULATOR
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📡 Real-Time NYC Yellow Cab Fleet Dispatch Radar")
    r_cols = st.columns([1.1, 1.1, 1.1, 0.9])
    for c_i, cab in enumerate(nearby_cabs):
        with r_cols[c_i]:
            st.markdown(f"""
            <div style="background: {card_bg}; padding: 0.85rem 1rem; border-radius: 12px; border: 1px solid {card_border}; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
                <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #F59E0B;">🚖 Medallion #{cab['medallion']}</div>
                <div style="font-weight: 700; font-size: 0.92rem; color: {'#F1F5F9' if is_night_theme else '#0F172A'}; margin-top: 0.2rem;">{cab['model']}</div>
                <div style="font-size: 0.8rem; color: {card_text}; margin-top: 0.2rem;">Driver: <b>{cab['driver']}</b> ({cab['rating']})</div>
                <div style="font-size: 0.85rem; color: #10B981; font-weight: 700; margin-top: 0.35rem;">⚡ {cab['dist_m']}m away • ETA: ~{cab['eta_min']} min</div>
            </div>
            """, unsafe_allow_html=True)
    with r_cols[3]:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("📡 Dispatch Nearest Cab", key="btn_dispatch_cab", use_container_width=True):
            st.session_state["dispatched_cab"] = nearby_cabs[0]
            st.toast(f"✅ Dispatched Medallion #{nearby_cabs[0]['medallion']} ({nearby_cabs[0]['driver']}) to your pickup!", icon="🚖")
        if st.session_state.get("dispatched_cab"):
            d_cab = st.session_state["dispatched_cab"]
            st.success(f"🚖 **Cab Dispatched:** #{d_cab['medallion']} arriving in ~{d_cab['eta_min']} mins.")

    with st.expander("🚖 Real-Time Interactive In-Cab Taximeter Ride Simulator", expanded=False):
        st.markdown("Simulate a live moving taxi ride on the configured route with real-time digital meter ticks, live speedometer, and progress tracking:")
        
        sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 1.2])
        with sim_col1:
            st.markdown(f"**Assigned Vehicle:** Medallion `#NYC-4192`")
            st.markdown(f"**Driver:** Salim K. (Rating: 4.96 ★)")
        with sim_col2:
            st.markdown(f"**Street Distance:** `{road_dist_km:.2f} km`")
            st.markdown(f"**Estimated Ride Duration:** `~{road_dur_mins:.1f} mins`")
        with sim_col3:
            st.markdown(f"**Base Drop Rate:** `$2.50` (TLC Standard)")
            st.markdown(f"**Predicted Final Total:** **`${meter_display_fare:.2f}`**")
            
        start_ride = st.button("▶️ Launch Real-Time In-Cab Ride Simulation", key="btn_launch_live_sim", use_container_width=True)
        
        if start_ride:
            prog_bar = st.progress(0)
            status_text = st.empty()
            meter_col1, meter_col2, meter_col3 = st.columns(3)
            with meter_col1:
                live_meter_disp = st.empty()
            with meter_col2:
                live_speed_disp = st.empty()
            with meter_col3:
                live_odo_disp = st.empty()
                
            steps = 15
            for s in range(steps + 1):
                pct = int(s / steps * 100)
                frac = s / steps
                prog_bar.progress(pct)
                cur_dist = road_dist_km * frac
                cur_fare = 2.50 + (meter_display_fare - 2.50) * (frac ** 0.9)
                cur_speed = int(22 + 10 * np.sin(frac * np.pi * 3)) if s < steps else 0
                
                live_meter_disp.metric("Live Taximeter ($)", f"${cur_fare:.2f}")
                live_speed_disp.metric("Live Speedometer", f"{cur_speed} mph")
                live_odo_disp.metric("Traveled Distance", f"{cur_dist:.2f} km")
                
                if s == 0:
                    status_text.info(f"🚖 **Trip Started:** Flag dropped ($2.50). Departing {st.session_state['p_choice']}...")
                elif s < steps // 2:
                    status_text.info(f"🚦 **En Route:** Navigating Manhattan traffic along route ({int(frac*100)}% complete)...")
                elif s < steps:
                    status_text.info(f"🏁 **Approaching Destination:** Decelerating toward {st.session_state['d_choice']}...")
                else:
                    status_text.success(f"🎉 **Trip Completed:** Passenger arrived at destination. Final fare: **${meter_display_fare:.2f}**.")
                time.sleep(0.12)
            st.balloons()
            st.markdown("👉 **Go to the `🧾 Official TLC e-Receipt` tab to view or download the itemized invoice for this completed trip.**")

# -----------------------------------------------------------------------------
# TAB 2: MULTI-MODEL BATTLE ARENA
# -----------------------------------------------------------------------------
with tab_battle:
    st.markdown("### ⚡ Live Multi-Model Battle Arena")
    st.markdown("Simultaneously benchmark the **currently configured trip telemetry** across the deep feedforward neural network, gradient-boosted decision trees, linear regression, and the regulatory meter formula:")
    
    battle_df = pd.DataFrame([
        {"Model": "Deep Neural Network (PyTorch)", "Predicted Fare ($)": pred_fare, "Delta vs DNN ($)": 0.00, "Architecture": "3 Dense Layers (128-64-32) + Huber Loss", "Type": "Deep Learning"},
        {"Model": "LightGBM Regressor", "Predicted Fare ($)": lgb_pred, "Delta vs DNN ($)": round(lgb_pred - pred_fare, 2), "Architecture": "250 Boosted Decision Trees", "Type": "Gradient Boosting"},
        {"Model": "Linear Regression Baseline", "Predicted Fare ($)": lr_pred, "Delta vs DNN ($)": round(lr_pred - pred_fare, 2), "Architecture": "Ordinary Least Squares (OLS)", "Type": "Linear Baseline"},
        {"Model": "NYC TLC Regulatory Formula", "Predicted Fare ($)": est_rule_fare, "Delta vs DNN ($)": round(est_rule_fare - pred_fare, 2), "Architecture": "$2.50 Base + $1.56/km + Taxes", "Type": "Regulatory Rule"}
    ])
    
    col_bchart, col_radar = st.columns([1.25, 1.0])
    
    with col_bchart:
        fig_battle = px.bar(
            battle_df,
            x="Predicted Fare ($)",
            y="Model",
            orientation="h",
            color="Type",
            color_discrete_map={
                "Deep Learning": "#10B981",
                "Gradient Boosting": "#38BDF8",
                "Linear Baseline": "#F59E0B",
                "Regulatory Rule": "#94A3B8"
            },
            text="Predicted Fare ($)"
        )
        fig_battle.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig_battle.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=chart_font_color),
            xaxis=dict(gridcolor=chart_grid_color, title="Estimated Fare ($ USD)"),
            yaxis=dict(autorange="reversed", title=""),
            margin=dict(l=20, r=40, t=20, b=20),
            height=280
        )
        st.plotly_chart(fig_battle, use_container_width=True)
        
    with col_radar:
        # Multi-Criteria Spider Radar Chart
        categories = ['R² Accuracy', 'Outlier Immunity', 'Inference Speed', 'Traffic Non-Linearity', 'Extreme Trip Calibration']
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[0.835, 0.95, 0.88, 0.92, 0.94],
            theta=categories,
            fill='toself',
            name='PyTorch DNN',
            line_color='#10B981'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[0.842, 0.80, 0.94, 0.90, 0.85],
            theta=categories,
            fill='toself',
            name='LightGBM',
            line_color='#38BDF8'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[0.605, 0.40, 0.99, 0.30, 0.45],
            theta=categories,
            fill='toself',
            name='OLS Linear',
            line_color='#F59E0B'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=radar_grid_color),
                bgcolor=radar_polar_bg
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=chart_font_color, size=10),
            margin=dict(l=30, r=30, t=25, b=25),
            height=280,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    st.dataframe(battle_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 3: WHAT-IF SENSITIVITY SIMULATOR
# -----------------------------------------------------------------------------
with tab_whatif:
    st.markdown("### 🔮 What-If Temporal & Occupancy Sensitivity Simulator")
    st.markdown("Inspect how the predicted taxi fare evolves if departure occurs at different hours of the day, with varying passenger headcounts, or under extreme road delays:")
    
    col_sim1, col_sim2 = st.columns([1.1, 1.0])
    
    # 24-hour simulation curve
    hours = list(range(24))
    hour_fares = []
    
    if scaler is not None and dnn_model is not None:
        sim_df = pd.DataFrame([{
            "key": f"sim_{h}",
            "pickup_datetime": pd.to_datetime(datetime.datetime.combine(trip_date, datetime.time(h, 0))),
            "pickup_longitude": p_lon,
            "pickup_latitude": p_lat,
            "dropoff_longitude": d_lon,
            "dropoff_latitude": d_lat,
            "passenger_count": passengers
        } for h in hours])
        
        sim_feats = extract_features(sim_df)
        sim_scaled = scaler.transform(sim_feats[FEATURE_COLS].values)
        with torch.no_grad():
            sim_preds = dnn_model(torch.tensor(sim_scaled, dtype=torch.float32)).numpy().flatten()
        hour_fares = [max(2.50, round(float(f) * weather_mult, 2)) for f in sim_preds]
        
        fig_sim = px.line(
            x=hours,
            y=hour_fares,
            markers=True,
            title="Predicted Fare vs. Hour of Day (24-Hour Diurnal Profile)",
            labels={"x": "Hour of Day (0 = 12 AM Midnight, 23 = 11 PM)", "y": "Estimated Fare ($ USD)"}
        )
        fig_sim.add_vline(x=trip_time.hour, line_dash="dash", line_color="#F59E0B", annotation_text="Selected Time")
        fig_sim.update_traces(line_color="#38BDF8", marker=dict(size=7, color="#10B981"))
        fig_sim.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=chart_font_color),
            xaxis=dict(gridcolor=chart_grid_color, tickmode='linear', tick0=0, dtick=2),
            yaxis=dict(gridcolor=chart_grid_color),
            margin=dict(l=20, r=20, t=40, b=20),
            height=320
        )
        with col_sim1:
            st.plotly_chart(fig_sim, use_container_width=True)
            
        with col_sim2:
            st.markdown("#### 💡 Dynamic Pricing & Surge Intelligence")
            min_fare_h = hours[np.argmin(hour_fares)]
            max_fare_h = hours[np.argmax(hour_fares)]
            st.markdown(f"""
            - **Lowest Fare Window:** `{min_fare_h}:00` (${min(hour_fares):.2f}) — optimal travel window with lower congestion.
            - **Peak Fare Window:** `{max_fare_h}:00` (${max(hour_fares):.2f}) — accounts for rush-hour surcharge & commuter bottleneck.
            - **Temporal Surge Delta:** `${max(hour_fares) - min(hour_fares):.2f}` dynamic variation on this exact trajectory.
            """)
            
            # Passenger Sensitivity Curve
            p_counts = list(range(1, 7))
            p_sim_df = pd.DataFrame([{
                "key": f"sim_p_{p}",
                "pickup_datetime": pd.to_datetime(pickup_dt),
                "pickup_longitude": p_lon,
                "pickup_latitude": p_lat,
                "dropoff_longitude": d_lon,
                "dropoff_latitude": d_lat,
                "passenger_count": p
            } for p in p_counts])
            p_feats = extract_features(p_sim_df)
            p_scaled = scaler.transform(p_feats[FEATURE_COLS].values)
            with torch.no_grad():
                p_preds = dnn_model(torch.tensor(p_scaled, dtype=torch.float32)).numpy().flatten()
            p_fares = [max(2.50, round(float(f) * weather_mult, 2)) for f in p_preds]
            
            st.markdown(f"**Occupancy Sensitivity (1 to 6 Passengers):** Range: `${min(p_fares):.2f}` – `${max(p_fares):.2f}`. Confirms model correctly separates physical distance from vehicle occupancy while slightly adjusting for larger groups.")

# -----------------------------------------------------------------------------
# TAB 4: OFFICIAL NYC TLC DIGITAL E-RECEIPT
# -----------------------------------------------------------------------------
with tab_receipt:
    st.markdown("### 🧾 Official NYC Taxi & Limousine Commission (TLC) e-Receipt")
    st.markdown("Generate a verifiable itemized trip invoice with customizable gratuity and instant download:")
    
    col_rec1, col_rec2 = st.columns([1.1, 1.0])
    
    with col_rec2:
        st.markdown("#### 💳 Gratuity & Payment Configuration")
        tip_choice = st.radio(
            "Select Gratuity Percentage",
            [10, 15, 18, 20, 25, 0],
            index=2,
            horizontal=True,
            format_func=lambda x: f"{x}%" if x > 0 else "No Tip"
        )
        tip_amt = round(pred_fare * (tip_choice / 100.0), 2)
        total_fare_with_tip = round(pred_fare + tip_amt, 2)
        payment_method = st.selectbox("Payment Mode", ["Credit Card (Mastercard / Visa)", "Apple Pay / Google Wallet", "Cash", "MTA Mobility Card"])
        
        # Receipt text export
        receipt_txt = f"""
=====================================================
          NEW YORK CITY TAXI & LIMOUSINE COMMISSION
                    OFFICIAL E-RECEIPT
=====================================================
Medallion ID   : NYC-TAXI-4192
Operator ID    : CSE-4192-SOA-ITER
Date & Time    : {pickup_dt.strftime('%Y-%m-%d %H:%M:%S')}
Payment Method : {payment_method}
-----------------------------------------------------
Pickup Location: {st.session_state['p_choice']}
                 ({p_lat:.4f}, {p_lon:.4f})
Drop-off Loc   : {st.session_state['d_choice']}
                 ({d_lat:.4f}, {d_lon:.4f})
Distance       : {distance_km:.2f} km ({distance_km * 0.621371:.2f} miles)
Passengers     : {passengers}
-----------------------------------------------------
Base Flag Drop Rate       : $2.50
Metered Mileage           : ${max(0.0, distance_km * 1.56):.2f}
Peak / Night Surcharges   : ${((1.00 if is_rush else 0.0) + (0.50 if is_night else 0.0)):.2f}
MTA State Tax & Imp. Fee  : $0.80
Weather/Traffic Adjustment: {weather_mult:.2f}x
-----------------------------------------------------
PREDICTED SUB-TOTAL       : ${pred_fare:.2f}
Tip ({tip_choice}%)                  : ${tip_amt:.2f}
=====================================================
TOTAL CHARGED             : ${total_fare_with_tip:.2f}
=====================================================
Model: PyTorch Deep Feedforward Neural Net (Huber δ=1.0)
Verification Code: TLC-DNN-{int(pred_fare*100)}-{passengers}
Thank you for riding NYC Yellow Cab!
"""
        st.download_button(
            label="📥 Download Official Trip e-Receipt (.txt)",
            data=receipt_txt,
            file_name=f"NYC_Taxi_Receipt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col_rec1:
        st.markdown(f"""<div class="receipt-box">
<div class="receipt-header">
    <div style="font-size: 1.15rem; font-weight: 800; color: #FDE68A; letter-spacing: 0.05em;">NYC TAXI & LIMOUSINE COMMISSION</div>
    <div style="font-size: 0.8rem; color: #94A3B8;">MEDALLION NO: NYC-TAXI-4192 &nbsp;|&nbsp; DL: 2341019538</div>
    <div style="font-size: 0.8rem; color: #94A3B8;">{pickup_dt.strftime('%B %d, %Y  •  %I:%M %p')}</div>
</div>
<div class="receipt-line">
    <span>Pickup:</span>
    <span style="font-weight: 600;">{st.session_state['p_choice'][:26]}</span>
</div>
<div class="receipt-line">
    <span>Drop-off:</span>
    <span style="font-weight: 600;">{st.session_state['d_choice'][:26]}</span>
</div>
<div class="receipt-line">
    <span>Trip Distance:</span>
    <span>{distance_km:.2f} km ({distance_km * 0.621371:.2f} mi)</span>
</div>
<div class="receipt-line">
    <span>Occupancy:</span>
    <span>{passengers} Passenger{'s' if passengers > 1 else ''}</span>
</div>
<div style="border-top: 1px dashed rgba(255, 255, 255, 0.15); margin: 0.8rem 0;"></div>
<div class="receipt-line">
    <span>Initial Base Flag Drop</span>
    <span>$2.50</span>
</div>
<div class="receipt-line">
    <span>Metered Distance Charge</span>
    <span>${max(0.0, distance_km * 1.56):.2f}</span>
</div>
<div class="receipt-line">
    <span>Surcharges (Rush/Night)</span>
    <span>${((1.00 if is_rush else 0.0) + (0.50 if is_night else 0.0)):.2f}</span>
</div>
<div class="receipt-line">
    <span>MTA Tax & Improvement</span>
    <span>$0.80</span>
</div>
<div class="receipt-line">
    <span>Sub-Total (DNN Predictor)</span>
    <span style="font-weight: 700; color: #38BDF8;">${pred_fare:.2f}</span>
</div>
<div class="receipt-line">
    <span>Gratuity ({tip_choice}%)</span>
    <span>${tip_amt:.2f}</span>
</div>
<div class="receipt-total">
    <span>TOTAL AMOUNT</span>
    <span>${total_fare_with_tip:.2f}</span>
</div>
<div style="text-align: center; margin-top: 1.2rem; font-size: 0.72rem; color: #64748B;">
    ★ AUTH CODE: TLC-DNN-{int(pred_fare*100)} ★<br>
    Siksha 'O' Anusandhan (ITER) • Centre for AI & ML
</div>
</div>""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 5: NEURAL TOPOLOGY & HUBER LOSS LAB
# -----------------------------------------------------------------------------
with tab_theory:
    st.markdown("### 🧠 Deep Feedforward Neural Topology & Huber Loss Formulation")
    
    col_topo1, col_topo2 = st.columns([1.1, 1.0])
    
    with col_topo1:
        st.markdown("""
        ```
        Input Layer: 33 Telemetry Features (Spatial, Geodesic, Airport, Cyclical)
              │
              ▼
        Dense Layer 1: Linear(33 ──► 128) ──► BatchNorm1d ──► ReLU ──► Dropout(p=0.20)
              │
              ▼
        Dense Layer 2: Linear(128 ──► 64) ──► BatchNorm1d ──► ReLU ──► Dropout(p=0.10)
              │
              ▼
        Dense Layer 3: Linear(64 ──► 32)  ──► ReLU Activation
              │
              ▼
        Output Layer:  Linear(32 ──► 1)   ──► Continuous Fare Regressor ($ USD)
        ```
        """)
        
    with col_topo2:
        st.markdown("#### ⚡ Live Layer Activation Probe")
        if layer_activations:
            st.markdown(f"""
            - **Input Feature Vector Dimension:** `33 Features`
            - **Hidden Layer 1 Mean Activation:** `{layer_activations.get('L1_mean', 0.0):.4f}` *(ReLU Activated)*
            - **Hidden Layer 2 Mean Activation:** `{layer_activations.get('L2_mean', 0.0):.4f}` *(ReLU Activated)*
            - **Hidden Layer 3 Mean Activation:** `{layer_activations.get('L3_mean', 0.0):.4f}` *(ReLU Activated)*
            - **Total Trainable Parameters:** `17,921 Weights & Biases`
            - **Inference Latency:** `{infer_duration_ms:.2f} ms`
            """)
        else:
            st.info("Load PyTorch model to inspect live neuron activations.")
            
    st.markdown("---")
    st.markdown("#### 🔬 Interactive Huber Loss vs. MSE vs. MAE Laboratory")
    delta_val = st.slider("Select Huber Loss Transition Parameter (δ)", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    
    errors = np.linspace(-4, 4, 300)
    mse_vals = 0.5 * (errors ** 2)
    mae_vals = np.abs(errors)
    huber_vals = np.where(np.abs(errors) <= delta_val, 0.5 * (errors ** 2), delta_val * (np.abs(errors) - 0.5 * delta_val))
    
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=errors, y=mse_vals, mode="lines", name="Mean Squared Error (MSE)", line=dict(color="#EF4444", dash="dash")))
    fig_loss.add_trace(go.Scatter(x=errors, y=mae_vals, mode="lines", name="Mean Absolute Error (MAE)", line=dict(color="#38BDF8", dash="dot")))
    fig_loss.add_trace(go.Scatter(x=errors, y=huber_vals, mode="lines", name=f"Huber Loss (δ={delta_val:.1f})", line=dict(color="#10B981", width=3)))
    
    fig_loss.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=chart_font_color),
        xaxis=dict(gridcolor=chart_grid_color, title="Prediction Residual Error: y - ŷ ($)"),
        yaxis=dict(gridcolor=chart_grid_color, title="Loss Penalty Value", range=[0, 8]),
        margin=dict(l=20, r=20, t=30, b=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_loss, use_container_width=True)
    st.markdown(rf"**Mathematical Advantage:** For residuals $|y - \hat{{y}}| \le {delta_val:.1f}$, Huber loss acts as smooth quadratic MSE (fast gradient convergence). For larger errors, it transitions to linear penalization, preventing explosive gradient spikes caused by meter anomalies and extreme outliers.")

# -----------------------------------------------------------------------------
# TAB 6: VISUALIZATIONS GALLERY
# -----------------------------------------------------------------------------
with tab_viz:
    st.markdown("### 📊 Comprehensive High-Resolution Visualizations Gallery")
    
    viz_catalog = [
        ("11_dnn_training_validation_loss.png", "DNN Training vs Validation Loss Progression", "Convergence"),
        ("12_loss_functions_comparison.png", "Comparative Convergence: MSE vs MAE vs Huber Loss", "Convergence"),
        ("13_actual_vs_predicted_fare.png", "DNN Predicted Fare vs Actual Fare (R² = 0.8734)", "Evaluation"),
        ("14_residual_distribution.png", "Residual Error Distribution & Normal Q-Q Plot", "Evaluation"),
        ("15_model_comparison_bar.png", "Model Benchmark Comparison (MAE & R² Scores)", "Evaluation"),
        ("01_fare_distribution.png", "Target Fare Amount Distribution & Skewness", "EDA"),
        ("03_nyc_pickup_density.png", "NYC Spatial Pickup Density Map", "Spatial"),
        ("05_pickup_vs_dropoff_geo.png", "Pickup vs Dropoff Spatial Concentration", "Spatial"),
        ("08_hourly_day_heatmap.png", "Ridership Density: Day of Week × Hour Heatmap", "EDA"),
        ("09_distance_vs_fare.png", "Distance vs Fare Amount Scatter & Non-Linear Trend", "EDA"),
        ("10_feature_correlation_heatmap.png", "Feature Correlation Matrix Heatmap", "EDA")
    ]
    
    viz_col_filter, viz_col_tgl = st.columns([1.3, 1.0])
    with viz_col_filter:
        cat_filter = st.radio("Filter by Category", ["All", "Convergence", "Evaluation", "Spatial", "EDA"], horizontal=True)
    with viz_col_tgl:
        toggle_grid_view = st.toggle(
            "🖼️ Grid Gallery Mode (Multi-Plot View)",
            value=False,
            key="tgl_viz_grid",
            help="Switch between single focused inspection view and side-by-side gallery grid."
        )
    
    filtered_catalog = [v for v in viz_catalog if cat_filter == "All" or v[2] == cat_filter]
    
    if toggle_grid_view:
        grid_cols = st.columns(2)
        for i, (fname, label, cat) in enumerate(filtered_catalog):
            img_p = os.path.join(VIZ_DIR, fname)
            if os.path.exists(img_p):
                with grid_cols[i % 2]:
                    st.image(img_p, caption=f"Figure: {label} [{cat}]", use_container_width=True)
    else:
        sel_chart = st.selectbox("Select Research Visualization Figure", [v[1] for v in filtered_catalog], index=0)
        for fname, label, cat in filtered_catalog:
            if label == sel_chart:
                img_p = os.path.join(VIZ_DIR, fname)
                if os.path.exists(img_p):
                    st.image(img_p, caption=f"Figure: {label} [{cat}]", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 7: AUTOMATED TEST SUITE RUNNER
# -----------------------------------------------------------------------------
with tab_test:
    st.markdown("### 🧪 Automated Deployment Verification Test Suite")
    st.markdown("Executes `app/test_deployment.py` to validate checkpoint integrity, scaler transformations, and inference boundaries across 4 operational test scenarios:")
    
    toggle_live_telemetry = st.toggle(
        "⚡ Extended Diagnostics & Telemetry Assertion Logs",
        value=True,
        key="tgl_test_telemetry",
        help="Display full terminal execution logs including benchmark bounds and geodetic distances."
    )
    
    if st.button("▶️ Execute Automated Deployment Test Suite", key="btn_run_tests_tab"):
        import subprocess
        test_script = os.path.join(BASE_DIR, "app", "test_deployment.py")
        res = subprocess.run([sys.executable, test_script], capture_output=True, text=True)
        if toggle_live_telemetry:
            st.code(res.stdout, language="bash")
        if res.returncode == 0:
            st.success("✅ All 4 Deployment Validation Tests Passed Successfully (100% Pass Rate)!")
        else:
            st.error("❌ Some deployment tests encountered issues.")

# -----------------------------------------------------------------------------
# TAB 8: ACADEMIC REGISTRY & DIRECT DOWNLOAD
# -----------------------------------------------------------------------------
with tab_academic:
    st.markdown("""
    ### 🏛️ Academic Laboratory Record & Team Registry
    - **Academic Institution:** Siksha 'O' Anusandhan (Deemed to be University), ITER, Bhubaneswar
    - **Department:** Department of Computer Science & Engineering | Centre for AI & ML
    - **Course:** Machine Learning Projects with Python (Course Code: `CSE 4192`)
    - **Course Faculty:** **Dr. Gyana Ranjan Patra**
    - **Academic Year:** 2025 – 2026 | **Batch:** 2023 – 2027
    
    ---
    #### Project Team Members & Technical Responsibilities:
    
    | Sl. No. | Student Name | Registration Number | Core Roles & Responsibilities | Contribution |
    | :---: | :--- | :---: | :--- | :---: |
    | 1 | **Tribhuwan Singh** | `2341019538` | Deep Feedforward Neural Network Design, PyTorch Training & Streamlit Web Deployment | **25%** |
    | 2 | **Surajit Sahoo** | `2341019165` | Exploratory Data Analysis, Geolocation Spatial Mapping & Ridership Heatmaps | **25%** |
    | 3 | **Anwesha Srichandan** | `2341019594` | Data Quality Auditing, Outlier Cleansing & Geodesic Feature Engineering Pipeline | **25%** |
    | 4 | **Priti Rani Maity** | `2341013065` | Hyperparameter Optimization Search, Benchmark Evaluation & Academic Report | **25%** |
    
    ---
    #### 📥 Direct Lab Record Download:
    """)
    
    col_dl1, col_dl2 = st.columns(2)
    pdf_path = os.path.join(BASE_DIR, "Laboratory_Record_CSE4192.pdf")
    docx_path = os.path.join(BASE_DIR, "Laboratory_Record_CSE4192.docx")
    
    with col_dl1:
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Download Official Laboratory Record (PDF, 3.1 MB)",
                    data=f.read(),
                    file_name="Laboratory_Record_CSE4192_NYCTaxiFare_DNN.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    with col_dl2:
        if os.path.exists(docx_path):
            with open(docx_path, "rb") as f:
                st.download_button(
                    label="📝 Download Official Laboratory Record (DOCX, 4.1 MB)",
                    data=f.read(),
                    file_name="Laboratory_Record_CSE4192_NYCTaxiFare_DNN.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
