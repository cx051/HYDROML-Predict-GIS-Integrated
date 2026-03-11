"""
UI Components for HYDROML PREDICT
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import base64
import os
import textwrap
import re

def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file):
        return ""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def inject_css():
    """Inject custom CSS styling"""
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        /* Global Styles */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #f1f5f9;
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
            max-width: 1200px;
        }

        /* Card / Container Styling */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: white;
            border-radius: 16px;
            border: 1px solid #e2e8f0 !important;
            padding: 2rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            transition: transform 0.2s ease;
        }
        
        /* Input Styling */
        div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input {
            background-color: white !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
            text-align: left !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            color: #0f172a !important;
            height: 48px !important;
            transition: all 0.2s;
        }
        
        div[data-testid="stTextInput"] input:focus, div[data-testid="stNumberInput"] input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
        }

        /* Primary Button (Analyze) */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            color: white !important;
            border: none !important;
            height: 58px !important;
            border-radius: 12px !important;
            width: 100% !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        }
        
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3) !important;
        }

        /* Labels and Text */
        .card-label {
            font-size: 0.7rem;
            font-weight: 800;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        h2, h3 {
            color: #0f172a !important;
            letter-spacing: -0.025em !important;
        }

        /* SHAP Analysis Chart Styling */
        .shap-row {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
            gap: 1.25rem;
        }
        
        .shap-label {
            width: 80px;
            font-size: 0.85rem;
            color: #334155;
            font-weight: 700;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Status Badge */
        .status-badge {
            background: #dcfce7;
            color: #15803d;
            font-size: 0.85rem;
            font-weight: 800;
            padding: 0.5rem 1.25rem;
            border-radius: 9999px;
            display: inline-block;
            letter-spacing: 0.025em;
        }
        
        .status-badge.unsafe {
            background: #fee2e2;
            color: #b91c1c;
        }

        /* Hide default headers/footers */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
        }
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }

        /* Responsive Header Classes */
        .header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 3.5rem;
            background: white;
            padding: 1.5rem 3rem;
            border-radius: 24px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 2.5rem;
        }

        .header-logo-container {
            background: white;
            border-radius: 16px;
            flex-shrink: 0;
        }

        .header-logo-container img {
            display: block;
            object-fit: contain;
            width: 240px;
            max-width: 100%;
        }

        .header-separator {
            height: 60px;
            width: 1.5px;
            background: #e2e8f0;
        }

        .header-title-container {
            text-align: left;
        }

        .header-title {
            margin: 0;
            color: #0f172a !important;
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.1;
            white-space: nowrap;
        }

        .header-subtitle {
            margin: 0.4rem 0 0;
            color: #64748b;
            font-size: 1rem;
            font-weight: 600;
        }

        .header-badges {
            display: flex;
            gap: 0.75rem;
        }

        .badge-ai {
            background: #f1f5f9;
            color: #475569;
            border: 1px solid #e2e8f0;
            padding: 0.6rem 1.2rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            white-space: nowrap;
        }

        .badge-bis {
            background: #dcfce7;
            color: #15803d;
            border: 1px solid #bbf7d0;
            padding: 0.6rem 1.2rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            white-space: nowrap;
        }

        /* Mobile Adjustments */
        @media (max-width: 768px) {
            .header-container {
                flex-direction: column;
                padding: 1.5rem;
                gap: 1.5rem;
                text-align: center;
                margin-bottom: 2rem;
            }

            .header-left {
                flex-direction: column;
                gap: 1.25rem;
                width: 100%;
            }

            .header-logo-container img {
                width: 180px;
                margin: 0 auto;
            }

            .header-separator {
                display: none;
            }

            .header-title-container {
                text-align: center;
            }

            .header-title {
                font-size: 1.75rem;
            }

            .header-subtitle {
                font-size: 0.875rem;
            }

            .header-badges {
                justify-content: center;
                width: 100%;
                flex-wrap: wrap;
            }
        }
        </style>
    """, unsafe_allow_html=True)

def get_icon(name):
    """Return SVG for common icons used in the UI"""
    icons = {
        'water': '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/></svg>',
        'anions': '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v8L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45L14 10V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/></svg>',
        'cations': '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>',
        'parameters': '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/></svg>',
        'shield': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
        'chart': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
        'check': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
        'cross': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
    }
    return icons.get(name, '')

def render_html(html):
    """Safely render HTML in Streamlit by removing newlines and extra whitespace"""
    clean_html = re.sub(r'\s+', ' ', html).strip()
    st.markdown(clean_html, unsafe_allow_html=True)

def render_results_header():
    """Render the Analysis Results section header"""
    st.markdown('<div id="results-section"></div>', unsafe_allow_html=True)
    html = textwrap.dedent("""
        <div style="text-align: center; margin: 4rem 0 3rem 0;">
            <h1 style="color: #0f172a; font-weight: 800; font-size: 2.5rem; letter-spacing: -0.05em; margin-bottom: 0.5rem;">Analysis Results</h1>
            <div style="height: 4px; width: 60px; background: #2563eb; margin: 0 auto; border-radius: 2px;"></div>
        </div>
    """)
    render_html(html)

def render_header():
    """Render the application header with logo"""
    logo_base64 = get_base64_of_bin_file("assets/hydroml_logo.png")
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="HydroML Logo">' if logo_base64 else ""
    
    html = textwrap.dedent(f"""
        <div class="header-container">
            <div class="header-left">
                <div class="header-logo-container">
                    {logo_html}
                </div>
                <div class="header-separator"></div>
                <div class="header-title-container">
                    <h1 class="header-title">HydroML Predict</h1>
                    <p class="header-subtitle">Hybrid AI + BIS Standards Integrated Analysis</p>
                </div>
            </div>
            <div class="header-badges">
                <span class="badge-ai">AI Powered</span>
                <span class="badge-bis">BIS Integrated</span>
            </div>
        </div>
    """)
    render_html(html)

def perfect_number_input(label, value, min_val, max_val, step, key):
    """Simple styled number input (Manual Entry only)"""
    if key not in st.session_state:
        st.session_state[key] = value
        
    st.markdown(f'<p style="font-size: 0.875rem; font-weight: 600; color: #334155; margin-bottom: -1rem;">{label}</p>', unsafe_allow_html=True)
    
    new_val = st.number_input(label, min_value=float(min_val), max_value=float(max_val), 
                             value=float(st.session_state[key]), step=float(step), 
                             key=f"input_{key}", label_visibility="hidden")
    st.session_state[key] = round(new_val, 2)
                
    st.markdown('<div style="margin-bottom: 0.5rem;"></div>', unsafe_allow_html=True)
    return st.session_state[key]

def render_input_section():
    """Render the input parameter cards section"""
    render_html(textwrap.dedent(f"""
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; margin-top: 1rem;">
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 0.75rem; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);">
                {get_icon('parameters')}
            </div>
            <h2 style="margin: 0; font-size: 1.6rem; font-weight: 800; color: #0f172a; letter-spacing: -0.025em;">Water Quality Parameters</h2>
        </div>
    """))
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        with st.container(border=True):
            render_html(textwrap.dedent(f"""
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 2rem;">
                    <div style="background: #e0f2fe; padding: 0.5rem; border-radius: 8px; color: #0369a1;">
                        {get_icon('water')}
                    </div>
                    <span style="color: #0369a1; font-weight: 800; font-size: 1.1rem;">Basic Analysis</span>
                </div>
            """))
            pH = perfect_number_input("pH Level", 7.00, 0.0, 14.0, 0.1, "pH")
            EC = perfect_number_input("EC (Conductivity)", 1000.00, 0.0, 5000.0, 10.0, "EC")
            CO3 = perfect_number_input("CO₃ (Carbonate)", 10.00, 0.0, 500.0, 1.0, "CO3")
            HCO3 = perfect_number_input("HCO₃ (Bicarbonate)", 150.00, 0.0, 1000.0, 5.0, "HCO3")
            Cl = perfect_number_input("Chloride (Cl)", 200.00, 0.0, 1000.0, 5.0, "Cl")
    
    with col2:
        with st.container(border=True):
            render_html(textwrap.dedent(f"""
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 2rem;">
                    <div style="background: #f0fdf4; padding: 0.5rem; border-radius: 8px; color: #15803d;">
                        {get_icon('anions')}
                    </div>
                    <span style="color: #15803d; font-weight: 800; font-size: 1.1rem;">Hardness & Mineral Contents</span>
                </div>
            """))
            SO4 = perfect_number_input("Sulphate (SO₄)", 150.00, 0.0, 1000.0, 5.0, "SO4")
            NO3 = perfect_number_input("Nitrate (NO₃)", 30.00, 0.0, 500.0, 1.0, "NO3")
            TH = perfect_number_input("Total Hardness (TH)", 200.00, 0.0, 1000.0, 5.0, "TH")
            Ca = perfect_number_input("Calcium (Ca)", 50.00, 0.0, 500.0, 1.0, "Ca")
            Mg = perfect_number_input("Magnesium (Mg)", 30.00, 0.0, 500.0, 1.0, "Mg")
    
    with col3:
        with st.container(border=True):
            render_html(textwrap.dedent(f"""
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 2rem;">
                    <div style="background: #fff7ed; padding: 0.5rem; border-radius: 8px; color: #c2410c;">
                        {get_icon('cations')}
                    </div>
                    <span style="color: #c2410c; font-weight: 800; font-size: 1.1rem;">Dissolved Minerals & TDS</span>
                </div>
            """))
            Na = perfect_number_input("Sodium (Na)", 40.00, 0.0, 1000.0, 1.0, "Na")
            K = perfect_number_input("Potassium (K)", 5.00, 0.0, 500.0, 0.5, "K")
            F = perfect_number_input("Fluoride (F)", 1.00, 0.0, 10.0, 0.1, "F")
            TDS = perfect_number_input("TDS (Total Solids)", 400.00, 0.0, 5000.0, 10.0, "TDS")
    
    return {
        'pH': pH, 'EC': EC, 'CO3': CO3, 'HCO3': HCO3, 'Cl': Cl,
        'SO4': SO4, 'NO3': NO3, 'TH': TH, 'Ca': Ca, 'Mg': Mg,
        'Na': Na, 'K': K, 'F': F, 'TDS': TDS
    }

def render_svg_gauge(percentage, status_text):
    """Render a circular gauge using SVG for perfect alignment and gradients"""
    # Calculate circle parameters
    radius = 90
    circumference = 2 * np.pi * radius
    dash_offset = circumference * (1 - percentage / 100)
    
    color_id = "blueGradient" if status_text == "SAFE" else "redGradient"
    badge_bg = "#dcfce7" if status_text == "SAFE" else "#fee2e2"
    badge_text = "#15803d" if status_text == "SAFE" else "#b91c1c"
    
    svg = f"""
    <div style="display: flex; justify-content: center; align-items: center; padding: 1rem 0; position: relative; width: 240px; margin: 0 auto;">
        <svg width="240" height="240" viewBox="0 0 240 240" style="transform: rotate(-90deg);">
            <defs>
                <linearGradient id="blueGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color: #2563eb; stop-opacity: 1" />
                    <stop offset="100%" style="stop-color: #60a5fa; stop-opacity: 1" />
                </linearGradient>
                <linearGradient id="redGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color: #ef4444; stop-opacity: 1" />
                    <stop offset="100%" style="stop-color: #f87171; stop-opacity: 1" />
                </linearGradient>
            </defs>
            <!-- Background Circle -->
            <circle cx="120" cy="120" r="{radius}" stroke="#f1f5f9" stroke-width="16" fill="none" />
            <!-- Progress Circle -->
            <circle cx="120" cy="120" r="{radius}" stroke="url(#{color_id})" stroke-width="16" fill="none" 
                    stroke-dasharray="{circumference}" stroke-dashoffset="{dash_offset}" 
                    stroke-linecap="round" style="transition: stroke-dashoffset 0.5s ease-out;" />
        </svg>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; width: 100%;">
            <div style="font-size: 3.8rem; font-weight: 800; color: #0f172a; line-height: 1; letter-spacing: -0.02em;">{percentage:.0f}%</div>
            <div style="display: inline-block; background: {badge_bg}; color: {badge_text}; font-size: 0.75rem; font-weight: 800; padding: 0.4rem 1.2rem; border-radius: 9999px; margin-top: 0.6rem; letter-spacing: 0.05em; text-transform: uppercase;">{status_text}</div>
        </div>
    </div>
    """
    render_html(svg)

def render_consumption_suitability(prediction, confidence):
    """Render Consumption Suitability card with circular gauge"""
    with st.container(border=True):
        st.markdown('<div class="card-label" style="margin-bottom: 0.5rem;">Consumption Suitability</div>', unsafe_allow_html=True)
        
        suitability_pct = confidence if prediction == 1 else 100 - confidence
        status_text = "SAFE" if prediction == 1 else "UNSAFE"
        
        # Render SVG gauge
        render_svg_gauge(suitability_pct, status_text)
        
        st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
        
        # Stats boxes
        render_html(textwrap.dedent(f"""
            <div style="display: flex; gap: 1rem;">
                <div style="flex: 1; background: white; padding: 1.25rem; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <p style="color: #64748b; font-size: 0.7rem; margin-bottom: 0.5rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em;">Confidence</p>
                    <p style="color: #2563eb; font-size: 1.4rem; font-weight: 800; margin: 0;">{confidence:.1f}%</p>
                </div>
                <div style="flex: 1; background: white; padding: 1.25rem; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <p style="color: #64748b; font-size: 0.7rem; margin-bottom: 0.5rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em;">Risk Factor</p>
                    <p style="color: {'#10b981' if prediction == 1 else '#ef4444'}; font-size: 1.4rem; font-weight: 800; margin: 0;">{('Low' if confidence > 80 else 'Medium') if prediction == 1 else 'High'}</p>
                </div>
            </div>
        """))

def render_shap_analysis(shap_dict, feature_names):
    """Render SHAP Feature Impact visualization with gradient bars and expand/collapse option"""
    if 'shap_expanded' not in st.session_state:
        st.session_state.shap_expanded = False

    # Mapping for nicer display names
    name_map = {
        'pH': 'pH Level', 'EC': 'Conductivity', 'CO3': 'Carbonates',
        'HCO3': 'Bicarbonates', 'Cl': 'Chloride', 'SO4': 'Sulphates',
        'NO3': 'Nitrate', 'TH': 'Hardness', 'Ca': 'Calcium',
        'Mg': 'Magnesium', 'Na': 'Sodium', 'K': 'Potassium',
        'F': 'Fluoride', 'TDS': 'TDS'
    }

    header_html = f"""
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 2.5rem; padding-bottom: 1.5rem; border-bottom: 1px dashed #e2e8f0;">
            <div>
                <h3 style="color: #0f172a; font-weight: 800; font-size: 1.35rem; margin: 0; letter-spacing: -0.02em;">Parameter Influence Map</h3>
                <p style="color: #64748b; font-size: 0.875rem; margin-top: 0.25rem; font-weight: 500;">SHAP Analysis: Relative impact of each parameter on the result</p>
            </div>
            <div style="display: flex; gap: 1.25rem; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.5rem;">
                <span style="display: flex; align-items: center; gap: 0.5rem; color: #10b981;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; background: linear-gradient(to right, #10b981, #34d399);"></span> Safe Influence
                </span>
                <span style="display: flex; align-items: center; gap: 0.5rem; color: #ef4444;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; background: linear-gradient(to left, #ef4444, #f87171);"></span> Risk Influence
                </span>
            </div>
        </div>
    """
    
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Expansion logic
    display_features = sorted_shap if st.session_state.shap_expanded else sorted_shap[:6]
    
    # Determine scaling. We want a central axis.
    max_abs = max(abs(f[1]) for f in sorted_shap) if sorted_shap else 1
    if max_abs == 0: max_abs = 1

    with st.container(border=True):
        render_html(header_html)
        
        for feat, val in display_features:
            display_name = name_map.get(feat, feat)
            # Use 40% as max width to leave space for labels
            width_pct = (abs(val) / max_abs) * 40 
            
            # Gradient colors
            if val > 0:
                bar_bg = "linear-gradient(to right, #10b981, #34d399)"
                color = "#10b981"
            else:
                bar_bg = "linear-gradient(to left, #ef4444, #f87171)"
                color = "#ef4444"
            
            # Label on the side of the bar
            val_text = f"{'+' if val > 0 else ''}{val:.2f}"
            val_pos_style = f"left: calc(50% + {width_pct}% + 10px);" if val > 0 else f"right: calc(50% + {width_pct}% + 10px);"
            val_align = "left" if val > 0 else "right"

            if val >= 0:
                bar_content = f'<div style="position: absolute; left: 50%; width: {width_pct}%; background: {bar_bg}; height: 100%; border-radius: 0 4px 4px 0; box-shadow: 2px 0 4px {color}30;"></div>'
            else:
                bar_content = f'<div style="position: absolute; right: 50%; width: {width_pct}%; background: {bar_bg}; height: 100%; border-radius: 4px 0 0 4px; box-shadow: -2px 0 4px {color}30;"></div>'

            row_html = f"""
                <div style="display: flex; align-items: center; gap: 0; margin-bottom: 1.25rem; position: relative;">
                    <div style="width: 120px; text-align: left; font-size: 0.8rem; font-weight: 800; color: #334155; padding-right: 1rem;">{display_name}</div>
                    <div style="flex: 1; position: relative; height: 18px; background: #f8fafc; border-radius: 6px; display: flex; align-items: center; border: 1px solid #f1f5f9;">
                        <div style="position: absolute; left: 50%; height: 100%; width: 2px; background: #94a3b8; z-index: 5;"></div>
                        {bar_content}
                        <div style="position: absolute; {val_pos_style} font-size: 0.7rem; font-weight: 800; color: {color}; text-align: {val_align}; white-space: nowrap;">
                            {val_text}
                        </div>
                    </div>
                </div>
            """
            render_html(row_html)
        
        # Expand/Collapse Button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            btn_label = "Collapse Parameters" if st.session_state.shap_expanded else "Expand All Parameters"
            if st.button(btn_label, key="shap_toggle_btn", use_container_width=True):
                st.session_state.shap_expanded = not st.session_state.shap_expanded
                st.rerun()
        
        render_html('<div style="margin-top: 1rem; text-align: center; font-size: 0.65rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;">Centered axis represents neutral impact</div>')

def render_status_cards(prediction, confidence, wqi, violation_count=0):
    """Render the three status cards matching the design"""
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        with st.container(border=True):
            # Final Decision Logic
            is_safe = (prediction == 1) and (violation_count == 0)
            
            color = "#10b981" if is_safe else "#ef4444"
            icon = get_icon('check') if is_safe else get_icon('cross')
            
            status_text = "Safe to Drink" if is_safe else "Not Safe"
            if violation_count > 0 and prediction == 1:
                status_text = "BIS Violation"
            elif violation_count == 0 and prediction == 0:
                status_text = "ML Warning"

            render_html(textwrap.dedent(f"""
                <p style="color: #64748b; font-size: 0.7rem; margin-bottom: 1.5rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 800; text-align: center;">Potability Status</p>
                <div style="width: 56px; height: 56px; border-radius: 50%; background: {color}; color: white; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.5rem; box-shadow: 0 4px 10px {color}40;">
                    {icon}
                </div>
                <p style="color: {color}; font-size: 1.3rem; font-weight: 800; text-align: center; margin: 0;">{ status_text }</p>
            """))
    
    with col2:
        with st.container(border=True):
            render_html(textwrap.dedent(f"""
                <p style="color: #64748b; font-size: 0.7rem; margin-bottom: 1.5rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 800; text-align: center;">Confidence Level</p>
                <div style="width: 56px; height: 56px; border-radius: 50%; background: #3b82f6; color: white; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.5rem; box-shadow: 0 4px 10px #3b82f640;">
                    {get_icon('shield')}
                </div>
                <p style="color: #3b82f6; font-size: 1.3rem; font-weight: 800; text-align: center; margin: 0;">{confidence:.1f}%</p>
            """))
    
    with col3:
        with st.container(border=True):
            render_html(textwrap.dedent(f"""
                <p style="color: #64748b; font-size: 0.7rem; margin-bottom: 1.5rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 800; text-align: center;">WQI Score</p>
                <div style="width: 56px; height: 56px; border-radius: 50%; background: #f59e0b; color: white; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.5rem; box-shadow: 0 4px 10px #f59e0b40;">
                    {get_icon('chart')}
                </div>
                <p style="color: #f59e0b; font-size: 1.3rem; font-weight: 800; text-align: center; margin: 0;">{wqi:.0f}</p>
            """))

def render_bis_compliance_table(bis_results, sample_data):
    """Render a section for BIS Compliance Check with a status table"""
    with st.container(border=True):
        st.markdown("""
            <h3 style="color: #0f172a; font-weight: 800; font-size: 1.25rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.5rem;">
                🏛 BIS Compliance Table
            </h3>
        """, unsafe_allow_html=True)
        
        # Table Header
        render_html("""
            <div style="display: flex; background: #f8fafc; padding: 1rem; border-radius: 8px; font-weight: 800; color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                <div style="flex: 2;">Parameter</div>
                <div style="flex: 1;">Value</div>
                <div style="flex: 2;">Status</div>
            </div>
        """)
        
        for param, status in bis_results.items():
            value = sample_data.get(param)
            
            if status == "acceptable":
                status_html = '<span style="color: #10b981; font-weight: 700;">✅ Acceptable</span>'
                bg_color = "transparent"
            elif status == "permissible":
                status_html = '<span style="color: #f59e0b; font-weight: 700;">⚠️ Permissible</span>'
                bg_color = "#fffbeb"
            else:
                status_html = '<span style="color: #ef4444; font-weight: 700;">❌ Violation</span>'
                bg_color = "#fef2f2"
                
            render_html(f"""
                <div style="display: flex; padding: 0.85rem 1rem; border-bottom: 1px solid #f1f5f9; background: {bg_color}; border-radius: 6px; align-items: center; margin-bottom: 2px;">
                    <div style="flex: 2; font-weight: 700; color: #334155;">{param}</div>
                    <div style="flex: 1; font-weight: 600; color: #475569;">{value:.2f}</div>
                    <div style="flex: 2;">{status_html}</div>
                </div>
            """)

def render_recommendations(recommendations):
    """Render treatment recommendations with modern professional icons"""
    st.markdown('<h2 style="color: #0f172a; font-weight: 800; font-size: 1.5rem; margin-top: 4rem; margin-bottom: 2rem;">Treatment Recommendations</h2>', unsafe_allow_html=True)
    
    if not recommendations:
        with st.container(border=True):
            render_html(textwrap.dedent("""
                <div style="display: flex; align-items: center; justify-content: center; gap: 1rem; color: #15803d; font-weight: 700; padding: 1rem;">
                    <span style="font-size: 1.5rem;">✅</span> 
                    <span>All parameters are within safe limits. No treatment required.</span>
                </div>
            """))
        return

    # Display in a grid if many, or list if few
    cols = st.columns(2) if len(recommendations) > 1 else [st.container()]
    
    for i, rec in enumerate(recommendations):
        with (cols[i % 2] if len(recommendations) > 1 else cols[0]):
            with st.container(border=True):
                contaminant = rec['contaminant']
                value = rec['value']
                status = rec['status']
                message = rec['message']
                treatment = rec['treatment']
                
                # Icon selection
                if status == 'Violation':
                    icon_bg, icon_color, icon = "#fee2e2", "#ef4444", "❌"
                    rec_title = f"CRITICAL: {contaminant} Violation"
                else:
                    icon_bg, icon_color, icon = "#fffbeb", "#f59e0b", "⚠️"
                    rec_title = f"ADVISORY: {contaminant} Level"
                    
                render_html(textwrap.dedent(f"""
                    <div style="display: flex; gap: 1.25rem; padding: 0.5rem 0;">
                        <div style="width: 56px; height: 56px; border-radius: 14px; background: {icon_bg}; color: {icon_color}; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; flex-shrink: 0; border: 1px solid {icon_color}15; box-shadow: 0 4px 6px -1px {icon_color}10;">
                            {icon}
                        </div>
                        <div>
                            <div style="font-weight: 800; color: #0f172a; font-size: 1.15rem; margin-bottom: 0.35rem; letter-spacing: -0.01em;">{rec_title}</div>
                            <p style="font-size: 0.925rem; color: #475569; line-height: 1.55; margin: 0; margin-bottom: 0.5rem;">
                                {message}
                            </p>
                            <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border-left: 3px solid {icon_color};">
                                <p style="font-size: 0.8rem; font-weight: 800; color: #64748b; text-transform: uppercase; margin-bottom: 0.25rem;">Suggested Treatment</p>
                                <p style="font-size: 0.875rem; color: #334155; font-weight: 600; margin: 0;">{', '.join(treatment)}</p>
                            </div>
                        </div>
                    </div>
                """))

def render_pdf_button():
    """Render PDF generation button matching the style"""
    st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)
    if st.button("Generate Detailed Analysis Report as PDF", use_container_width=True, type="primary"):
        return True
    return False

