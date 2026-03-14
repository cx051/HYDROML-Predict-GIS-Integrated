"""
HYDROML PREDICT - Main Application Logic
"""

import os

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import tempfile
import streamlit.components.v1 as components

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

from recommender import generate_recommendations, BIS_LIMITS, TRAINING_STANDARDS
from ui import (
    inject_css, render_header, render_input_section,
    render_results_header, render_status_cards, render_consumption_suitability,
    render_shap_analysis, render_recommendations, render_pdf_button,
    render_bis_compliance_table
)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="HydroML Predict - Water Quality Analysis",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# INJECT CSS
# ===============================
inject_css()

# ===============================
# RENDER HEADER
# ===============================
render_header()

# ===============================
# SIMPLE NAVIGATION (TAB-LIKE)
# ===============================
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Home"

nav_col1, nav_col2, nav_col3 = st.columns(3)
with nav_col1:
    if st.button("Home", use_container_width=True):
        st.session_state["active_page"] = "Home"
with nav_col2:
    if st.button("Analyser", use_container_width=True):
        st.session_state["active_page"] = "Analyser"
with nav_col3:
    if st.button("GIS Mapping", use_container_width=True):
        st.session_state["active_page"] = "GIS Mapping"

active_page = st.session_state["active_page"]

if active_page == "Home":
    home_html = """
    <div style="padding: 2.5rem 2rem 2rem; background: white; border-radius: 24px;
                border: 1px solid #e2e8f0; box-shadow: 0 20px 25px -5px rgba(15,23,42,0.06);
                display: flex; flex-direction: column; gap: 1.75rem;">
        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div style="font-size: 0.8rem; font-weight: 800; letter-spacing: 0.2em;
                        text-transform: uppercase; color: #38bdf8;">
                Welcome to
            </div>
            <div style="font-size: 2.1rem; font-weight: 800; color: #0f172a;
                        letter-spacing: -0.04em;">
                HydroML&nbsp;Predict – Intelligent Water Quality Decision Support
            </div>
            <p style="margin: 0; color: #64748b; font-size: 0.98rem; max-width: 46rem;">
                HydroML Predict blends <strong>machine learning</strong>, <strong>BIS IS&nbsp;10500:2012 standards</strong> 
                and an intuitive interface to help you quickly understand whether a water sample is 
                safe to drink, how its quality evolves, and what treatments are required.
            </p>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 1.5rem; margin-top: 0.5rem;">
            <div style="padding: 1.4rem 1.3rem; border-radius: 18px; background: #eff6ff;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #1d4ed8;
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Potability Insight
                </div>
                <div style="margin-top: 0.6rem; font-size: 1.05rem; font-weight: 700; color: #0f172a;">
                    AI-powered safe / not-safe prediction
                </div>
                <p style="margin-top: 0.4rem; font-size: 0.9rem; color: #475569;">
                    Upload key physico‑chemical parameters and receive a model-backed
                    decision with confidence and WQI.
                </p>
            </div>

            <div style="padding: 1.4rem 1.3rem; border-radius: 18px; background: #ecfeff;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #0e7490;
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Standards Aware
                </div>
                <div style="margin-top: 0.6rem; font-size: 1.05rem; font-weight: 700; color: #0f172a;">
                    BIS tiered compliance check
                </div>
                <p style="margin-top: 0.4rem; font-size: 0.9rem; color: #475569;">
                    Every parameter is tagged as <strong>acceptable</strong>, <strong>permissible</strong>, 
                    or <strong>violation</strong> based on BIS guidelines, making reports easy to interpret.
                </p>
            </div>

            <div style="padding: 1.4rem 1.3rem; border-radius: 18px; background: #fef9c3;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #854d0e;
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Actionable Guidance
                </div>
                <div style="margin-top: 0.6rem; font-size: 1.05rem; font-weight: 700; color: #0f172a;">
                    Treatment recommendations &amp; PDF report
                </div>
                <p style="margin-top: 0.4rem; font-size: 0.9rem; color: #475569;">
                    Get targeted treatment suggestions for critical contaminants and
                    export a professional PDF summary for documentation or reporting.
                </p>
            </div>
        </div>

        <div style="margin-top: 0.75rem; padding: 1.1rem 1.2rem; border-radius: 16px;
                    background: #0f172a; display: flex; flex-wrap: wrap; gap: 0.75rem;
                    align-items: center; justify-content: space-between;">
            <div style="color: #e5e7eb; font-size: 0.9rem; max-width: 32rem;">
                Use the <strong>Analyser</strong> tab to run water quality predictions for a specific sample,
                or switch to <strong>GIS Mapping</strong> to explore spatial patterns for different years.
            </div>
            <div style="display: flex; gap: 0.6rem; flex-wrap: wrap;">
                <span style="background: rgba(59,130,246,0.1); color: #bfdbfe; padding: 0.35rem 0.9rem;
                             border-radius: 999px; font-size: 0.78rem; font-weight: 600;">
                    💧 AI‑driven risk screening
                </span>
                <span style="background: rgba(52,211,153,0.1); color: #bbf7d0; padding: 0.35rem 0.9rem;
                             border-radius: 999px; font-size: 0.78rem; font-weight: 600;">
                    📊 Standards‑aligned WQI
                </span>
            </div>
        </div>
    </div>
    """

    components.html(home_html, height=520, scrolling=True)

# ===============================
# LOAD MODEL & EXPLAINER (Cached for performance)
# ===============================
@st.cache_resource
def load_model_and_explainer():
    model = xgb.XGBClassifier()
    model.load_model("model/clean_xgboost_model.json")
    explainer = shap.TreeExplainer(model)
    return model, explainer

try:
    model, explainer = load_model_and_explainer()
    feature_names = model.get_booster().feature_names
except Exception as e:
    st.error(f"Failed to load model or explainer: {e}")
    st.stop()

# ===============================
# BIS COMPLIANCE CHECK
# ===============================
def bis_check(data):
    results = {}
    for param, limits in BIS_LIMITS.items():
        value = data[param]
        if param == "pH":
            min_acc, max_acc = limits["acceptable"]
            if min_acc <= value <= max_acc:
                results[param] = "acceptable"
            else:
                results[param] = "violation"
        else:
            if value <= limits["acceptable"]:
                results[param] = "acceptable"
            elif value <= limits["permissible"]:
                results[param] = "permissible"
            else:
                results[param] = "violation"
    return results

# ===============================
# WQI CALCULATION (Training Aligned)
# ===============================
def calculate_weights(standards, params):
    sum_reciprocals = sum(1 / standards[p] for p in params)
    k = 1 / sum_reciprocals
    return {p: k / standards[p] for p in params}


def calculate_wqi(data):
    standards = TRAINING_STANDARDS
    params = list(standards.keys())
    weights = calculate_weights(standards, params)

    wqi_sum = 0
    weight_sum = 0

    for p in params:
        observed = data[p]
        w = weights[p]

        if p == 'pH':
            if observed < 7:
                qi = 100 * (7 - observed) / (7 - 6.5)
            else:
                qi = 100 * (observed - 7) / (8.5 - 7)
        else:
            qi = 100 * (observed / standards[p])

        wqi_sum += qi * w
        weight_sum += w

    return round(wqi_sum / weight_sum, 2)

if active_page == "Analyser":
    sample_data = render_input_section()
    X = pd.DataFrame([sample_data])

    if set(feature_names) != set(X.columns):
        st.error("Feature mismatch between training and app input. Please ensure all parameters are provided.")
        st.stop()

    X = X[feature_names]

    st.markdown('<div style="margin-top: 2rem; margin-bottom: 3rem;">', unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        analyze_clicked = st.button("Analyse Water Quality", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)

    if 'analyzed' not in st.session_state:
        st.session_state['analyzed'] = False

    if analyze_clicked:
        st.session_state['analyzed'] = True
        components.html(
            """
            <script>
                window.parent.document.getElementById('results-anchor').scrollIntoView({behavior: 'smooth'});
            </script>
            """,
            height=0,
        )

        try:
            prediction = model.predict(X)[0]
            prob = model.predict_proba(X)[0]

            positive_class_index = list(model.classes_).index(1)
            predicted_class_index = list(model.classes_).index(prediction)
            confidence = prob[predicted_class_index] * 100

            wqi = calculate_wqi(sample_data)
            bis_results = bis_check(sample_data)
            violation_count = sum(1 for status in bis_results.values() if status == "violation")

            recommendations = generate_recommendations(sample_data, bis_results)

            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[positive_class_index]

            shap_array = np.array(shap_values).flatten()
            shap_dict = {col: float(shap_array[i]) for i, col in enumerate(feature_names)}

            st.session_state['prediction'] = prediction
            st.session_state['confidence'] = confidence
            st.session_state['wqi'] = wqi
            st.session_state['sample_data'] = sample_data
            st.session_state['bis_results'] = bis_results
            st.session_state['violation_count'] = violation_count
            st.session_state['recommendations'] = recommendations
            st.session_state['shap_dict'] = shap_dict

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.session_state['analyzed'] = False
            st.stop()

    if st.session_state['analyzed']:
        prediction = st.session_state.get('prediction')
        confidence = st.session_state.get('confidence')
        wqi = st.session_state.get('wqi')
        sample_data = st.session_state.get('sample_data')
        bis_results = st.session_state.get('bis_results', {})
        violation_count = st.session_state.get('violation_count', 0)
        recommendations = st.session_state.get('recommendations', [])
        shap_dict = st.session_state.get('shap_dict', {})

        render_results_header()
        render_status_cards(prediction, confidence, wqi, violation_count)

        render_bis_compliance_table(bis_results, sample_data)

        st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)

        col_output1, col_output2 = st.columns([1, 1.5])

        with col_output1:
            render_consumption_suitability(prediction, confidence)

        with col_output2:
            render_shap_analysis(shap_dict, feature_names)

        render_recommendations(recommendations)

        if render_pdf_button():
            try:
                pdf_prediction = st.session_state.get('prediction')
                pdf_confidence = st.session_state.get('confidence')
                pdf_wqi = st.session_state.get('wqi')
                pdf_sample_data = st.session_state.get('sample_data')
                pdf_bis_results = st.session_state.get('bis_results', {})
                pdf_recs = st.session_state.get('recommendations', [])

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
                elements = []
                styles = getSampleStyleSheet()

                elements.append(Paragraph("HydroML Predict - Water Quality Analysis Report", styles["Title"]))
                elements.append(Spacer(1, 0.1 * inch))
                elements.append(Paragraph("WQI calculated using weighted arithmetic index method (Qi-based) aligned with BIS training standards.", styles["Normal"]))
                elements.append(Spacer(1, 0.1 * inch))
                elements.append(Paragraph("Compliance Tiers: Acceptable (within ideal limits), Permissible (acceptable in absence of alternate source), Violation (exceeds maximum limits).", styles["Normal"]))
                elements.append(Spacer(1, 0.2 * inch))

                pdf_is_safe = (pdf_prediction == 1) and (st.session_state.get('violation_count', 0) == 0)
                elements.append(Paragraph(f"Potability Decision: {'SAFE' if pdf_is_safe else 'NOT SAFE'}", styles["Heading2"]))
                elements.append(Paragraph(f"Confidence: {pdf_confidence:.2f}%", styles["Normal"]))
                elements.append(Paragraph(f"Water Quality Index (WQI): {pdf_wqi}", styles["Normal"]))
                elements.append(Spacer(1, 0.2 * inch))

                elements.append(Paragraph("BIS Compliance Status:", styles["Heading3"]))
                for param, status in pdf_bis_results.items():
                    val = pdf_sample_data.get(param)
                    status_icon = "✅" if status == "acceptable" else "⚠️" if status == "permissible" else "❌"
                    elements.append(Paragraph(f"{status_icon} {param}: {val:.2f} ({status.capitalize()})", styles["Normal"]))
                elements.append(Spacer(1, 0.2 * inch))

                elements.append(Paragraph("Treatment Recommendations:", styles["Heading3"]))
                if not pdf_recs:
                    elements.append(Paragraph("No treatments required. All parameters are within safe limits.", styles["Normal"]))
                for rec in pdf_recs:
                    elements.append(Paragraph(f"<b>{rec['contaminant']} ({rec['status']}):</b> {rec['message']}", styles["Normal"]))
                    elements.append(Paragraph(f"Suggested Treatment: {', '.join(rec['treatment'])}", styles["Normal"]))
                    elements.append(Spacer(1, 0.1 * inch))

                doc.build(elements)

                with open(temp_file.name, "rb") as f:
                    st.download_button(
                        "⬇️ Download Detailed PDF Report",
                        f,
                        file_name="HydroML_Report.pdf",
                        type="primary",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

if active_page == "GIS Mapping":
    st.subheader("GIS Mapping – Spatial View of Water Quality")

    st.markdown(
        "Select a **year** to explore the corresponding interactive web map. "
        "This page expects folders like `webmap_2024` and `webmap_2025` inside the project."
    )

    year = st.selectbox("Select Year", ["2024", "2025"], index=0, key="gis_year")

    map_file = f"webmap_{year}/index.html"

    with open(map_file, "r", encoding="utf-8") as f:
        map_html = f.read()

    components.html(map_html, height=720, scrolling=True)

    st.caption(f"Map source: {map_file}")

# Footer
st.markdown(f"""
    <div style="text-align: center; padding: 6rem 2rem 3rem; color: #94a3b8; font-size: 0.875rem;">
        <div style="margin-bottom: 1rem; color: #0f172a; font-weight: 800; font-size: 1.1rem; letter-spacing: -0.025em;">HYDROML PREDICT</div>
        <p style="margin-bottom: 0.3rem;">Advanced AI-Driven Water Quality Analysis System</p>
        <p style="margin-bottom: 0.2rem; color: #64748b;">Developed by <strong>Niyatha Jyothish</strong> &amp; <strong>Devananth S L</strong></p>
        <p style="margin-bottom: 0.2rem; color: #64748b;">Data Support: <strong>Vaisakh S S</strong> &amp; <strong>Dhanush R L</strong></p>
        <p style="margin-bottom: 0.3rem; color: #64748b;">Support and Guidance: <strong>Anu Elizabeth Panicker</strong></p>
        <p>© 2026 | All Rights Reserved</p>
    </div>
""", unsafe_allow_html=True)
