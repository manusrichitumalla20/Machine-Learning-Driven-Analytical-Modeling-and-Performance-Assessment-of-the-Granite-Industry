"""
granite_streamlit_app.py
─────────────────────────
Streamlit app for Granite Industry — Regression & Classification
Loads pre-trained PKL models from the extracted granite_models/ folder.

Folder structure expected:
    granite_streamlit_app.py
    granite_models/
        metadata.pkl
        scaler.pkl
        le_state.pkl
        le_city.pkl
        oe_scale.pkl
        le_class.pkl
        best_reg_<ModelName>.pkl
        best_cls_<ModelName>.pkl

Run:
    pip install streamlit plotly scikit-learn joblib pandas numpy
    streamlit run granite_streamlit_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Granite Industry ML",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.page-title {
    font-size: 2rem; font-weight: 700;
    color: #1e293b; margin-bottom: 2px;
}
.page-sub { font-size: 0.95rem; color: #64748b; margin-bottom: 20px; }
.kpi-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 16px 20px; text-align: center;
}
.kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem; font-weight: 700; color: #0f172a;
}
.kpi-lbl { font-size: 0.73rem; color: #94a3b8;
           text-transform: uppercase; letter-spacing: 0.07em; margin-top: 3px; }
.pred-box {
    border-radius: 14px; padding: 22px 26px; margin: 8px 0;
    border: 1.5px solid #e2e8f0;
}
.pred-title { font-size: 0.75rem; color: #94a3b8;
              text-transform: uppercase; letter-spacing: 0.07em; }
.pred-val   { font-size: 2rem; font-weight: 700;
              font-family: 'JetBrains Mono', monospace; margin: 6px 0 2px; }
.pred-sub   { font-size: 0.85rem; color: #64748b; }
.badge {
    display: inline-block; padding: 3px 12px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
}
.badge-high   { background:#dcfce7; color:#16a34a; }
.badge-medium { background:#fef9c3; color:#ca8a04; }
.badge-low    { background:#fee2e2; color:#dc2626; }
.section-hd {
    font-size: 1.1rem; font-weight: 600; color: #1e293b;
    border-left: 3px solid #6366f1; padding-left: 10px;
    margin: 20px 0 12px;
}
.info-box {
    background: #f0f9ff; border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    font-size: 0.875rem; color: #0c4a6e;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────
MODELS_DIR = "granite_models"

@st.cache_resource(show_spinner="Loading models…")
def load_all():
    meta     = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))
    scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    le_state = joblib.load(os.path.join(MODELS_DIR, "le_state.pkl"))
    le_city  = joblib.load(os.path.join(MODELS_DIR, "le_city.pkl"))
    oe_scale = joblib.load(os.path.join(MODELS_DIR, "oe_scale.pkl"))
    le_class = joblib.load(os.path.join(MODELS_DIR, "le_class.pkl"))

    reg_file = f"best_reg_{meta['best_reg_model'].replace(' ', '_')}.pkl"
    cls_file = f"best_cls_{meta['best_cls_model'].replace(' ', '_')}.pkl"
    reg_mdl  = joblib.load(os.path.join(MODELS_DIR, reg_file))
    cls_mdl  = joblib.load(os.path.join(MODELS_DIR, cls_file))

    return meta, scaler, le_state, le_city, oe_scale, le_class, reg_mdl, cls_mdl

# ── Check models folder exists ──────────────────────────────────────────
if not os.path.isdir(MODELS_DIR):
    st.error(
        f"**`{MODELS_DIR}/` folder not found.**\n\n"
        f"Place the extracted `granite_models/` folder in the same directory as this script:\n\n"
        f"```\n"
        f"granite_streamlit_app.py\n"
        f"granite_models/\n"
        f"    metadata.pkl\n"
        f"    scaler.pkl\n"
        f"    best_reg_*.pkl\n"
        f"    best_cls_*.pkl\n"
        f"    ...\n"
        f"```"
    )
    st.stop()

meta, scaler, le_state, le_city, oe_scale, le_class, reg_mdl, cls_mdl = load_all()

FEATURES  = meta['features']
STATES    = sorted(meta['states'])
CITIES    = sorted(meta['cities'])
CLASSES   = meta['classes']

# city → state mapping for smart dropdown
CITY_STATE_MAP = {
    'Jaipur':'Rajasthan','Jodhpur':'Rajasthan','Udaipur':'Rajasthan',
    'Kishangarh':'Rajasthan','Makrana':'Rajasthan',
    'Chennai':'Tamil Nadu','Salem':'Tamil Nadu','Madurai':'Tamil Nadu',
    'Krishnagiri':'Tamil Nadu','Dharmapuri':'Tamil Nadu',
    'Kurnool':'Andhra Pradesh','Ongole':'Andhra Pradesh','Nellore':'Andhra Pradesh',
    'Kadapa':'Andhra Pradesh','Guntur':'Andhra Pradesh',
    'Hyderabad':'Telangana','Warangal':'Telangana','Karimnagar':'Telangana',
    'Khammam':'Telangana','Nizamabad':'Telangana',
    'Bengaluru':'Karnataka','Mysuru':'Karnataka','Hassan':'Karnataka',
    'Chitradurga':'Karnataka','Ramanagara':'Karnataka',
    'Surat':'Gujarat','Ahmedabad':'Gujarat','Vadodara':'Gujarat',
    'Bhavnagar':'Gujarat','Rajkot':'Gujarat',
}
STATE_CITIES = {}
for city, state in CITY_STATE_MAP.items():
    STATE_CITIES.setdefault(state, []).append(city)

# ── Helper ──────────────────────────────────────────────────────────────
def safe_enc(le, val):
    if val in le.classes_:
        return le.transform([val])[0]
    return 0

def build_row(state, city, scale, prod, mach, skilled, transport, rejection, price):
    row = {
        'State_enc':                  safe_enc(le_state, state),
        'City_enc':                   safe_enc(le_city, city),
        'Scale_enc':                  int(oe_scale.transform([[scale]])[0][0]),
        'Monthly_Production_Tons':    prod,
        'Machine_Utilization_Pct':    mach,
        'Skilled_Labor_Pct':          skilled,
        'Transport_Cost_Rs_per_Ton':  transport,
        'Quality_Rejection_Rate_Pct': rejection,
        'Avg_Selling_Price_Rs_sqm':   price,
    }
    row_df = pd.DataFrame([row])[FEATURES]
    return scaler.transform(row_df)

def badge_html(label):
    cls = {'High': 'badge-high', 'Medium': 'badge-medium', 'Low': 'badge-low'}.get(label, '')
    return f'<span class="badge {cls}">{label}</span>'

# ── Header ─────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🪨 Granite Industry ML Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="page-sub">Regression · Classification &nbsp;|&nbsp; '
    f'Models: <b>{meta["best_reg_model"]}</b> (reg) &nbsp;·&nbsp; '
    f'<b>{meta["best_cls_model"]}</b> (cls)</div>',
    unsafe_allow_html=True
)

# KPI strip
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (f'{meta["reg_r2"]:.4f}',    "Regression R²"),
    (f'{meta["reg_rmse"]:.3f}',  "Regression RMSE"),
    (f'{meta["cls_accuracy"]:.4f}', "Classification Accuracy"),
    (f'{meta["cls_f1_macro"]:.4f}', "Macro F1"),
    (str(len(FEATURES)),          "Features Used"),
]
for col, (val, lbl) in zip([k1,k2,k3,k4,k5], kpis):
    col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
                 f'<div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮  Predict",
    "📊  Model Info",
    "📦  Loaded Files",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-hd">Enter Granite Unit Details</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        state = st.selectbox("State", STATES, index=STATES.index('Tamil Nadu') if 'Tamil Nadu' in STATES else 0)

        city_options = STATE_CITIES.get(state, CITIES)
        city  = st.selectbox("City", city_options)

        scale = st.selectbox("Scale of Operation", ["Small", "Medium", "Large"], index=1)

        prod  = st.slider(
            "Monthly Production (Tons)", 200.0, 8000.0, 2700.0, 50.0,
            help="Total granite output per month in tons")

        mach  = st.slider(
            "Machine Utilization (%)", 35.0, 98.0, 76.0, 0.5,
            help="Percentage of machine capacity being used")

    with col_r:
        skilled = st.slider(
            "Skilled Labor (%)", 20.0, 92.0, 60.0, 0.5,
            help="Percentage of workforce that is skilled")

        transport = st.slider(
            "Transport Cost (₹/Ton)", 400.0, 5000.0, 1800.0, 50.0,
            help="Logistics cost per ton of granite")

        rejection = st.slider(
            "Quality Rejection Rate (%)", 0.5, 18.0, 5.5, 0.1,
            help="Percentage of output rejected for quality issues")

        price = st.slider(
            "Avg Selling Price (₹/sq.m)", 700.0, 3800.0, 2100.0, 50.0,
            help="Average revenue per square metre")

    predict_btn = st.button("🔮  Run Prediction", type="primary", use_container_width=True)

    if predict_btn:
        row_sc = build_row(state, city, scale, prod, mach, skilled, transport, rejection, price)

        # Regression
        reg_val  = reg_mdl.predict(row_sc)[0]
        pct      = (reg_val - 1.5) / (28.0 - 1.5) * 100

        # Classification
        cls_enc  = cls_mdl.predict(row_sc)[0]
        cls_prob = cls_mdl.predict_proba(row_sc)[0]
        cls_lbl  = le_class.inverse_transform([cls_enc])[0]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-hd">Prediction Results</div>', unsafe_allow_html=True)

        rc1, rc2 = st.columns(2, gap="large")

        with rc1:
            pct_color = "#16a34a" if reg_val >= 16 else "#ca8a04" if reg_val >= 8 else "#dc2626"
            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-title">Regression — Net Profit Margin</div>
              <div class="pred-val" style="color:{pct_color}">{reg_val:.2f}%</div>
              <div class="pred-sub">
                Approx. <b>{pct:.0f}th percentile</b> in dataset &nbsp;|&nbsp;
                Range: 1.5% → 28.0% &nbsp;|&nbsp; Median: ~13.2%
              </div>
            </div>""", unsafe_allow_html=True)

            # Gauge chart
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=reg_val,
                title={"text": "Net Profit Margin %", "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0, 28], "tickwidth": 1},
                    "bar":  {"color": pct_color},
                    "steps": [
                        {"range": [0,  8],  "color": "#fee2e2"},
                        {"range": [8,  16], "color": "#fef9c3"},
                        {"range": [16, 28], "color": "#dcfce7"},
                    ],
                    "threshold": {"line": {"color": "#1e293b", "width": 3}, "value": reg_val},
                },
                number={"suffix": "%", "font": {"size": 32}},
            ))
            fig_g.update_layout(height=240, margin=dict(l=20,r=20,t=40,b=10),
                                paper_bgcolor='rgba(0,0,0,0)', font_family="Inter")
            st.plotly_chart(fig_g, use_container_width=True)

        with rc2:
            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-title">Classification — Profitability Class</div>
              <div class="pred-val">{badge_html(cls_lbl)}</div>
              <div class="pred-sub" style="margin-top:8px">
                Low &lt; 8% &nbsp;·&nbsp; Medium 8–16% &nbsp;·&nbsp; High &gt; 16%
              </div>
            </div>""", unsafe_allow_html=True)

            # Probability bar chart
            prob_df = pd.DataFrame({
                "Class": CLASSES,
                "Probability": cls_prob * 100
            })
            clr_map = {"Low": "#ef4444", "Medium": "#eab308", "High": "#22c55e"}
            fig_p = px.bar(
                prob_df, x="Class", y="Probability",
                color="Class",
                color_discrete_map=clr_map,
                text="Probability",
                template="plotly_white",
                title="Class Probabilities (%)",
            )
            fig_p.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_p.update_layout(
                height=240, margin=dict(l=0,r=0,t=40,b=0),
                showlegend=False, yaxis=dict(range=[0,115]),
                font_family="Inter",
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_p, use_container_width=True)

        # Input summary table
        st.markdown('<div class="section-hd">Input Summary</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Feature": [
                "State", "City", "Scale of Operation",
                "Monthly Production (Tons)", "Machine Utilization (%)",
                "Skilled Labor (%)", "Transport Cost (₹/Ton)",
                "Quality Rejection Rate (%)", "Avg Selling Price (₹/sq.m)"
            ],
            "Value": [
                state, city, scale,
                f"{prod:.0f}", f"{mach:.1f}",
                f"{skilled:.1f}", f"{transport:.0f}",
                f"{rejection:.1f}", f"{price:.0f}"
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True, height=360)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="section-hd">Regression Model</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
        <b>Model:</b> {meta['best_reg_model']}<br>
        <b>Target:</b> Net_Profit_Margin_Pct (continuous)<br>
        <b>R² Score:</b> {meta['reg_r2']}<br>
        <b>RMSE:</b> {meta['reg_rmse']}
        </div>
        """, unsafe_allow_html=True)

        # Feature importance if available
        if hasattr(reg_mdl, 'feature_importances_'):
            imp = pd.Series(reg_mdl.feature_importances_, index=FEATURES).sort_values()
            fig_fi = px.bar(imp.reset_index(), x=imp.values, y=imp.index,
                            orientation='h', template='plotly_white',
                            color=imp.values, color_continuous_scale='Blues',
                            title="Feature Importances — Regression")
            fig_fi.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=0),
                                 coloraxis_showscale=False, yaxis_title="",
                                 paper_bgcolor='rgba(0,0,0,0)', font_family="Inter")
            st.plotly_chart(fig_fi, use_container_width=True)
        elif hasattr(reg_mdl, 'coef_'):
            coef = pd.Series(np.abs(reg_mdl.coef_), index=FEATURES).sort_values()
            fig_cf = px.bar(coef.reset_index(), x=coef.values, y=coef.index,
                            orientation='h', template='plotly_white',
                            color=coef.values, color_continuous_scale='Purples',
                            title="|Coefficients| — Regression")
            fig_cf.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=0),
                                 coloraxis_showscale=False, yaxis_title="",
                                 paper_bgcolor='rgba(0,0,0,0)', font_family="Inter")
            st.plotly_chart(fig_cf, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    with col_b:
        st.markdown('<div class="section-hd">Classification Model</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
        <b>Model:</b> {meta['best_cls_model']}<br>
        <b>Target:</b> Profitability_Class (Low / Medium / High)<br>
        <b>Accuracy:</b> {meta['cls_accuracy']}<br>
        <b>Macro F1:</b> {meta['cls_f1_macro']}<br>
        <b>Classes:</b> {' · '.join(meta['classes'])}
        </div>
        """, unsafe_allow_html=True)

        if hasattr(cls_mdl, 'feature_importances_'):
            imp_c = pd.Series(cls_mdl.feature_importances_, index=FEATURES).sort_values()
            fig_fi2 = px.bar(imp_c.reset_index(), x=imp_c.values, y=imp_c.index,
                             orientation='h', template='plotly_white',
                             color=imp_c.values, color_continuous_scale='Greens',
                             title="Feature Importances — Classification")
            fig_fi2.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=0),
                                  coloraxis_showscale=False, yaxis_title="",
                                  paper_bgcolor='rgba(0,0,0,0)', font_family="Inter")
            st.plotly_chart(fig_fi2, use_container_width=True)
        elif hasattr(cls_mdl, 'coef_'):
            coef_c = pd.Series(np.abs(cls_mdl.coef_).mean(axis=0), index=FEATURES).sort_values()
            fig_cf2 = px.bar(coef_c.reset_index(), x=coef_c.values, y=coef_c.index,
                             orientation='h', template='plotly_white',
                             color=coef_c.values, color_continuous_scale='Oranges',
                             title="|Coefficients| — Classification")
            fig_cf2.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=0),
                                  coloraxis_showscale=False, yaxis_title="",
                                  paper_bgcolor='rgba(0,0,0,0)', font_family="Inter")
            st.plotly_chart(fig_cf2, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    # Features list
    st.markdown('<div class="section-hd">Features Used for Prediction</div>', unsafe_allow_html=True)
    feat_df = pd.DataFrame({
        "Feature":     FEATURES,
        "Description": [
            "State (label encoded)",
            "City (label encoded)",
            "Scale of Operation (ordinal: 0=Small, 1=Medium, 2=Large)",
            "Monthly granite output in tons",
            "% of machine capacity utilised",
            "% of workforce that is skilled",
            "Logistics cost per ton (₹)",
            "% of output rejected for quality",
            "Average revenue per square metre (₹)",
        ]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — LOADED FILES
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hd">PKL Files in granite_models/</div>', unsafe_allow_html=True)

    rows = []
    for f in sorted(os.listdir(MODELS_DIR)):
        fpath = os.path.join(MODELS_DIR, f)
        sz    = os.path.getsize(fpath) / 1024
        tag   = ""
        if "best_reg" in f:   tag = "✅ Active regression model"
        elif "best_cls" in f: tag = "✅ Active classification model"
        elif "scaler"   in f: tag = "StandardScaler"
        elif "metadata" in f: tag = "Metadata (features, scores, class names)"
        elif "le_state" in f: tag = "State LabelEncoder"
        elif "le_city"  in f: tag = "City LabelEncoder"
        elif "le_class" in f: tag = "Profitability Class LabelEncoder"
        elif "oe_scale" in f: tag = "Scale OrdinalEncoder"
        elif "reg_"     in f: tag = "Regression model (not active)"
        elif "cls_"     in f: tag = "Classification model (not active)"
        rows.append({"File": f, "Size (KB)": f"{sz:.1f}", "Role": tag})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-hd">Encoder Reference</div>', unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)
    with ec1:
        st.write("**States available:**")
        st.write(", ".join(meta['states']))
    with ec2:
        st.write("**Cities available:**")
        st.write(", ".join(meta['cities']))

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🪨 Granite ML App")
    st.markdown("---")
    st.markdown("**Regression**")
    st.markdown(f"Model: `{meta['best_reg_model']}`")
    st.markdown(f"R²: `{meta['reg_r2']}`")
    st.markdown(f"RMSE: `{meta['reg_rmse']}`")
    st.markdown("---")
    st.markdown("**Classification**")
    st.markdown(f"Model: `{meta['best_cls_model']}`")
    st.markdown(f"Accuracy: `{meta['cls_accuracy']}`")
    st.markdown(f"Macro F1: `{meta['cls_f1_macro']}`")
    st.markdown("---")
    st.markdown("**Classes**")
    for c in meta['classes']:
        badge = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(c, "⚪")
        st.markdown(f"{badge} {c}")
    st.markdown("---")
    st.markdown("**How to run**")
    st.code("streamlit run granite_streamlit_app.py", language="bash")