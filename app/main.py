"""
INVEREX MVP — Streamlit Demo App

Patient selector -> Molecular summary -> Drug rankings -> Trial mapping -> Rationale
"""
import base64
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA_CACHE = ROOT / "data" / "cache"
APP_DIR = Path(__file__).resolve().parent

def _find_asset(name: str) -> Path:
    """Look for asset in app/ dir first (Streamlit Cloud), then parent dir (local)."""
    for base in [APP_DIR, ROOT.parent]:
        p = base / name
        if p.exists():
            return p
    return APP_DIR / name

LOGO_SVG_PATH = _find_asset("Symbol.svg")
BRANDING_IMG_PATH = _find_asset("Branding Visual 6.png")

# -- Color palette ---------------------------------------------------------
COLORS = {
    "bg": "#0E0B1A",
    "surface": "rgba(45, 38, 89, 0.4)",
    "surface_solid": "#1A1533",
    "primary": "#2D2659",
    "primary_light": "#575ECF",
    "accent": "#8B7FD4",
    "text": "#E8E4F0",
    "muted": "#9B95AD",
    "success": "#00CC96",
    "warning": "#EF553B",
    "border": "rgba(139, 127, 212, 0.25)",
    "glow": "rgba(87, 94, 207, 0.3)",
}

# Chart color scale: purple -> blue gradient
CHART_COLORS = [
    "#2D2659", "#3A3180", "#4B3DA6", "#575ECF",
    "#6B73DB", "#8B7FD4", "#A5B4FC", "#C4B5FD",
]
CHART_SEQUENTIAL = [
    [0.0, "#2D2659"], [0.25, "#575ECF"], [0.5, "#6B73DB"],
    [0.75, "#00CC96"], [1.0, "#00FFAA"],
]


# -- Helpers: encode assets as base64 for embedding -----------------------
def _svg_to_b64(path: Path, recolor: str | None = None) -> str:
    if path.exists():
        content = path.read_text()
        if recolor:
            # Replace the fill color so the logo is visible on dark backgrounds
            content = content.replace("fill: #2d2659;", f"fill: {recolor};")
            content = content.replace("fill:#2d2659", f"fill:{recolor}")
        return base64.b64encode(content.encode()).decode()
    return ""


def _img_to_b64(path: Path) -> str:
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return ""


# -- Page config -----------------------------------------------------------
st.set_page_config(
    page_title="INVEREX — Breast Cancer Drug Ranking",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Inject custom CSS for glassmorphism + dark theme ----------------------
def inject_custom_css():
    logo_b64 = _svg_to_b64(LOGO_SVG_PATH)
    brand_b64 = _img_to_b64(BRANDING_IMG_PATH)

    st.markdown(f"""
    <style>
    /* ====================================================================
       GLOBAL OVERRIDES
       ==================================================================== */

    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {{
        font-family: 'Inter', sans-serif !important;
    }}

    /* Root background */
    .stApp {{
        background: {COLORS['bg']};
        color: {COLORS['text']};
    }}

    /* Main content area */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    /* ====================================================================
       SIDEBAR
       ==================================================================== */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(
            180deg,
            rgba(14, 11, 26, 0.97) 0%,
            rgba(26, 21, 51, 0.97) 100%
        );
        border-right: 1px solid {COLORS['border']};
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1rem;
    }}

    /* ====================================================================
       GLASSMORPHISM CARD MIXIN (applied to key containers)
       ==================================================================== */

    /* Metric cards */
    [data-testid="stMetric"] {{
        background: {COLORS['surface']};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 40px {COLORS['glow']},
                    inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLORS['muted']} !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    [data-testid="stMetricValue"] {{
        color: {COLORS['text']} !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }}

    /* ====================================================================
       EXPANDER (Trial matching)
       ==================================================================== */
    .streamlit-expanderHeader {{
        background: {COLORS['surface']} !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {COLORS['border']} !important;
        border-radius: 12px !important;
        color: {COLORS['text']} !important;
        font-weight: 500 !important;
    }}

    details {{
        background: {COLORS['surface']} !important;
        backdrop-filter: blur(12px);
        border: 1px solid {COLORS['border']} !important;
        border-radius: 12px !important;
    }}

    /* ====================================================================
       DATAFRAME
       ==================================================================== */
    [data-testid="stDataFrame"] {{
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        overflow: hidden;
    }}

    /* ====================================================================
       SELECTBOX & INPUTS
       ==================================================================== */
    .stSelectbox > div > div {{
        background: {COLORS['surface']} !important;
        backdrop-filter: blur(12px);
        border: 1px solid {COLORS['border']} !important;
        border-radius: 10px !important;
        color: {COLORS['text']} !important;
    }}

    /* ====================================================================
       HEADINGS
       ==================================================================== */
    h1 {{
        color: {COLORS['text']} !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}

    h2 {{
        color: {COLORS['accent']} !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
        border-bottom: 2px solid {COLORS['border']};
        padding-bottom: 0.5rem;
        margin-top: 2.5rem !important;
    }}

    h3 {{
        color: {COLORS['primary_light']} !important;
        font-weight: 600 !important;
    }}

    /* ====================================================================
       WARNING / DISCLAIMER BANNER
       ==================================================================== */
    .stAlert [data-testid="stAlertContentWarning"] {{
        background: linear-gradient(
            135deg,
            rgba(239, 85, 59, 0.15) 0%,
            rgba(45, 38, 89, 0.3) 100%
        ) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(239, 85, 59, 0.35) !important;
        border-radius: 12px !important;
        color: {COLORS['text']} !important;
    }}

    /* Also style the outer wrapper */
    [data-testid="stAlert"] {{
        border-radius: 12px !important;
    }}

    /* ====================================================================
       CAPTION / MUTED TEXT
       ==================================================================== */
    .stCaption, [data-testid="stCaption"] {{
        color: {COLORS['muted']} !important;
    }}

    /* ====================================================================
       DIVIDER
       ==================================================================== */
    hr {{
        border-color: {COLORS['border']} !important;
    }}

    /* ====================================================================
       PLOTLY CHARTS — dark background
       ==================================================================== */
    .js-plotly-plot .plotly {{
        border-radius: 12px;
    }}

    /* ====================================================================
       SIDEBAR METRIC overrides (smaller)
       ==================================================================== */
    section[data-testid="stSidebar"] [data-testid="stMetric"] {{
        padding: 0.8rem 1rem;
        border-radius: 12px;
    }}

    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
        font-size: 1.2rem !important;
    }}

    /* ====================================================================
       CUSTOM CLASSES for injected HTML
       ==================================================================== */
    .glass-card {{
        background: {COLORS['surface']};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }}

    .glass-card-header {{
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {COLORS['muted']};
        margin-bottom: 0.5rem;
    }}

    .glass-card-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {COLORS['text']};
    }}

    .glass-card-accent {{
        color: {COLORS['primary_light']};
    }}

    .disclaimer-banner {{
        background: linear-gradient(
            135deg,
            rgba(239, 85, 59, 0.12) 0%,
            rgba(45, 38, 89, 0.25) 100%
        );
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(239, 85, 59, 0.3);
        border-left: 4px solid {COLORS['warning']};
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        color: {COLORS['text']};
        font-size: 0.9rem;
        line-height: 1.5;
    }}

    .disclaimer-banner strong {{
        color: {COLORS['warning']};
    }}

    .logo-container {{
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }}

    .logo-container img {{
        width: 100px;
        height: auto;
        filter: drop-shadow(0 4px 12px {COLORS['glow']});
    }}

    .brand-title {{
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: {COLORS['text']};
        letter-spacing: 0.15em;
        margin: 0.3rem 0 0.1rem 0;
    }}

    .brand-subtitle {{
        text-align: center;
        font-size: 0.7rem;
        font-weight: 400;
        color: {COLORS['muted']};
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }}

    .branding-image {{
        width: 100%;
        border-radius: 12px;
        margin: 0.5rem 0 1rem 0;
        opacity: 0.85;
        border: 1px solid {COLORS['border']};
    }}

    .section-number {{
        display: inline-block;
        background: linear-gradient(135deg, {COLORS['primary_light']}, {COLORS['accent']});
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        width: 1.6rem;
        height: 1.6rem;
        line-height: 1.6rem;
        text-align: center;
        border-radius: 8px;
        margin-right: 0.5rem;
        vertical-align: middle;
    }}

    .erbb2-badge {{
        background: linear-gradient(135deg, rgba(87, 94, 207, 0.2), rgba(139, 127, 212, 0.15));
        backdrop-filter: blur(12px);
        border: 1px solid {COLORS['primary_light']};
        border-radius: 10px;
        padding: 0.7rem 1.2rem;
        color: {COLORS['primary_light']};
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.5rem 0 1rem 0;
    }}

    .erbb2-badge .icon {{
        font-size: 1.1rem;
    }}

    .trial-card {{
        background: rgba(45, 38, 89, 0.25);
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }}

    .trial-card a {{
        color: {COLORS['primary_light']} !important;
        font-weight: 600;
        text-decoration: none;
    }}

    .trial-card a:hover {{
        color: {COLORS['accent']} !important;
        text-decoration: underline;
    }}

    .trial-meta {{
        color: {COLORS['muted']};
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }}

    .footer-disclaimer {{
        background: {COLORS['surface']};
        backdrop-filter: blur(16px);
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: {COLORS['muted']};
        font-size: 0.82rem;
        line-height: 1.6;
        margin-top: 2rem;
    }}

    .footer-disclaimer strong {{
        color: {COLORS['warning']};
    }}

    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLORS['bg']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['primary']};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['primary_light']};
    }}
    </style>
    """, unsafe_allow_html=True)


inject_custom_css()


# -- Plotly theme defaults ------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=13),
    xaxis=dict(
        gridcolor="rgba(139, 127, 212, 0.1)",
        zerolinecolor="rgba(139, 127, 212, 0.15)",
    ),
    yaxis=dict(
        gridcolor="rgba(139, 127, 212, 0.1)",
        zerolinecolor="rgba(139, 127, 212, 0.15)",
    ),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=CHART_COLORS,
)


def styled_plotly(fig, **kwargs):
    """Apply the INVEREX dark theme to any Plotly figure."""
    layout = {**PLOTLY_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    return fig


# -- Disclaimer banner (prominent, top of page) ---------------------------
st.markdown("""
<div class="disclaimer-banner">
    <strong>RESEARCH DEMO ONLY</strong> — This is a retrospective mock ranking model
    trained on cell-line data. It is <strong>NOT</strong> a clinical treatment recommendation.
    All results require expert review.
</div>
""", unsafe_allow_html=True)

# -- Title area ------------------------------------------------------------
_title_logo_b64 = _svg_to_b64(LOGO_SVG_PATH, recolor=COLORS["accent"])
_title_logo_html = (
    f'<img src="data:image/svg+xml;base64,{_title_logo_b64}" '
    f'style="height:2.2rem;vertical-align:middle;margin-right:0.6rem;" />'
    if _title_logo_b64 else ""
)
st.markdown(f"""
<h1 style="margin-bottom: 0.1rem; display: flex; align-items: center;">
    {_title_logo_html}INVEREX
</h1>
""", unsafe_allow_html=True)

st.caption(
    "Retrospective mock demo: TCGA-BRCA patient molecular profile \u2192 "
    "disease signature \u2192 drug ranking \u2192 trial suggestions"
)


# -- Load data -------------------------------------------------------------
@st.cache_data
def load_data():
    examples = pd.read_csv(RESULTS / "example_patients.csv", index_col=0)

    # Load patient reports
    reports_path = RESULTS / "patient_reports.json"
    if reports_path.exists():
        with open(reports_path) as f:
            reports = json.load(f)
    else:
        reports = []

    # Load feature importances
    imp_path = RESULTS / "feature_importances.csv"
    importances = pd.read_csv(imp_path) if imp_path.exists() else pd.DataFrame()

    # Load model metrics
    metrics_path = RESULTS / "lightgbm_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return examples, reports, importances, metrics


examples, reports, importances, metrics = load_data()


# -- Sidebar ---------------------------------------------------------------
with st.sidebar:
    # Logo
    logo_b64 = _svg_to_b64(LOGO_SVG_PATH, recolor=COLORS["accent"])
    if logo_b64:
        st.markdown(f"""
        <div class="logo-container">
            <img src="data:image/svg+xml;base64,{logo_b64}" alt="INVEREX Logo" />
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="brand-title">INVEREX</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Breast Cancer Drug Ranking</div>',
        unsafe_allow_html=True,
    )

    # Branding image
    brand_b64 = _img_to_b64(BRANDING_IMG_PATH)
    if brand_b64:
        st.markdown(f"""
        <img class="branding-image" src="data:image/png;base64,{brand_b64}"
             alt="INVEREX Branding" />
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Patient selector
    st.markdown(
        f'<p style="color:{COLORS["accent"]};font-weight:600;font-size:0.85rem;'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">'
        f'Patient Selection</p>',
        unsafe_allow_html=True,
    )

    # Build patient labels
    patient_options = {}
    for sid, row in examples.iterrows():
        subtype = row.get("pam50_subtype", "Unknown")
        mut_cols = [
            c.replace("mut_", "")
            for c in row.index
            if c.startswith("mut_") and row[c] == 1
        ]
        muts = ", ".join(mut_cols) if mut_cols else "none"
        patient_options[sid] = f"{sid}  |  {subtype}  |  muts: {muts}"

    selected_sid = st.selectbox(
        "Select a TCGA-BRCA patient:",
        options=list(patient_options.keys()),
        format_func=lambda x: patient_options[x],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Model info
    st.markdown(
        f'<p style="color:{COLORS["accent"]};font-weight:600;font-size:0.85rem;'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">'
        f'Model Info</p>',
        unsafe_allow_html=True,
    )

    if metrics:
        st.metric("CV RMSE", f"{metrics.get('cv_rmse_mean', 0):.1f}%")
        st.metric("Training Samples", f"{metrics.get('n_samples', 0):,}")
        st.metric("Features", f"{metrics.get('n_features', 0):,}")
        # Show matched drugs from top gene features list
        top_genes = metrics.get("top_gene_features", [])
        if top_genes:
            st.metric("Top Gene", top_genes[0].get("feature", "N/A"))


# -- Main content -----------------------------------------------------------
patient_row = examples.loc[selected_sid]

# == 1. Molecular Summary ==================================================
st.markdown("""
<h2><span class="section-number">1</span> Molecular Summary</h2>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    subtype = patient_row.get("pam50_subtype", "Unknown")
    st.metric("PAM50 Subtype", subtype)
with col2:
    er = patient_row.get("er_status", "Unknown")
    pr = patient_row.get("pr_status", "Unknown")
    her2 = patient_row.get("her2_status", "Unknown")
    st.metric("ER / PR / HER2", f"{er} / {pr} / {her2}")
with col3:
    mut_cols = [
        c.replace("mut_", "")
        for c in patient_row.index
        if c.startswith("mut_") and patient_row[c] == 1
    ]
    st.metric("Key Mutations", ", ".join(mut_cols) if mut_cols else "None detected")

if patient_row.get("ERBB2_amp", 0) == 1:
    st.markdown("""
    <div class="erbb2-badge">
        <span class="icon">&#x1F9EC;</span>
        ERBB2 amplification detected (GISTIC2 score &ge; 2)
    </div>
    """, unsafe_allow_html=True)


# == 2. Drug Rankings ======================================================
st.markdown("""
<h2><span class="section-number">2</span> Computationally Ranked Drugs</h2>
""", unsafe_allow_html=True)

st.caption(
    "Drugs ranked by predicted percent inhibition from the LightGBM model. "
    "Rankings are based on expression signature reversal and have no clinical validation."
)

safe_id = selected_sid.replace("/", "_")
rankings_path = RESULTS / f"drug_rankings_{safe_id}.csv"

if rankings_path.exists():
    rankings = pd.read_csv(rankings_path)

    # Bar chart
    fig = px.bar(
        rankings.head(20),
        x="predicted_inhibition",
        y="drug_name",
        orientation="h",
        color="predicted_inhibition",
        color_continuous_scale=CHART_SEQUENTIAL,
        range_color=[0, 100],
        labels={
            "predicted_inhibition": "Predicted Inhibition (%)",
            "drug_name": "Drug",
        },
    )
    fig = styled_plotly(
        fig,
        yaxis=dict(autorange="reversed", gridcolor="rgba(139,127,212,0.1)"),
        height=600,
        coloraxis_colorbar=dict(
            title=dict(text="Inhibition %", font=dict(color=COLORS["muted"])),
            tickfont=dict(color=COLORS["muted"]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    display_cols = [
        "drug_name", "predicted_inhibition", "best_dose_um",
        "confidence", "top_contributing_genes",
    ]
    display_df = rankings[display_cols].copy()
    display_df.columns = [
        "Drug", "Pred. Inhibition (%)", "Best Dose (uM)",
        "Confidence", "Top Contributing Genes",
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.error(f"No ranking file found for {selected_sid}")


# == 3. Mapped Clinical Trials =============================================
st.markdown("""
<h2><span class="section-number">3</span> Mapped Breast Cancer Trials</h2>
""", unsafe_allow_html=True)

st.caption(
    "Active trials from ClinicalTrials.gov matching top-ranked drugs. "
    "Inclusion criteria and eligibility require manual review."
)

def _fetch_trials_for_drug(drug_name: str, max_results: int = 3) -> list[dict]:
    """Fetch active breast cancer trials from ClinicalTrials.gov API v2."""
    import requests as _req
    try:
        params = {
            "query.cond": "breast cancer",
            "query.intr": drug_name,
            "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING",
            "pageSize": max_results,
            "format": "json",
        }
        resp = _req.get("https://clinicaltrials.gov/api/v2/studies", params=params, timeout=10)
        if resp.status_code != 200:
            return []
        studies = resp.json().get("studies", [])
        results = []
        for s in studies:
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            arms = proto.get("armsInterventionsModule", {})
            results.append({
                "nct_id": ident.get("nctId"),
                "title": ident.get("briefTitle"),
                "phase": design.get("phases", [None])[0] if design.get("phases") else None,
                "status": status.get("overallStatus"),
                "interventions": [i.get("name") for i in arms.get("interventions", [])],
            })
        return results
    except Exception:
        return []

# Fetch trials live for top 5 drugs of selected patient
if rankings_path.exists():
    top_drugs = rankings.head(5)["drug_name"].tolist()
    for drug_name in top_drugs:
        trials = _fetch_trials_for_drug(drug_name)
        with st.expander(f"**{drug_name}** — {len(trials)} trial(s)"):
            if trials:
                for t in trials:
                    st.markdown(f"""
                    <div class="trial-card">
                        <a href="https://clinicaltrials.gov/study/{t['nct_id']}"
                           target="_blank">{t['nct_id']}</a>
                        <br/>{t['title']}
                        <div class="trial-meta">
                            Phase: {t.get('phase', 'N/A')} &bull;
                            Status: {t.get('status', 'N/A')} &bull;
                            Interventions: {', '.join(t.get('interventions', [])[:4])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No active breast cancer trials found for this compound.")
else:
    st.info("Select a patient to see matched clinical trials.")


# == 4. Model Rationale ====================================================
st.markdown("""
<h2><span class="section-number">4</span> Model Rationale</h2>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Feature Importance Distribution")
    if len(importances) > 0:
        gene_imp = importances[
            ~importances.feature.str.startswith("ecfp_")
            & (importances.feature != "log_dose_um")
        ]
        ecfp_imp = importances[importances.feature.str.startswith("ecfp_")]
        dose_imp = importances[importances.feature == "log_dose_um"]

        fig_pie = go.Figure(data=[go.Pie(
            labels=["Gene Expression", "Drug Fingerprint (ECFP4)", "Dose"],
            values=[
                gene_imp.importance.sum(),
                ecfp_imp.importance.sum(),
                dose_imp.importance.sum(),
            ],
            marker_colors=[COLORS["primary_light"], COLORS["accent"], COLORS["success"]],
            hole=0.45,
            textfont=dict(color=COLORS["text"]),
        )])
        fig_pie = styled_plotly(fig_pie, height=350)
        fig_pie.update_traces(
            textinfo="label+percent",
            hoverinfo="label+value+percent",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Top Gene Features")
    if len(importances) > 0:
        gene_imp_top = gene_imp.head(20)
        fig_genes = px.bar(
            gene_imp_top,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=[
                [0.0, COLORS["primary"]],
                [0.5, COLORS["primary_light"]],
                [1.0, COLORS["accent"]],
            ],
            labels={"importance": "Model Importance", "feature": "Gene"},
        )
        fig_genes = styled_plotly(
            fig_genes,
            yaxis=dict(autorange="reversed", gridcolor="rgba(139,127,212,0.1)"),
            height=400,
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_genes, use_container_width=True)


# == 5. Cohort Overview ====================================================
st.markdown("""
<h2><span class="section-number">5</span> Cohort Overview</h2>
""", unsafe_allow_html=True)

subtype_counts = examples["pam50_subtype"].value_counts()

# Subtype color map in brand palette
subtype_color_map = {
    "LumA": "#575ECF",
    "LumB": "#8B7FD4",
    "Her2": "#00CC96",
    "Basal": "#A5B4FC",
    "Normal": "#C4B5FD",
}

fig_subtypes = px.bar(
    x=subtype_counts.index,
    y=subtype_counts.values,
    labels={"x": "PAM50 Subtype", "y": "Count"},
    color=subtype_counts.index,
    color_discrete_map=subtype_color_map,
)
fig_subtypes = styled_plotly(
    fig_subtypes,
    showlegend=False,
    height=320,
    bargap=0.3,
)
fig_subtypes.update_traces(
    marker_line_width=0,
    opacity=0.9,
)
st.plotly_chart(fig_subtypes, use_container_width=True)


# -- Footer / Disclaimer ---------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div class="footer-disclaimer">
    <strong>DISCLAIMER</strong>: This is a retrospective mock ranking from the INVEREX MVP.
    It is <strong>NOT</strong> a clinical treatment recommendation.
    The model is trained on cell-line data with weak supervision labels.
    All outputs require expert review and have no clinical validation.
</div>
""", unsafe_allow_html=True)
