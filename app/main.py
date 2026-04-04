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
WORDMARK_SVG_PATH = _find_asset("logo_inverex_light_blue.svg")
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
    [data-testid="stExpander"] {{
        background: {COLORS['surface_solid']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 12px !important;
    }}

    [data-testid="stExpander"] details {{
        background: transparent !important;
        border: none !important;
    }}

    [data-testid="stExpander"] summary {{
        color: {COLORS['text']} !important;
        font-weight: 500 !important;
    }}

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background: transparent !important;
    }}

    /* Legacy selectors for older Streamlit versions */
    .streamlit-expanderHeader {{
        background: {COLORS['surface_solid']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 12px !important;
        color: {COLORS['text']} !important;
    }}

    details {{
        background: {COLORS['surface_solid']} !important;
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
_title_sym_b64 = _svg_to_b64(LOGO_SVG_PATH, recolor=COLORS["accent"])
_title_wm_b64 = _svg_to_b64(WORDMARK_SVG_PATH)
_sym_html = f'<img src="data:image/svg+xml;base64,{_title_sym_b64}" style="height:2rem;vertical-align:middle;margin-right:0.5rem;" />' if _title_sym_b64 else ""
_wm_html = f'<img src="data:image/svg+xml;base64,{_title_wm_b64}" style="height:2.2rem;vertical-align:middle;" alt="INVEREX" />' if _title_wm_b64 else "INVEREX"
st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:0.5rem;">
    {_sym_html}{_wm_html}
</div>
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
    # Symbol logo on top, wordmark below, then tagline
    symbol_b64 = _svg_to_b64(LOGO_SVG_PATH, recolor=COLORS["accent"])
    wordmark_b64 = _svg_to_b64(WORDMARK_SVG_PATH)
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0 0;">
        {'<img src="data:image/svg+xml;base64,' + symbol_b64 + '" alt="INVEREX Symbol" style="width:60px; margin-bottom:0.5rem;" />' if symbol_b64 else ''}
        <br/>
        {'<img src="data:image/svg+xml;base64,' + wordmark_b64 + '" alt="INVEREX" style="width:80%; max-width:220px;" />' if wordmark_b64 else '<div class="brand-title">INVEREX</div>'}
        <div style="color:{COLORS['muted']}; font-size:0.8rem; line-height:1.4; margin-top:0.4rem;">
            Explainable AI<br/>for decisions that matter
        </div>
    </div>
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
<h2><span class="section-number">1</span> Molecular summary</h2>
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

# Treatability score from rankings CSV (set by personalized ranker)
_rankings_path_early = RESULTS / f"drug_rankings_{selected_sid.replace('/', '_')}.csv"
if _rankings_path_early.exists():
    _rank_df = pd.read_csv(_rankings_path_early)
    if "treatability_prob" in _rank_df.columns and _rank_df["treatability_prob"].notna().any():
        _treat_prob = _rank_df["treatability_prob"].iloc[0]
        _treat_label = _rank_df["treatability_label"].iloc[0]
        _treat_color = {"high": COLORS["success"], "moderate": COLORS["accent"], "low": COLORS["warning"]}.get(_treat_label, COLORS["muted"])
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-radius:12px;padding:0.8rem 1.2rem;margin:0.5rem 0;">'
            f'<span style="color:{COLORS["muted"]};font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;">'
            f'Pan-cancer treatability</span><br/>'
            f'<span style="color:{_treat_color};font-size:1.3rem;font-weight:700;">{_treat_prob:.0%}</span>'
            f'<span style="color:{_treat_color};font-size:0.85rem;margin-left:0.5rem;">{_treat_label}</span>'
            f'<span title="Probability of treatment response estimated from expression patterns '
            f'learned across 3,730 patients and 11 cancer types. Not drug-specific."'
            f' style="color:{COLORS["muted"]};font-size:0.75rem;margin-left:0.8rem;'
            f'border-bottom:1px dotted {COLORS["muted"]};cursor:help;">what is this?</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# == 2. Drug Rankings ======================================================
st.markdown("""
<h2><span class="section-number">2</span> Computationally ranked drugs</h2>
""", unsafe_allow_html=True)

st.caption(
    "Drugs ranked by a composite score combining molecular, clinical, and computational evidence. "
    "Hover over 'Composite Personalized Score' below for details."
)

safe_id = selected_sid.replace("/", "_")
rankings_path = RESULTS / f"drug_rankings_{safe_id}.csv"

if rankings_path.exists():
    rankings = pd.read_csv(rankings_path)
    score_column = "final_score" if "final_score" in rankings.columns else "predicted_inhibition"
    score_label = "Composite Personalized Score" if score_column == "final_score" else "Predicted Inhibition (%)"

    if score_column == "final_score":
        st.markdown(
            '<p style="margin:0 0 1rem 0; font-size:0.9rem;">'
            'Ranked by <span title="Weighted sum of 5 components: '
            'RNA reversal vs LINCS signatures (35%), '
            'mutation/pathway match (22%), '
            'PAM50 subtype context (18%), '
            'clinical actionability (17%), '
            'and cell-line ML prior (8%). '
            'Evidence tiers reflect biomarker-drug pairing across 11 molecular markers."'
            f' style="border-bottom:1px dotted {COLORS["accent"]}; cursor:help; '
            f'color:{COLORS["accent"]};">'
            'Composite Personalized Score</span></p>',
            unsafe_allow_html=True,
        )

    # Bar chart
    fig = px.bar(
        rankings.head(20),
        x=score_column,
        y="drug_name",
        orientation="h",
        color=score_column,
        color_continuous_scale=CHART_SEQUENTIAL,
        range_color=[rankings[score_column].min(), rankings[score_column].max()],
        labels={
            score_column: score_label,
            "drug_name": "Drug",
        },
    )
    fig = styled_plotly(
        fig,
        yaxis=dict(autorange="reversed", gridcolor="rgba(139,127,212,0.1)"),
        height=600,
        coloraxis_colorbar=dict(
            title=dict(text=score_label, font=dict(color=COLORS["muted"])),
            tickfont=dict(color=COLORS["muted"]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    display_cols = ["drug_name"]
    display_labels = ["Drug"]
    if "final_score" in rankings.columns:
        display_cols.extend(["final_score", "evidence_tier", "predicted_inhibition", "best_dose_um", "confidence", "rationale_short"])
        display_labels.extend(["Final Score", "Evidence Tier", "ML Pred. Inhibition (%)", "Best Dose (uM)", "Confidence", "Short Rationale"])
    else:
        display_cols.extend(["predicted_inhibition", "best_dose_um", "confidence", "top_contributing_genes"])
        display_labels.extend(["Pred. Inhibition (%)", "Best Dose (uM)", "Confidence", "Top Contributing Genes"])

    display_df = rankings[[col for col in display_cols if col in rankings.columns]].copy()
    display_df.columns = display_labels[: len(display_df.columns)]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if "rationale_long" in rankings.columns:
        st.markdown("#### Top Drug Rationales")
        for _, row in rankings.head(5).iterrows():
            with st.expander(f"{row['drug_name']}  |  {row.get('evidence_tier', 'NA')}"):
                st.markdown(f"**Composite score:** {row.get('final_score', float('nan')):.3f}")
                st.markdown(f"**RNA rationale:** {row.get('rna_rationale', 'NA')}")
                st.markdown(f"**Mutation rationale:** {row.get('mutation_rationale', 'NA')}")
                st.markdown(f"**Context rationale:** {row.get('context_rationale', 'NA')}")
                st.markdown(f"**Clinical rationale:** {row.get('clinical_rationale', 'NA')}")
                st.markdown(f"**Confidence notes:** {row.get('confidence_notes', 'NA')}")
else:
    st.error(f"No ranking file found for {selected_sid}")


# == 3. Mapped Clinical Trials =============================================
st.markdown("""
<h2><span class="section-number">3</span> Mapped breast cancer trials</h2>
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
    if "excluded_flag" in rankings.columns:
        trial_rankings = rankings[~rankings["excluded_flag"].fillna(False)].copy()
    else:
        trial_rankings = rankings.copy()

    for _, row in trial_rankings.head(5).iterrows():
        drug_name = row["drug_name"]
        query_term = row.get("trial_query_term", drug_name)
        trials = _fetch_trials_for_drug(query_term)
        with st.expander(f"**{drug_name}** — {len(trials)} trial(s)"):
            if trials:
                for t in trials:
                    nct = t.get("nct_id", "")
                    title = t.get("title", "")
                    phase = t.get("phase", "N/A")
                    status = t.get("status", "N/A")
                    intv = ", ".join(t.get("interventions", [])[:4])
                    st.markdown(
                        f"[**{nct}**](https://clinicaltrials.gov/study/{nct})  \n"
                        f"{title}  \n"
                        f"*Phase:* {phase} | *Status:* {status}  \n"
                        f"*Interventions:* {intv}"
                    )
                    st.divider()
            else:
                st.write("No active breast cancer trials found for this compound.")
else:
    st.info("Select a patient to see matched clinical trials.")


# == 4. Model Rationale ====================================================
st.markdown("""
<h2><span class="section-number">4</span> Model rationale</h2>
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
<h2><span class="section-number">5</span> Cohort overview</h2>
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


# == 6. Patient validation =================================================
st.markdown("""
<h2><span class="section-number">6</span> Patient validation</h2>
""", unsafe_allow_html=True)

st.caption(
    "Models tested on real breast cancer patients with known drug response "
    "(CTR-DB 2.0 / GEO). Leave-one-dataset-out cross-validation."
)

_val_path = RESULTS / "ctrdb_validation_results.csv"
_lodo_path = RESULTS / "patient_model_lodo_results.csv"
_pancancer_path = RESULTS / "pan_cancer_model_lodo_results.csv"

if _val_path.exists() and _lodo_path.exists():
    _val = pd.read_csv(_val_path)
    _lodo = pd.read_csv(_lodo_path)
    _lodo_datasets = _lodo[_lodo["held_out_dataset"] != "MEAN"]
    _lodo_mean = _lodo[_lodo["held_out_dataset"] == "MEAN"]

    # Load pan-cancer results if available
    _pancancer_breast_auc = None
    _pancancer_auc_map = {}
    if _pancancer_path.exists():
        _pc = pd.read_csv(_pancancer_path)
        _pc_breast = _pc[_pc.get("cancer_type", pd.Series(dtype=str)).str.contains("Breast", case=False, na=False)]
        if not _pc_breast.empty:
            _pancancer_breast_auc = _pc_breast["auc"].mean()
            _pancancer_auc_map = dict(zip(_pc["held_out_dataset"], _pc["auc"]))

    # Summary metrics
    n_metric_cols = 4 if _pancancer_breast_auc else 3
    metric_cols = st.columns(n_metric_cols)
    with metric_cols[0]:
        st.metric(
            "Cell-line model",
            f"AUC {_val['auc'].mean():.2f}",
            help="LINCS/GDSC2 cell-line LightGBM tested on patients (0.5 = random)",
        )
    with metric_cols[1]:
        _lodo_auc = float(_lodo_mean["auc"].iloc[0]) if not _lodo_mean.empty else 0
        st.metric(
            "Breast-only patient model",
            f"AUC {_lodo_auc:.2f}",
            help="Trained on 10 breast cancer CTR-DB datasets, LODO-CV",
        )
    if _pancancer_breast_auc:
        with metric_cols[2]:
            _delta = f"+{(_pancancer_breast_auc - _lodo_auc):.2f}" if _pancancer_breast_auc > _lodo_auc else f"{(_pancancer_breast_auc - _lodo_auc):.2f}"
            st.metric(
                "Pan-cancer patient model",
                f"AUC {_pancancer_breast_auc:.2f}",
                delta=_delta,
                help="Trained on 3,730 patients across 11 cancer types, evaluated on breast held-out sets",
            )
    with metric_cols[-1]:
        st.metric(
            "Validation datasets",
            f"{len(_val)} datasets",
            help=f"{int(_val['n_patients'].sum()):,} patients total",
        )

    # Per-dataset comparison chart
    _combined = pd.DataFrame({
        "Dataset": _val["geo_id"],
        "Drug": _val["drug"].str[:30],
        "N": _val["n_patients"],
        "Cell-line AUC": _val["auc"],
    })
    _lodo_auc_map = dict(zip(_lodo_datasets["held_out_dataset"], _lodo_datasets["auc"]))
    _combined["Breast-only AUC"] = _combined["Dataset"].map(_lodo_auc_map)
    if _pancancer_auc_map:
        _combined["Pan-cancer AUC"] = _combined["Dataset"].map(_pancancer_auc_map)

    _melt_cols = ["Cell-line AUC", "Breast-only AUC"]
    _color_map = {
        "Cell-line AUC": COLORS["muted"],
        "Breast-only AUC": COLORS["primary_light"],
    }
    if "Pan-cancer AUC" in _combined.columns:
        _melt_cols.append("Pan-cancer AUC")
        _color_map["Pan-cancer AUC"] = COLORS["success"]

    _melted = _combined.melt(
        id_vars=["Dataset", "Drug", "N"],
        value_vars=_melt_cols,
        var_name="Model",
        value_name="AUC",
    ).dropna(subset=["AUC"])

    fig_val = px.bar(
        _melted,
        x="Dataset",
        y="AUC",
        color="Model",
        barmode="group",
        color_discrete_map=_color_map,
        hover_data=["Drug", "N"],
    )
    fig_val.add_hline(
        y=0.5, line_dash="dot",
        line_color=COLORS["warning"],
        annotation_text="random (0.5)",
        annotation_font_color=COLORS["muted"],
    )
    fig_val = styled_plotly(fig_val, height=380, bargap=0.25)
    st.plotly_chart(fig_val, use_container_width=True)

    # Table
    with st.expander("Per-dataset details"):
        _display = _combined.copy()
        for _col in ["Cell-line AUC", "Breast-only AUC", "Pan-cancer AUC"]:
            if _col in _display.columns:
                _display[_col] = _display[_col].map(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "—"
                )
        st.dataframe(_display, use_container_width=True, hide_index=True)
else:
    st.info("Validation results not yet generated. Run the CTR-DB validation pipeline.")


# == 7. Drug recommendation engine =========================================
st.markdown("""
<h2><span class="section-number">7</span> Drug recommendation engine</h2>
""", unsafe_allow_html=True)

st.caption(
    "Evaluate any drug — known or new — against the selected patient. "
    "For new drugs, provide a SMILES structure and known targets."
)

_rec_tab_known, _rec_tab_new = st.tabs(["Known drug", "New drug"])

with _rec_tab_known:
    _known_drug = st.selectbox(
        "Select a drug from the training set:",
        options=[""] + sorted(rankings["drug_name"].tolist()) if rankings_path.exists() else [""],
        key="known_drug_select",
    )
    if _known_drug:
        # Show existing ranking info for this drug
        _drug_row = rankings[rankings["drug_name"] == _known_drug]
        if not _drug_row.empty:
            _dr = _drug_row.iloc[0]
            _score_col = "final_score" if "final_score" in _dr.index else "predicted_inhibition"
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Score", f"{_dr[_score_col]:.3f}")
            with col_b:
                st.metric("Evidence tier", _dr.get("evidence_tier", "N/A"))
            with col_c:
                st.metric("Confidence", _dr.get("confidence", "N/A"))
            if "rationale_short" in _dr.index:
                st.markdown(f"**Rationale:** {_dr['rationale_short']}")
            # Trial matches
            _drug_trials = _fetch_trials_for_drug(_known_drug)
            if _drug_trials:
                with st.expander(f"Active trials for {_known_drug} ({len(_drug_trials)})"):
                    for t in _drug_trials:
                        nct = t.get("nct_id", "")
                        st.markdown(
                            f"[**{nct}**](https://clinicaltrials.gov/study/{nct})  \n"
                            f"{t.get('title', '')}  \n"
                            f"*Phase:* {t.get('phase', 'N/A')} | *Status:* {t.get('status', 'N/A')}"
                        )
                        st.divider()

with _rec_tab_new:
    st.markdown(
        f'<p style="color:{COLORS["muted"]};font-size:0.85rem;">'
        f'Enter a SMILES structure and known gene targets to evaluate a new drug '
        f'against this patient. The engine finds structurally similar drugs in our '
        f'training set and transfers their predictions.</p>',
        unsafe_allow_html=True,
    )
    _new_name = st.text_input("Drug name", value="", placeholder="e.g. INVEREX-001", key="new_drug_name")
    _new_smiles = st.text_input("SMILES", value="", placeholder="e.g. CC1=C(C=C(C=C1)NC(=O)...)F", key="new_drug_smiles")
    _new_targets = st.text_input("Gene targets (comma-separated)", value="", placeholder="e.g. CDK4, CDK6", key="new_drug_targets")

    if _new_smiles and _new_name:
        _targets_list = [t.strip().upper() for t in _new_targets.split(",") if t.strip()] if _new_targets else []

        try:
            from src.recommendation.trial_recommender import TrialRecommender
            _recommender = TrialRecommender.from_project_artifacts()
            _rec_results = _recommender.recommend_for_patient(
                selected_sid,
                candidate_drugs=[{"name": _new_name, "smiles": _new_smiles, "targets": _targets_list}],
                expression=None, cohort=None, top_k=1,
            )
            if not _rec_results.empty:
                _r = _rec_results.iloc[0]
                st.markdown("---")
                col_n1, col_n2, col_n3 = st.columns(3)
                with col_n1:
                    st.metric("Score", f"{_r.get('score', 0):.3f}")
                with col_n2:
                    _conf = _r.get("confidence", "low")
                    _conf_color = {"very_high": COLORS["success"], "high": COLORS["success"],
                                   "moderate": COLORS["accent"], "low": COLORS["warning"],
                                   "very_low": COLORS["warning"]}.get(_conf, COLORS["muted"])
                    st.metric("Confidence", _conf)
                with col_n3:
                    st.metric("Scenario", _r.get("scenario", "new"))

                # Nearest analogs
                _analogs = _r.get("nearest_analogs", [])
                if _analogs:
                    st.markdown("**Nearest known analogs:**")
                    for _a in _analogs[:5]:
                        if isinstance(_a, dict):
                            st.markdown(f"- {_a.get('drug', '?')} (similarity: {_a.get('similarity', 0):.2f})")
                        elif isinstance(_a, (list, tuple)) and len(_a) >= 2:
                            st.markdown(f"- {_a[0]} (similarity: {_a[1]:.2f})")
                        else:
                            st.markdown(f"- {_a}")

                # Rationale
                _rationale = _r.get("rationale", "")
                if _rationale:
                    st.markdown(f"**Rationale:** {_rationale}")

                # Target vulnerability
                if _targets_list:
                    _vuln = _r.get("target_vulnerability", {})
                    if isinstance(_vuln, dict) and _vuln:
                        _vs = _vuln.get("vulnerability_score", 0)
                        _nd = _vuln.get("n_targets_dysregulated", 0)
                        st.markdown(
                            f"**Target vulnerability:** {_vs:.2f} "
                            f"({_nd}/{len(_targets_list)} targets dysregulated in this patient)"
                        )

                # Trial matches for new drug (by name + by target)
                _new_trials = _r.get("trial_matches", [])
                if not _new_trials:
                    _new_trials = _fetch_trials_for_drug(_new_name)
                    if not _new_trials and _targets_list:
                        for _tgt in _targets_list[:2]:
                            _new_trials.extend(_fetch_trials_for_drug(f"{_tgt} inhibitor breast cancer"))
                if _new_trials:
                    with st.expander(f"Related trials ({len(_new_trials)})"):
                        for t in _new_trials[:5]:
                            nct = t.get("nct_id", "")
                            st.markdown(
                                f"[**{nct}**](https://clinicaltrials.gov/study/{nct})  \n"
                                f"{t.get('title', '')}  \n"
                                f"*Phase:* {t.get('phase', 'N/A')} | *Status:* {t.get('status', 'N/A')}"
                            )
                            st.divider()
            else:
                st.warning("Could not generate recommendation. Check SMILES validity.")
        except Exception as _exc:
            st.error(f"Recommendation engine error: {_exc}")
    elif _new_name and not _new_smiles:
        st.info("Enter a SMILES structure to evaluate this drug.")


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
