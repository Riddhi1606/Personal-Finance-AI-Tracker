"""
theme.py
--------
Design system for the Personal Finance AI Tracker dashboard.

Design direction: an Indian bank passbook / ledger book aesthetic —
ruled lines, stub-style entries, a warm marigold accent (money/festival
gold) instead of a generic fintech teal-and-white template.

Palette:
    Ink Navy    #12233F   headers, primary text
    Paper       #FBF9F4   page background
    Card        #FFFFFF   card surfaces
    Marigold    #E8A93B   primary accent (positive emphasis, brand)
    Deep Teal   #0F6B5C   savings / positive values
    Terracotta  #C1443C   anomalies / alerts / negative values
    Slate       #5B6472   secondary text

Type:
    Fraunces        display / headings (characterful serif)
    Space Grotesk   numbers, amounts (ledger-figure feel)
    Inter           body text, labels, UI chrome
"""

import streamlit as st

COLORS = {
    "ink": "#12233F",
    "paper": "#FBF9F4",
    "card": "#FFFFFF",
    "marigold": "#E8A93B",
    "teal": "#0F6B5C",
    "terracotta": "#C1443C",
    "slate": "#5B6472",
    "hairline": "#E4DFD3",
}

# Category color sequence for charts — warm/ink palette instead of Plotly defaults
CATEGORY_COLORS = [
    "#E8A93B", "#12233F", "#0F6B5C", "#C1443C", "#8C6E4A",
    "#5B6472", "#D9C27E", "#3B5D73", "#A85C32", "#6B8F71", "#B08968",
]

PLOTLY_TEMPLATE = {
    "layout": {
        "font": {"family": "Inter, sans-serif", "color": COLORS["ink"]},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "colorway": CATEGORY_COLORS,
        "title": {"font": {"family": "Fraunces, serif", "size": 20, "color": COLORS["ink"]}},
        "xaxis": {"gridcolor": COLORS["hairline"], "zerolinecolor": COLORS["hairline"]},
        "yaxis": {"gridcolor": COLORS["hairline"], "zerolinecolor": COLORS["hairline"]},
        "legend": {"font": {"family": "Inter, sans-serif"}},
    }
}


def inject_css():
    st.markdown(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;1,500&family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}
        .stApp {{
            background-color: {COLORS['paper']};
        }}
        #MainMenu, footer, .stDeployButton {{ visibility: hidden; }}

        /* ---- Hero header ---- */
        .fat-hero {{
            background: linear-gradient(135deg, {COLORS['ink']} 0%, #1B3358 100%);
            border-radius: 10px;
            padding: 2rem 2.25rem;
            margin-bottom: 1.75rem;
            position: relative;
            overflow: hidden;
        }}
        .fat-hero::after {{
            content: "";
            position: absolute;
            top: -40%; right: -8%;
            width: 260px; height: 260px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(232,169,59,0.28) 0%, rgba(232,169,59,0) 70%);
        }}
        .fat-hero h1 {{
            font-family: 'Fraunces', serif;
            font-weight: 600;
            font-style: italic;
            color: #FBF9F4;
            font-size: 2.1rem;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.01em;
        }}
        .fat-hero p {{
            font-family: 'Inter', sans-serif;
            color: #C9D2E0;
            font-size: 0.95rem;
            margin: 0;
        }}
        .fat-eyebrow {{
            display: inline-block;
            font-family: 'Space Grotesk', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {COLORS['marigold']};
            border: 1px solid rgba(232,169,59,0.4);
            border-radius: 20px;
            padding: 0.2rem 0.7rem;
            margin-bottom: 0.75rem;
        }}

        /* ---- KPI passbook-stub cards ---- */
        .kpi-row {{ display: flex; gap: 1rem; margin-bottom: 0.5rem; flex-wrap: wrap; }}
        .kpi-card {{
            background: {COLORS['card']};
            border-top: 3px solid var(--accent, {COLORS['marigold']});
            border-radius: 6px;
            padding: 1rem 1.25rem 0.9rem 1.25rem;
            flex: 1;
            min-width: 190px;
            box-shadow: 0 1px 2px rgba(18,35,63,0.06);
            position: relative;
        }}
        .kpi-card::after {{
            content: "";
            position: absolute;
            left: 1.25rem; right: 1.25rem; bottom: 0.55rem;
            border-bottom: 1px dashed {COLORS['hairline']};
        }}
        .kpi-label {{
            font-family: 'Inter', sans-serif;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: {COLORS['slate']};
            margin-bottom: 0.3rem;
        }}
        .kpi-value {{
            font-family: 'Space Grotesk', monospace;
            font-weight: 700;
            font-size: 1.6rem;
            color: {COLORS['ink']};
            line-height: 1.1;
        }}
        .kpi-sub {{
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            color: {COLORS['slate']};
            margin-top: 0.85rem;
        }}

        /* ---- Section headers ---- */
        .fat-section-title {{
            font-family: 'Fraunces', serif;
            font-weight: 600;
            font-size: 1.35rem;
            color: {COLORS['ink']};
            margin: 0.25rem 0 0.15rem 0;
        }}
        .fat-section-sub {{
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            color: {COLORS['slate']};
            margin-bottom: 1rem;
            max-width: 68ch;
        }}

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            border-bottom: 1px solid {COLORS['hairline']};
        }}
        .stTabs [data-baseweb="tab"] {{
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            color: {COLORS['slate']};
            height: 42px;
        }}
        .stTabs [aria-selected="true"] {{
            color: {COLORS['ink']} !important;
            border-bottom: 2px solid {COLORS['marigold']} !important;
        }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {{
            background-color: {COLORS['ink']};
        }}
        section[data-testid="stSidebar"] * {{
            color: #E7ECF3 !important;
        }}
        section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {{
            color: #93A2BC !important;
        }}

        /* ---- Dataframes ---- */
        [data-testid="stDataFrame"] {{
            border: 1px solid {COLORS['hairline']};
            border-radius: 6px;
        }}

        /* ---- Alert / info boxes ---- */
        div[data-testid="stAlert"] {{
            border-radius: 6px;
            font-family: 'Inter', sans-serif;
        }}

        /* ---- Divider ---- */
        .fat-hairline {{
            border: none;
            border-top: 1px dashed {COLORS['hairline']};
            margin: 1.5rem 0;
        }}
    </style>
    """, unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = ""):
    eyebrow_html = f'<div class="fat-eyebrow">{eyebrow}</div><br/>' if eyebrow else ""
    st.markdown(f"""
    <div class="fat-hero">
        {eyebrow_html}
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, sublabel: str = "", accent: str = None):
    accent = accent or COLORS["marigold"]
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sublabel}</div>
    </div>
    """, unsafe_allow_html=True)


def section_title(title: str, subtitle: str = ""):
    sub_html = f'<div class="fat-section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="fat-section-title">{title}</div>
    {sub_html}
    """, unsafe_allow_html=True)


def hairline():
    st.markdown('<hr class="fat-hairline"/>', unsafe_allow_html=True)


def style_fig(fig, height=380):
    """Applies the ledger theme to a Plotly figure in-place-ish (returns fig)."""
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=height, margin=dict(t=50, l=10, r=10, b=10))
    return fig
