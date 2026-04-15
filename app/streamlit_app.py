"""
app/streamlit_app.py — Credit Risk Decision System
System-themed (auto dark/light), polished 3-panel layout
"""

import sys, time
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.utils.shared import load_borrowers, load_retriever, load_scorer

st.set_page_config(
    page_title="Credit Risk Decision System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Geist:wght@300;400;500;600&display=swap');

/* ═══════════════════════════════════════════
   CSS VARIABLES — light default
   ═══════════════════════════════════════════ */
:root {
  --bg-base:        #F5F3EF;
  --bg-surface:     #FFFFFF;
  --bg-raised:      #FAF9F7;
  --bg-subtle:      #F0EDE8;
  --border:         #E5E0D8;
  --border-strong:  #D4CECC;
  --text-primary:   #1C1917;
  --text-secondary: #78716C;
  --text-muted:     #A8A29E;
  --accent:         #6366F1;
  --accent-bg:      #EEF2FF;
  --accent-border:  #C7D2FE;

  --tier-low-text:    #15803D; --tier-low-bg:      #F0FDF4; --tier-low-border:  #BBF7D0;
  --tier-med-text:    #B45309; --tier-med-bg:      #FFFBEB; --tier-med-border:  #FDE68A;
  --tier-high-text:   #C2410C; --tier-high-bg:     #FFF7ED; --tier-high-border: #FED7AA;
  --tier-crit-text:   #B91C1C; --tier-crit-bg:     #FEF2F2; --tier-crit-border: #FECACA;

  --chunk-uw-border: #93C5FD; --chunk-uw-bg: #F0F7FF; --chunk-uw-tag-bg: #DBEAFE; --chunk-uw-tag-text: #1D4ED8;
  --chunk-co-border: #FCD34D; --chunk-co-bg: #FFFDF0; --chunk-co-tag-bg: #FEF9C3; --chunk-co-tag-text: #92400E;
  --chunk-sv-border: #6EE7B7; --chunk-sv-bg: #F0FFF8; --chunk-sv-tag-bg: #D1FAE5; --chunk-sv-tag-text: #065F46;
  --chunk-cp-border: #FCA5A5; --chunk-cp-bg: #FFF5F5; --chunk-cp-tag-bg: #FEE2E2; --chunk-cp-tag-text: #991B1B;

  --summary-bg: #F0FDF4; --summary-border: #BBF7D0; --summary-text: #14532D;
  --reason-bg:  #FEFCE8; --reason-border:  #FEF08A; --reason-text:  #713F12;
  --json-bg:    #FAFAF9; --json-border:    #E5E0D8; --json-text:    #292524;

  --shadow-sm: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
  --shadow-md: 0 4px 12px rgba(0,0,0,.08), 0 2px 4px rgba(0,0,0,.04);
  --radius-sm: 8px; --radius-md: 10px; --radius-lg: 14px;
}

/* ═══════════════════════════════════════════
   DARK MODE
   ═══════════════════════════════════════════ */
@media (prefers-color-scheme: dark) {
  :root {
    --bg-base:        #0C0A09;
    --bg-surface:     #161412;
    --bg-raised:      #1C1917;
    --bg-subtle:      #231F1C;
    --border:         #292524;
    --border-strong:  #3D3632;
    --text-primary:   #F5F0EB;
    --text-secondary: #A8A29E;
    --text-muted:     #57534E;
    --accent:         #818CF8;
    --accent-bg:      #1E1B4B;
    --accent-border:  #3730A3;

    --tier-low-text:    #4ADE80; --tier-low-bg:      #052E16; --tier-low-border:  #14532D;
    --tier-med-text:    #FCD34D; --tier-med-bg:      #1C1202; --tier-med-border:  #78350F;
    --tier-high-text:   #FB923C; --tier-high-bg:     #1C0F02; --tier-high-border: #7C2D12;
    --tier-crit-text:   #F87171; --tier-crit-bg:     #1C0505; --tier-crit-border: #7F1D1D;

    --chunk-uw-border: #1D4ED8; --chunk-uw-bg: #0D1B3E; --chunk-uw-tag-bg: #1E3A5F; --chunk-uw-tag-text: #93C5FD;
    --chunk-co-border: #B45309; --chunk-co-bg: #1C1000; --chunk-co-tag-bg: #2D1B00; --chunk-co-tag-text: #FCD34D;
    --chunk-sv-border: #047857; --chunk-sv-bg: #022C22; --chunk-sv-tag-bg: #064E3B; --chunk-sv-tag-text: #6EE7B7;
    --chunk-cp-border: #B91C1C; --chunk-cp-bg: #1C0505; --chunk-cp-tag-bg: #450A0A; --chunk-cp-tag-text: #FCA5A5;

    --summary-bg: #052E16; --summary-border: #14532D; --summary-text: #4ADE80;
    --reason-bg:  #1C1202; --reason-border:  #78350F; --reason-text:  #FCD34D;
    --json-bg:    #161412; --json-border:    #292524; --json-text:    #D6D3D1;

    --shadow-sm: 0 1px 3px rgba(0,0,0,.4);
    --shadow-md: 0 4px 12px rgba(0,0,0,.5);
  }
}


/* ═══════════════════════════════════════════
   TOP BAR ELEMENTS
   ═══════════════════════════════════════════ */
.topbar-left { display: flex; align-items: center; gap: 12px; }
.topbar-logo {
  width: 28px; height: 28px;
  background: linear-gradient(135deg, var(--accent), #A78BFA);
  border-radius: 7px;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px;
  box-shadow: 0 2px 8px rgba(99,102,241,.35);
}
.topbar-title {
  font-family: 'Geist Mono', monospace;
  font-size: 12px; font-weight: 600;
  letter-spacing: .06em; text-transform: uppercase;
  color: var(--text-primary);
}
.topbar-divider { width:1px; height:18px; background: var(--border); }
.topbar-sub { font-size: 11px; color: var(--text-muted); font-family: 'Geist Mono', monospace; }
.pill {
  font-family: 'Geist Mono', monospace; font-size: 10px;
  padding: 3px 10px; border-radius: 20px;
  background: var(--accent-bg); color: var(--accent);
  border: 1px solid var(--accent-border); font-weight: 500;
}
.pill.green { background: var(--tier-low-bg); color: var(--tier-low-text); border-color: var(--tier-low-border); }
.pill.amber { background: var(--tier-med-bg); color: var(--tier-med-text); border-color: var(--tier-med-border); }

/* panel shell */
.panel-shell {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 18px 20px 18px;
  min-height: calc(100vh - 118px);
  box-shadow: var(--shadow-sm);
}

/* chunk type variants */
.chunk.collections { border-left-color: var(--chunk-co-border); background: var(--chunk-co-bg); }
.chunk.servicing   { border-left-color: var(--chunk-sv-border); background: var(--chunk-sv-bg); }
.chunk.complaint   { border-left-color: var(--chunk-cp-border); background: var(--chunk-cp-bg); }
.chunk.underwriter { border-left-color: var(--chunk-uw-border); background: var(--chunk-uw-bg); }

.chunk-num { font-weight: 600; color: var(--text-secondary); }
.type-tag {
  font-family: 'Geist Mono', monospace; font-size: 9px; font-weight: 600;
  padding: 2px 8px; border-radius: 20px; letter-spacing: .04em;
  background: var(--chunk-uw-tag-bg); color: var(--chunk-uw-tag-text);
}
.type-tag.collections { background: var(--chunk-co-tag-bg); color: var(--chunk-co-tag-text); }
.type-tag.servicing   { background: var(--chunk-sv-tag-bg); color: var(--chunk-sv-tag-text); }
.type-tag.complaint   { background: var(--chunk-cp-tag-bg); color: var(--chunk-cp-tag-text); }
.type-tag.underwriter { background: var(--chunk-uw-tag-bg); color: var(--chunk-uw-tag-text); }

.score-badge {
  font-family: 'Geist Mono', monospace; font-size: 9px;
  padding: 2px 7px; border-radius: 20px;
  background: var(--accent-bg); color: var(--accent); border: 1px solid var(--accent-border);
}
.chunk-note-id { color: var(--text-muted); font-size: 10px; }

/* tier card elements */
.tier-card::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse at top, rgba(255,255,255,.07), transparent 70%);
  pointer-events: none;
}
.tier-lbl {
  font-family: 'Geist Mono', monospace; font-size: 9px;
  letter-spacing: .2em; text-transform: uppercase; margin-bottom: 6px; opacity: .7;
}
.tier-val { font-family: 'Geist Mono', monospace; font-size: 34px; font-weight: 600; letter-spacing: .03em; }
.tier-sub { font-size: 12px; opacity: .65; margin-top: 4px; }

.stat-v { font-family:'Geist Mono',monospace; font-size:18px; font-weight:600; color:var(--text-primary); }
.stat-l { font-family:'Geist Mono',monospace; font-size:9px; color:var(--text-muted); text-transform:uppercase; letter-spacing:.1em; margin-top:2px; }

.conf-track { background: var(--bg-subtle); border-radius:4px; height:6px; overflow:hidden; margin-top:5px; }
.conf-fill  { height:6px; border-radius:4px; transition: width .6s ease; }

.json-blk {
  background: var(--json-bg); border: 1px solid var(--json-border); border-radius: var(--radius-md);
  padding: 14px 16px; font-family: 'Geist Mono', monospace; font-size: 11px;
  color: var(--json-text); line-height: 2; white-space: pre-wrap; word-break: break-word;
}
.chip {
  display: inline-block; background: var(--accent-bg); border: 1px solid var(--accent-border);
  border-radius: 20px; padding: 3px 11px; font-size: 11px; color: var(--accent);
  margin: 3px 3px 3px 0; font-family: 'Geist Mono', monospace; font-weight: 500;
}
.summary-box {
  background: var(--summary-bg); border: 1px solid var(--summary-border); border-radius: var(--radius-md);
  padding: 14px 16px; font-size: 13px; color: var(--summary-text); line-height: 1.7;
}
.reason-box {
  background: var(--reason-bg); border: 1px solid var(--reason-border); border-radius: var(--radius-md);
  padding: 14px 16px; font-size: 12px; color: var(--reason-text); line-height: 1.75;
}
.load-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
  animation: blink 1s infinite; flex-shrink: 0;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ═══════════════════════════════════════════
   COLUMN / PAGE LAYOUT
   ═══════════════════════════════════════════ */
html, body, [class*="css"] {
  font-family: 'Geist', sans-serif !important;
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }

.block-container {
  padding: 16px 22px 24px 22px !important;
  max-width: 1680px !important;
}

[data-testid="stVerticalBlock"] {
  gap: 8px !important;
}

[data-testid="column"] {
  padding: 0 8px !important;
  border-right: none !important;
}

/* ═══════════════════════════════════════════
   TOP BAR
   ═══════════════════════════════════════════ */
.topbar {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 0 22px;
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: var(--shadow-sm);
  position: sticky;
  top: 10px;
  z-index: 200;
  margin-bottom: 14px;
}

/* ═══════════════════════════════════════════
   PANEL SHELL
   ═══════════════════════════════════════════ */
.panel-shell {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 18px 20px 18px;
  min-height: calc(100vh - 118px);
  box-shadow: var(--shadow-sm);
}

.panel-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: 'Geist Mono', monospace;
  font-size: 9px;
  font-weight: 600;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 18px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}

.panel-num {
  width: 18px;
  height: 18px;
  border-radius: 5px;
  background: var(--accent-bg);
  color: var(--accent);
  font-size: 9px;
  font-weight: 700;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--accent-border);
}

/* ═══════════════════════════════════════════
   SECTION TITLES
   ═══════════════════════════════════════════ */
.sec {
  font-family: 'Geist Mono', monospace;
  font-size: 9px;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 20px 0 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.sec::after { content:''; flex:1; height:1px; background: var(--border); }

/* ═══════════════════════════════════════════
   INPUT PANEL
   ═══════════════════════════════════════════ */
.ctrl-lbl {
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  color: var(--text-muted);
  letter-spacing: .06em;
  text-transform: uppercase;
  margin-bottom: 6px;
  margin-top: 16px;
}

.snap-card {
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px;
  margin: 14px 0 12px;
  box-shadow: var(--shadow-sm);
}

.snap-header {
  font-family: 'Geist Mono', monospace;
  font-size: 9px;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 10px;
}

.snap-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--bg-subtle);
}
.snap-row:last-child {
  border-bottom: none;
  padding-bottom: 0;
}
.snap-k {
  font-family:'Geist Mono',monospace;
  font-size:11px;
  color: var(--text-muted);
}
.snap-v {
  font-family:'Geist Mono',monospace;
  font-size:11px;
  font-weight:500;
  color: var(--text-primary);
}
            
.retrieval-scroll {
  max-height: 620px;
  overflow-y: auto;
  padding-right: 4px;
}

/* ═══════════════════════════════════════════
   SIGNAL ROWS
   ═══════════════════════════════════════════ */
.sig-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 12px;
  margin-bottom: 6px;
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: 8px;
  transition: border-color .15s;
}
.sig-row:hover { border-color: var(--border-strong); }
.sig-k {
  font-family:'Geist Mono',monospace;
  font-size:11px;
  color: var(--text-secondary);
}
.sig-v {
  font-family:'Geist Mono',monospace;
  font-size:12px;
  font-weight:500;
  color: var(--text-primary);
}

/* ═══════════════════════════════════════════
   CHUNK CARDS
   ═══════════════════════════════════════════ */
.chunk {
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  border-left: 3px solid var(--chunk-uw-border);
  padding: 12px 14px;
  margin-bottom: 10px;
  background: var(--chunk-uw-bg);
  box-shadow: var(--shadow-sm);
  transition: box-shadow .15s;
}
.chunk:hover { box-shadow: var(--shadow-md); }

.chunk-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  color: var(--text-muted);
}

.chunk-text {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.7;
  font-family: 'Geist', sans-serif;
}

/* ═══════════════════════════════════════════
   DECISION OUTPUT
   ═══════════════════════════════════════════ */
.tier-card {
  border-radius: var(--radius-lg);
  padding: 24px 24px;
  text-align: center;
  margin-bottom: 16px;
  border: 1px solid;
  box-shadow: var(--shadow-md);
  position: relative;
  overflow: hidden;
}

.stats {
  display: flex;
  gap: 10px;
  margin-bottom: 16px;
}

.stat {
  flex: 1;
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 12px 10px;
  text-align: center;
  box-shadow: var(--shadow-sm);
}

.json-blk,
.summary-box,
.reason-box {
  margin-bottom: 14px;
}

/* ═══════════════════════════════════════════
   EMPTY / LOADING STATES
   ═══════════════════════════════════════════ */
.empty {
  text-align: center;
  padding: 80px 20px 40px;
  font-family: 'Geist Mono', monospace;
  font-size: 11px;
  color: var(--text-muted);
  line-height: 2;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 420px;
}
.empty-icon {
  font-size: 28px;
  margin-bottom: 10px;
  opacity: .4;
}

.load-step {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  background: var(--accent-bg);
  border: 1px solid var(--accent-border);
  border-radius: var(--radius-sm);
  font-family: 'Geist Mono', monospace;
  font-size: 11px;
  color: var(--accent);
  margin-bottom: 8px;
}

/* ═══════════════════════════════════════════
   STREAMLIT WIDGET OVERRIDES
   ═══════════════════════════════════════════ */
div[data-testid="stSelectbox"] {
  margin-bottom: 14px !important;
}

div[data-testid="stSlider"] {
  margin-top: 8px !important;
  margin-bottom: 12px !important;
}

div[data-testid="stCheckbox"] {
  margin-top: 6px !important;
  margin-bottom: 2px !important;
}

div[data-testid="stButton"] > button {
  width: 100% !important;
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-family: 'Geist Mono', monospace !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: .06em !important;
  padding: 13px !important;
  box-shadow: 0 2px 8px rgba(99,102,241,.3) !important;
  transition: all .15s !important;
}
div[data-testid="stButton"] > button:hover {
  background: #4F46E5 !important;
  box-shadow: 0 4px 14px rgba(99,102,241,.45) !important;
  transform: translateY(-1px) !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
  display: none !important;
}

div[data-testid="stSelectbox"] > div > div {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Geist Mono', monospace !important;
  font-size: 12px !important;
  color: var(--text-primary) !important;
}

div[data-testid="stCheckbox"] label {
  font-family: 'Geist Mono', monospace !important;
  font-size: 11px !important;
  color: var(--text-secondary) !important;
}
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────────────────
TIER_CFG = {
    "LOW":      ("var(--tier-low-text)",  "var(--tier-low-bg)",  "var(--tier-low-border)",  "✓ Low Risk"),
    "MEDIUM":   ("var(--tier-med-text)",  "var(--tier-med-bg)",  "var(--tier-med-border)",  "⚠ Monitor"),
    "HIGH":     ("var(--tier-high-text)", "var(--tier-high-bg)", "var(--tier-high-border)", "↑ High Risk"),
    "CRITICAL": ("var(--tier-crit-text)", "var(--tier-crit-bg)", "var(--tier-crit-border)", "✕ Critical"),
}

def tier_css(tier):
    return TIER_CFG.get(tier, ("var(--text-muted)","var(--bg-surface)","var(--border)","—"))

TYPE_CLS = {"underwriter":"underwriter","collections":"collections",
            "servicing":"servicing","complaint":"complaint"}

# ── top bar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">⚖</div>
    <div class="topbar-title">Credit Risk Decision System</div>
    <div class="topbar-divider"></div>
    <div class="topbar-sub">RAG · XGBoost · LLM</div>
  </div>
  <div style="display:flex;gap:6px;align-items:center;">
    <span class="pill">v1.0</span>
    <span class="pill green">● Live</span>
    <span class="pill amber">Groq</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ── data ─────────────────────────────────────────────────────────────────────
df = load_borrowers()
if df is None:
    st.error("No data. Run `python scripts/generate_mock_data.py` first.")
    st.stop()

col1, col2, col3 = st.columns([1.05, 1.35, 1.45], gap="medium")

# ╔══════════════════════╗
# ║  PANEL 1 — INPUT     ║
with col1:
    panel1 = st.container()
    with panel1:
        st.markdown('<div class="panel-label"><span class="panel-num">1</span> Input & Configuration</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Geist Mono', monospace; font-size:10px; color:var(--text-muted); margin-bottom:14px;">
        Active borrower pool: {len(df)}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="ctrl-lbl">Risk Tier Filter</div>', unsafe_allow_html=True)
        tier_filter = st.selectbox("t", ["All","CRITICAL","HIGH","MEDIUM","LOW"], label_visibility="collapsed")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        st.markdown('<div class="ctrl-lbl">Loan Type</div>', unsafe_allow_html=True)
        loan_types  = ["All"] + sorted(df["loan_type"].unique().tolist())
        loan_filter = st.selectbox("l", loan_types, label_visibility="collapsed")

        pool = df.copy()
        if tier_filter != "All":
            pool = pool[pool["analyst_risk_tier"] == tier_filter]
        if loan_filter != "All":
            pool = pool[pool["loan_type"] == loan_filter]

        tier_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        pool["tier_rank"] = pool["analyst_risk_tier"].map(tier_order)
        pool = pool.sort_values(["tier_rank", "loan_amount"], ascending=[True, False]).drop(columns=["tier_rank"])

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        st.markdown('<div class="ctrl-lbl">Borrower ID</div>', unsafe_allow_html=True)
        borrower_id = st.selectbox("b", pool["borrower_id"].tolist(), label_visibility="collapsed")

        row  = df[df["borrower_id"] == borrower_id].iloc[0].to_dict()
        tier = row.get("analyst_risk_tier","")
        tc, tbg, tborder, tsub = tier_css(tier)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="snap-card">
          <div class="snap-header">Borrower Snapshot</div>
          <div class="snap-row"><span class="snap-k">ID</span><span class="snap-v">{borrower_id}</span></div>
          <div class="snap-row"><span class="snap-k">Loan Type</span><span class="snap-v">{row.get('loan_type','').upper()}</span></div>
          <div class="snap-row"><span class="snap-k">Amount</span><span class="snap-v">${row.get('loan_amount',0):,.0f}</span></div>
          <div class="snap-row"><span class="snap-k">Vintage</span><span class="snap-v">{row.get('vintage_year','—')}</span></div>
          <div class="snap-row"><span class="snap-k">Analyst Tier</span><span class="snap-v" style="color:{tc};font-weight:600;">{tier} — {tsub}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec">Retrieval Controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="ctrl-lbl">Evidence Sources (top-k)</div>', unsafe_allow_html=True)
        top_k = st.slider("k", 3, 10, 5, label_visibility="collapsed")

        st.markdown('<div class="sec">Model Options</div>', unsafe_allow_html=True)
        show_reasoning = st.checkbox("Show reasoning chain", value=True)
        use_xgb = st.checkbox("Include XGBoost signal", value=False)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        run = st.button("▶ Generate Decision", type="primary")

# ╔══════════════════════════╗
# ║  PANEL 2 — RETRIEVAL     ║
# ╚══════════════════════════╝
with col2:
    panel2 = st.container()
    with panel2:
        st.markdown('<div class="panel-label"><span class="panel-num">2</span> Retrieval Layer</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-family:'Geist Mono', monospace; font-size:10px; color:var(--text-muted); margin-bottom:14px;">
        Structured + unstructured evidence assembly
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stats" style="margin-bottom:16px;">
          <div class="stat">
            <div class="stat-v">{borrower_id}</div>
            <div class="stat-l">Borrower</div>
          </div>
          <div class="stat">
            <div class="stat-v">{row.get('loan_type','—')}</div>
            <div class="stat-l">Loan Type</div>
          </div>
          <div class="stat">
            <div class="stat-v">{top_k}</div>
            <div class="stat-l">Top-K</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec">Structured Signals</div>', unsafe_allow_html=True)

        signals = [
            ("FICO Score",          str(row.get("fico_score", "—"))),
            ("DTI Ratio",           f"{row.get('dti_ratio', 0):.0%}"),
            ("Late Pmts (30d)",     str(row.get("payments_late_30d", "—"))),
            ("Late Pmts (60d)",     str(row.get("payments_late_60d", "—"))),
            ("Late Pmts (90d)",     str(row.get("payments_late_90d", "—"))),
            ("Deferrals",           str(row.get("num_deferrals", "—"))),
            ("Empl. Gap (mo)",      str(row.get("employment_gap_months", "—"))),
            ("Annual Income",       f"${row.get('annual_income', 0):,.0f}"),
            ("Credit History (yr)", str(row.get("credit_history_yrs", "—"))),
            ("Open Accounts",       str(row.get("num_open_accounts", "—"))),
        ]

        for k, v in signals:
            st.markdown(
                f'<div class="sig-row"><span class="sig-k">{k}</span><span class="sig-v">{v}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec">Retrieved Evidence</div>', unsafe_allow_html=True)

        if not run:
            st.markdown(
                '<div style="font-family:Geist Mono, monospace; font-size:10px; color:var(--text-muted); margin-bottom:8px;">Awaiting retrieval execution</div>',
                unsafe_allow_html=True
            )
            st.markdown("""
            <div class="empty">
              <div class="empty-icon">🔍</div>
              Run the pipeline to retrieve<br>grounding evidence from<br>718 indexed case notes
            </div>
            """, unsafe_allow_html=True)

        else:
            if show_reasoning:
                ph = st.empty()
                for msg in [
                    "Embedding query vector…",
                    "Searching FAISS index (718 vectors)…",
                    "Applying similarity threshold…",
                ]:
                    ph.markdown(
                        f'<div class="load-step"><div class="load-dot"></div>{msg}</div>',
                        unsafe_allow_html=True
                    )
                    time.sleep(0.4)
                ph.empty()

            retriever = load_retriever()
            chunks = retriever.retrieve(
                query=f"credit risk payment history delinquency collections hardship {row.get('loan_type','')} borrower {borrower_id}",
                filters=None,
                top_k=top_k,
            )
            st.session_state["chunks"] = chunks

            if not chunks:
                st.markdown(
                    '<div style="color:var(--text-muted);font-family:\'Geist Mono\',monospace;font-size:11px;padding:12px 0;">No chunks above similarity threshold.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="font-family:\'Geist Mono\',monospace;font-size:10px;color:var(--text-muted);margin-bottom:10px;">{len(chunks)} chunks retrieved · sorted by similarity</div>',
                    unsafe_allow_html=True
                )

                st.markdown('<div class="retrieval-scroll">', unsafe_allow_html=True)

                for i, chunk in enumerate(chunks, 1):
                    cls = TYPE_CLS.get(chunk.note_type, "underwriter")
                    st.markdown(f"""
                    <div class="chunk {cls}">
                      <div class="chunk-meta">
                        <span class="chunk-num">#{i}</span>
                        <span class="type-tag {cls}">{chunk.note_type.upper()}</span>
                        <span class="score-badge">sim {chunk.similarity_score:.3f}</span>
                        <span class="chunk-note-id">{chunk.note_id[:16]}…</span>
                      </div>
                      <div class="chunk-text">{chunk.text[:300]}{"…" if len(chunk.text) > 300 else ""}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)


# ╔══════════════════════════╗
# ║  PANEL 3 — DECISION      ║
# ╚══════════════════════════╝
with col3:
    panel3 = st.container()
    with panel3:
        st.markdown('<div class="panel-label"><span class="panel-num">3</span> Decision Output</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'Geist Mono', monospace; font-size:10px; color:var(--text-muted); margin-bottom:14px;">
        Final risk assessment, rationale, and model outputs
        </div>
        """, unsafe_allow_html=True)

        if not run:
            st.markdown(
                '<div style="font-family:Geist Mono, monospace; font-size:10px; color:var(--text-muted); margin-bottom:8px;">Decision engine idle</div>',
                unsafe_allow_html=True
            )
            st.markdown("""
            <div class="empty">
              <div class="empty-icon">⚖</div>
              Select a borrower and click<br>▶ Generate Decision<br>to run the full pipeline
            </div>
            """, unsafe_allow_html=True)

        else:
            chunks = st.session_state.get("chunks", [])

            xgb_signal = None
            if use_xgb:
                try:
                    scorer = load_scorer()
                    xgb_signal = scorer.score(row)
                except Exception as e:
                    st.warning(f"XGBoost: {e}")

            if show_reasoning:
                ph2 = st.empty()
                for msg in [
                    "Building context block…",
                    "Calling Groq LLM…",
                    "Parsing structured output…",
                ]:
                    ph2.markdown(
                        f'<div class="load-step"><div class="load-dot"></div>{msg}</div>',
                        unsafe_allow_html=True
                    )
                    time.sleep(0.35)
                ph2.empty()

            from generation.generator import generate_assessment
            result = generate_assessment(borrower_id, row, chunks, xgb_signal=xgb_signal)

            rtier = result.risk_tier
            rc, rbg, rborder, rsub = tier_css(rtier)

            st.markdown(f"""
            <div class="tier-card" style="background:{rbg};border-color:{rborder};">
              <div class="tier-lbl" style="color:{rc};">AI Risk Decision</div>
              <div class="tier-val" style="color:{rc};">{rtier}</div>
              <div class="tier-sub" style="color:{rc};">{rsub}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            conf_pct = int(result.confidence * 100)
            bar_clr = (
                "var(--tier-low-text)" if result.confidence > 0.75
                else "var(--tier-med-text)" if result.confidence > 0.5
                else "var(--tier-crit-text)"
            )

            st.markdown(f"""
            <div class="stats">
              <div class="stat">
                <div class="stat-v">{result.confidence:.0%}</div>
                <div class="stat-l">Confidence</div>
              </div>
              <div class="stat">
                <div class="stat-v">{result.sources_used}</div>
                <div class="stat-l">Sources</div>
              </div>
              <div class="stat">
                <div class="stat-v">{result.latency_ms}ms</div>
                <div class="stat-l">Latency</div>
              </div>
            </div>
            <div style="margin-bottom:16px;">
              <div style="display:flex;justify-content:space-between;
                          font-family:'Geist Mono',monospace;font-size:9px;
                          color:var(--text-muted);margin-bottom:4px;">
                <span>CONFIDENCE</span><span>{result.confidence:.2f}</span>
              </div>
              <div class="conf-track">
                <div class="conf-fill" style="width:{conf_pct}%;background:{bar_clr};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            st.markdown('<div class="sec">Structured Output</div>', unsafe_allow_html=True)
            sigs_fmt = "\n".join(f'    "{s[:58]}"' for s in result.key_signals[:4])
            json_out = (
                '{\n'
                f'  "risk_tier": "{rtier}",\n'
                f'  "confidence": {result.confidence:.2f},\n'
                f'  "decision": "{result.decision[:65]}{"..." if len(result.decision) > 65 else ""}",\n'
                f'  "key_signals": [\n{sigs_fmt}\n  ],\n'
                f'  "sources_used": {result.sources_used}\n'
                '}'
            )
            st.markdown(f'<div class="json-blk">{json_out}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            if result.key_signals:
                st.markdown('<div class="sec">Key Risk Signals</div>', unsafe_allow_html=True)
                chips = "".join(
                    f'<span class="chip">◆ {s[:52]}</span>' for s in result.key_signals
                )
                st.markdown(f'<div style="margin-bottom:12px;">{chips}</div>', unsafe_allow_html=True)
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            st.markdown('<div class="sec">Natural Language Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-box">{result.decision}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            if show_reasoning and result.reasoning:
                st.markdown('<div class="sec">Reasoning Chain</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="reason-box">{result.reasoning}</div>', unsafe_allow_html=True)
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            if xgb_signal:
                xc, xbg, xborder, _ = tier_css(xgb_signal.predicted_tier)
                st.markdown('<div class="sec">XGBoost Signal</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="stats">
                  <div class="stat" style="background:{xbg};border-color:{xborder};">
                    <div class="stat-v" style="color:{xc};font-size:15px;">{xgb_signal.predicted_tier}</div>
                    <div class="stat-l">ML Tier</div>
                  </div>
                  <div class="stat">
                    <div class="stat-v" style="font-size:15px;">{xgb_signal.probability:.1%}</div>
                    <div class="stat-l">Default Prob</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

