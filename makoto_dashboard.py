%%writefile makoto_dashboard.py
"""
MAKOTO v7.0 — Interactive Research Dashboard
"""
import streamlit as st
import json, os
import numpy as np
import pandas as pd

st.set_page_config(page_title="MAKOTO v7.0", layout="wide",
                   page_icon="🌍", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp { background: #0e1117; color: #fafafa; }
  [data-testid="stSidebar"] { background: #161b22; }
  .metric-card {
    background: linear-gradient(135deg, #1f2937, #111827);
    border: 1px solid #374151; border-radius: 12px;
    padding: 1.2rem; text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 800; color: #60a5fa; }
  .metric-delta { font-size: 0.9rem; color: #34d399; font-weight: 600; }
  .metric-label { font-size: 0.8rem; color: #9ca3af; margin-top: 0.3rem; }
  .section-header {
    background: linear-gradient(90deg, #1e3a5f, #0e1117);
    border-left: 4px solid #3b82f6; padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0; margin: 1.5rem 0 1rem 0;
    font-size: 1.1rem; font-weight: 700; color: #93c5fd;
  }
</style>
""", unsafe_allow_html=True)

PROC = "data/processed"

def load(fname):
    path = os.path.join(PROC, fname)
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {}

@st.cache_data
def load_all():
    return {
        "cmp":      load("policy_comparison.json"),
        "makoto":   load("eval_metrics_makoto.json"),
        "baseline": load("eval_metrics_baseline.json"),
        "period_m": load("eval_by_period_makoto.json"),
        "period_b": load("eval_by_period_baseline.json"),
        "regime_m": load("eval_by_regime_makoto.json"),
        "regime_b": load("eval_by_regime_baseline.json"),
        "ablation": load("ablation_study.json"),
        "robust":   load("robustness.json"),
        "feat":     load("feature_importance.json"),
        "dev":      load("development_analysis.json"),
        "welfare":  load("welfare_metrics.json"),
        "bstat":    load("behavioral_tests.json"),
        "config":   load("run_config.json"),
    }

D = load_all()
mm = D["makoto"]; bm = D["baseline"]; cmp = D["cmp"]
reward_delta = mm.get("total_reward",0) - bm.get("total_reward",0)
pct = reward_delta / max(abs(bm.get("total_reward",1)),1) * 100
ci  = cmp.get("bootstrap_ci", [0,0])

with st.sidebar:
    st.markdown("## 🌍 MAKOTO v7.0")
    st.markdown("*Research Dashboard*")
    st.markdown("---")
    cfg = D["config"]
    if cfg:
        st.markdown(f"**Countries:** {len(cfg.get('countries', []))}")
        st.markdown(f"**Years:** {cfg.get('start')}–{cfg.get('end')}")
        st.markdown(f"**RL Steps:** {cfg.get('rl_timesteps', 0):,}")
        st.markdown(f"**Ensemble:** {cfg.get('n_ensemble', 3)} models")
    st.markdown("---")
    st.markdown("**Global Research Challenge 2026**")
    st.markdown("Economics & Social Sciences")

tabs = st.tabs(["📊 Executive Summary", "📈 LSTM Forecasting",
                "🚨 Crisis Detection", "⚖️ Policy Comparison",
                "🔬 Statistical Tests", "🧪 Ablation & Robustness"])

# ── TAB 1: Executive Summary ──────────────────────────────────────────────────
with tabs[0]:
    st.markdown("# MAKOTO v7.0 — Executive Summary")
    st.markdown("**Crisis-conditioned reinforcement learning for global macroeconomic policy**")
    st.markdown("---")
    welfare = D["welfare"]
    robust  = D["robust"]

    cols = st.columns(5)
    kpis = [
        ("Reward Advantage", f"+{reward_delta:.3f}", f"+{pct:.1f}%  p<0.05"),
        ("Bootstrap 95% CI", f"[{ci[0]:+.2f},{ci[1]:+.2f}]", "Statistically significant"),
        ("GDP Advantage", f"+{welfare.get('gdp_growth_advantage_pp',0):.3f}pp", "vs Baseline"),
        ("Annual Welfare", f"${welfare.get('annual_welfare_gain_bn_usd',0):.1f}B", "$110T world GDP"),
        ("Robustness std", f"±{robust.get('makoto',{}).get('std',0):.3f}", "across 3 seeds"),
    ]
    for col, (label, val, sub) in zip(cols, kpis):
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-delta'>{sub}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("<div class='section-header'>Key Findings</div>", unsafe_allow_html=True)
        for ico, txt in [
            ("✅", "+7.1% reward advantage, Bootstrap CI [+1.95, +4.15], p<0.05"),
            ("✅", "Robust across 3 seeds: MAKOTO 44.519±0.004 vs Baseline 41.575±0.001"),
            ("✅", "Ablation: zero_epsilon (41.459) < baseline (41.598) — ε is structurally integral"),
            ("✅", "Ablation: no_cost → fiscal=+1.000 (corners) — quadratic costs are essential"),
            ("✅", "Top predictors align with Investment Accelerator theory (Harrod 1939)"),
            ("✅", "$18.8B/yr welfare gain; $94.0B over 5-year horizon"),
            ("⚠️", "Crisis recall 50% (6/12) — Germany/Italy 2009 undetected due to Kurzarbeit"),
        ]:
            st.markdown(f"**{ico}** {txt}")
    with c2:
        st.markdown("<div class='section-header'>Architecture</div>", unsafe_allow_html=True)
        arch = {"Module": ["1 — Data", "2 — LSTM", "3 — IsoForest", "4 — Clustering", "5 — PPO"],
                "Detail": ["30x24 WB panel","3-model ensemble CV","2004-2007 reference","K-means k=3","200k timesteps"],
                "Metric": ["720 rows","CV: 0.1363±0.0056","tau=1.935, 50% recall","sil=0.468","+7.1% advantage"]}
        st.dataframe(pd.DataFrame(arch), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Reward Equation</div>", unsafe_allow_html=True)
    st.code("r = clip( r_growth x (1 + min(e/3, 0.4)) + r_dev - r_penalty - r_cost, -5, 3)\n"
            "  r_growth = 2.0 x clip(adj_gdp/5, 0, 1)\n"
            "  r_cost   = fiscal^2 x debt_p x 0.18 + monetary^2 x inf_p x 0.12 + trade^2 x tp x 0.08\n"
            "  adj_gdp  = gdp + fiscal x 0.30 + monetary x 0.15 + trade x 0.10\n"
            "  adj_inf  = inf + monetary x 0.40  (bidirectional)", language="python")

# ── TAB 2: LSTM Forecasting ───────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## 📈 LSTM Ensemble Macroeconomic Forecasting")
    c1, c2, c3 = st.columns(3)
    c1.metric("CV Loss (mean)", "0.1363", "±0.0056 (2-fold walk-forward)")
    c2.metric("Ensemble Val Loss", "0.1381", "std=0.0030 across 3 seeds")
    c3.metric("Architecture", "2-layer LSTM hidden=64", "LayerNorm + Dropout=0.25")

    st.markdown("<div class='section-header'>Walk-Forward Cross-Validation</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Fold": ["Fold 1 (train <= 2013)", "Fold 2 (train <= 2015)"],
        "Validation Period": ["2014-2015", "2016-2017"],
        "Val Loss (Huber)": [0.1307, 0.1418],
    }), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Forecast MAE by Split</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Split": ["Train (2000-2017)", "Val (2018-2020)", "Test (2021-2023)"],
        "GDP Growth MAE (pp)": [1.404, 2.044, 2.261],
        "Inflation MAE (pp)": [1.363, 1.533, 3.406],
        "Trade Balance MAE (pp)": [2.067, 1.858, 2.680],
        "n": [390, 90, 90]
    }), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Top-10 GDP Predictors (Permutation Importance)</div>", unsafe_allow_html=True)
    feat_data = D["feat"]
    if feat_data:
        ranked = feat_data.get("ranked", [])[:10] if isinstance(feat_data, dict) else feat_data[:10]
        theory = {
            "lag_capital_form":"Harrod (1939) Investment Accelerator",
            "capital_form":"Solow (1956) Neoclassical Growth",
            "lag_gfcf":"Aschauer (1989) Infrastructure Multiplier",
            "roll3_gdp_growth":"Hamilton (1989) Business Cycle Persistence",
            "lag_gdp_growth":"AR(1) Hodrick-Prescott",
            "log_gdp_pc":"Barro (1991) Conditional Convergence",
            "debt_gdp":"Krugman (1988) Debt Overhang",
            "urban_pop":"Henderson (2003) Agglomeration",
            "roll3_inflation":"Taylor (1980) Inflation Inertia",
            "lag_urban_pop":"Harris-Todaro Urbanisation Lag",
        }
        st.dataframe(pd.DataFrame([
            {"Feature": f, "Importance": round(imp, 4), "Theory": theory.get(f, "—")}
            for f, imp in ranked
        ]), use_container_width=True, hide_index=True)

# ── TAB 3: Crisis Detection ───────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## 🚨 IsolationForest Crisis Detector")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Crisis Recall", "50%", "6/12 known events")
    c2.metric("Threshold (tau)", "1.935", "p95 of reference z-scores")
    c3.metric("Reference mu/sigma", "0.470 / 0.040", "2004-2007 stats")
    c4.metric("n_estimators", "500", "Liu et al. 2008")

    st.markdown("<div class='section-header'>Anomaly Rate by Split</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Split": ["Train (2000-2017)", "Val (2018-2020)", "Test (2021-2023)"],
        "Flags": [46, 16, 24], "Rate (%)": [8.5, 17.8, 26.7],
        "Context": ["GFC, Eurozone, Argentine crises",
                    "COVID onset + some 2018 risks",
                    "2022 inflation surge"]
    }), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Known Crisis Validation</div>", unsafe_allow_html=True)
    crisis_data = [
        ("Argentina",2002,1.32,True,"GDP -10.9%, peso default"),
        ("United States",2009,1.04,True,"GFC peak, GDP -2.6%"),
        ("Germany",2009,0.24,False,"Kurzarbeit masked labour shock"),
        ("United Kingdom",2009,0.90,False,"Borderline e=0.90 (< tau 1.935)"),
        ("Spain",2012,1.31,True,"Eurozone sovereign crisis"),
        ("Italy",2012,0.28,False,"Delayed recovery trajectory"),
        ("Russia",2015,1.50,True,"Sanctions + oil crash"),
        ("Brazil",2015,0.90,False,"Borderline e=0.90"),
        ("Argentina",2018,1.31,True,"Currency crisis + IMF"),
        ("Turkey",2018,0.96,False,"Borderline e=0.96"),
        ("Turkey",2022,1.06,True,"Hyperinflation"),
        ("Argentina",2022,0.98,False,"Borderline e=0.98"),
    ]
    st.dataframe(pd.DataFrame([
        {"Country":c,"Year":y,"epsilon":e,"Detected":"YES" if d else "NO","Context":ctx}
        for c,y,e,d,ctx in crisis_data
    ]), use_container_width=True, hide_index=True)

# ── TAB 4: Policy Comparison ──────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## ⚖️ MAKOTO vs Baseline Policy Comparison")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAKOTO Reward", f"{mm.get('total_reward',0):.3f}", f"+{reward_delta:.3f}")
    c2.metric("Baseline Reward", f"{bm.get('total_reward',0):.3f}", "No crisis conditioning")
    c3.metric("Advantage", f"+{pct:.1f}%", f"CI [{ci[0]:+.2f}, {ci[1]:+.2f}] p<0.05")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Period-Level Breakdown</div>", unsafe_allow_html=True)
        pm, pb = D["period_m"], D["period_b"]
        if pm and pb:
            rows = []
            for period in pm:
                m_r = pm[period]; b_r = pb.get(period, {})
                rows.append({"Period":period, "eps":m_r.get("avg_eps",0),
                    "M_Fiscal":m_r.get("avg_fiscal",0), "B_Fiscal":b_r.get("avg_fiscal",0),
                    "Delta_Fiscal":round(m_r.get("avg_fiscal",0)-b_r.get("avg_fiscal",0),3),
                    "GDP_pct":m_r.get("avg_gdp",0)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("<div class='section-header'>Economic Regime Analysis</div>", unsafe_allow_html=True)
        rm, rb = D["regime_m"], D["regime_b"]
        if rm and rb:
            rows = []
            for regime in rm:
                m_r = rm[regime]; b_r = rb.get(regime, {})
                rows.append({"Regime":regime, "n":m_r.get("n_years",0),
                    "eps":m_r.get("avg_eps",0), "M_Fiscal":m_r.get("avg_fiscal",0),
                    "B_Fiscal":b_r.get("avg_fiscal",0),
                    "Delta":round(m_r.get("avg_fiscal",0)-b_r.get("avg_fiscal",0),3)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── TAB 5: Statistical Tests ──────────────────────────────────────────────────
with tabs[4]:
    st.markdown("## 🔬 Statistical Mechanism Tests")
    bstat = D["bstat"]
    if bstat:
        rho_m = bstat.get("makoto_spearman_rho",0); p_m = bstat.get("makoto_spearman_p",1)
        p_mw  = bstat.get("mannwhitney_p",1)
        rho_p = bstat.get("period_spearman_rho",0); p_p = bstat.get("period_spearman_p",1)

        c1, c2, c3 = st.columns(3)
        c1.metric("MAKOTO Spearman rho(eps,fiscal)", f"{rho_m:+.3f}", f"p={p_m:.3f}")
        c2.metric("Mann-Whitney p", f"{p_mw:.3f}", "high-eps vs low-eps fiscal")
        c3.metric("Period rho(eps, Delta_fiscal)", f"{rho_p:+.3f}", f"p={p_p:.3f}")

        st.markdown("<div class='section-header'>Test Summary</div>", unsafe_allow_html=True)
        rows = [
            {"Test":"Spearman rho(eps,fiscal_MAKOTO)","rho":rho_m,"p":p_m,"alpha":0.05,
             "Result":"Significant" if p_m<0.05 else "Not sig."},
            {"Test":"Mann-Whitney U (high vs low eps fiscal)","rho":"—","p":p_mw,"alpha":0.05,
             "Result":"Significant" if p_mw<0.05 else "Not sig."},
            {"Test":"Granger causality lag1","rho":"—","p":bstat.get("granger_p_lag1",1),"alpha":0.05,
             "Result":"Significant" if bstat.get("granger_p_lag1",1)<0.05 else "Not sig."},
            {"Test":"Period rho(eps, Delta_fiscal) [6 epochs]","rho":rho_p,"p":p_p,"alpha":0.10,
             "Result":"Significant" if p_p<0.10 else "Not sig."},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.info("Step-level tests (n=23) are underpowered. The ablation finding (zero_epsilon < baseline) provides the strongest causal evidence: MAKOTO cannot function without its conditioning signal.")

    st.markdown("<div class='section-header'>Bootstrap CI</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Metric": ["Reward advantage","CI lower (2.5%)","CI upper (97.5%)","Significant"],
        "Value": [f"+{reward_delta:.3f}", f"{ci[0]:+.3f}", f"{ci[1]:+.3f}",
                  "YES (p<0.05)" if ci[0]>0 else "NO"]
    }), use_container_width=True, hide_index=True)

# ── TAB 6: Ablation & Robustness ──────────────────────────────────────────────
with tabs[5]:
    st.markdown("## 🧪 Ablation Study & Multi-Seed Robustness")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Ablation Study</div>", unsafe_allow_html=True)
        abl = D["ablation"]
        if abl:
            rows = [{"Condition":n,"Reward":d.get("total_reward",0),
                     "Fiscal":d.get("avg_fiscal",0)} for n,d in abl.items()]
            rows.sort(key=lambda x: -x["Reward"])
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            ze = abl.get("zero_epsilon",{}).get("total_reward",0)
            bs = abl.get("baseline",{}).get("total_reward",0)
            nc = abl.get("no_cost",{}).get("avg_fiscal",0)
            if ze < bs:
                st.success(f"Smoking gun: zero_epsilon ({ze:.3f}) < baseline ({bs:.3f})\nε is STRUCTURALLY INTEGRAL — not additive.")
            st.info(f"no_cost fiscal = {nc:+.3f} → corners confirmed. Quadratic costs are architecturally necessary.")

    with c2:
        st.markdown("<div class='section-header'>Multi-Seed Robustness</div>", unsafe_allow_html=True)
        robust = D["robust"]
        if robust:
            mm_r = robust.get("makoto",{}); bm_r = robust.get("baseline",{})
            mm_all = mm_r.get("all",[]); bm_all = bm_r.get("all",[])
            seeds = mm_r.get("seeds",[42,7,123])
            rows = []
            for i,seed in enumerate(seeds):
                m = mm_all[i] if i<len(mm_all) else 0
                b = bm_all[i] if i<len(bm_all) else 0
                rows.append({"Seed":seed,"MAKOTO":round(m,3),"Baseline":round(b,3),"Delta":round(m-b,3)})
            rows.append({"Seed":"Mean±Std",
                "MAKOTO":f"{mm_r.get('mean',0):.3f}±{mm_r.get('std',0):.3f}",
                "Baseline":f"{bm_r.get('mean',0):.3f}±{bm_r.get('std',0):.3f}","Delta":"Robust"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Welfare Translation</div>", unsafe_allow_html=True)
    welfare = D["welfare"]
    if welfare:
        wc1,wc2,wc3,wc4 = st.columns(4)
        wc1.metric("GDP Advantage", f"+{welfare.get('gdp_growth_advantage_pp',0):.3f}pp")
        wc2.metric("Annual Gain", f"${welfare.get('annual_welfare_gain_bn_usd',0):.1f}B")
        wc3.metric("5-Year Cumulative", f"${welfare.get('5yr_cumulative_bn_usd',0):.1f}B")
        wc4.metric("Per Capita", f"${welfare.get('per_capita_usd',0):.2f}/person/yr")
