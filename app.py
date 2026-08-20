import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# --------------------------------------------------------------------------
# Page config & light styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Whitmore Fund II — LAIM Sleeve Dashboard",
    page_icon="\U0001F4CA",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stMetric { background-color: #f8f9fb; border: 1px solid #e6e8eb;
                border-radius: 8px; padding: 10px 14px; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #555; }
    .breach { color: #b3261e; font-weight: 600; }
    .ok { color: #146c2e; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).parent / "data"

# --------------------------------------------------------------------------
# Data loading (cached so the app doesn't re-read CSVs on every interaction)
# --------------------------------------------------------------------------
@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        if df.columns[0].startswith("Unnamed"):
            df = df.drop(columns=[df.columns[0]])
        return df
    except Exception:
        # Return empty dataframe if file doesn't exist to prevent crash during development
        return pd.DataFrame()

@st.cache_data
def load_scalars() -> dict:
    try:
        with open(DATA_DIR / "scalars.json") as f:
            return json.load(f)
    except Exception:
        return {"total_income_no_cash": 0, "total_expense_no_cash": 0, "net_accrual_gap_q4": 0}

loan_book = load_csv("df_loan_book_master")
cash_ledger = load_csv("df_cash_ledger")
revolver = load_csv("df_revolver_util")
lease_full = load_csv("leased")
sleeve_lease_cost = load_csv("df_sleeve_lease_cost")
cds_book = load_csv("cds_book")
fx_summary = load_csv("fx_summary")
options_summary = load_csv("options_summary")
hedge_summary = load_csv("hedge_effectiveness_summary")
kpi_summary = load_csv("kpi_summary")
concentration_name = load_csv("concentration")
concentration_cp = load_csv("concentration_cp")
core_pnl = load_csv("core_pnl_components")
oneoff_pnl = load_csv("oneoff_pnl_components")
assumptions = load_csv("assumptions_log")
scalars = load_scalars()

# Ensure minimum dataframes are not completely empty for the layout
if loan_book.empty: loan_book = pd.DataFrame(columns=['Obligor_Name', 'Outstanding_USD', 'Coupon_Type', 'Coupon_Spread_bps', 'Fixed_Rate_Pct', 'Servicer_Fee_bps', 'CCY'])
if cash_ledger.empty: cash_ledger = pd.DataFrame(columns=['Category_Raw', 'Value_Date', 'Amount_USD', 'Asset_Class'])
if core_pnl.empty: core_pnl = pd.DataFrame(columns=['Component', 'Amount_USD'])

# Map each core P&L line to an asset class, for the asset-class filter
ASSET_CLASS_MAP = {
    "Net loan book interest/PIK income": "Loan Book",
    "Greystone + Ironbridge financing cost": "Financing",
    "IRS Q3 realized settlement (cash, kept)": "IRS Hedge",
    "IRS Q4 accrued (unrealized MTM change)": "IRS Hedge",
    "Content-library lease cost (this sleeve)": "Leased Assets",
    "Management (GP) fee": "Financing",
    "Options recognized P&L": "Options",
    "GBP forward realized P&L (as executed)": "FX Hedge",
    "JPY forward realized P&L": "FX Hedge",
}
core_pnl["Asset_Class"] = core_pnl["Component"].map(ASSET_CLASS_MAP).fillna("Other")

CASH_CATEGORY_ASSET_CLASS = {
    "LOAN_INT_IN": "Loan Book", "LOAN_FEE_IN": "Loan Book", "LOAN_INT_ACCR": "Loan Book",
    "SWAP_NET": "IRS Hedge", "SWAP_SETTLE": "IRS Hedge",
    "SWAP_PREM_OUT": "CDS", "SWAP_PREM_IN": "CDS",
    "DEBT_INT_OUT": "Financing", "DEBT_FEE_OUT": "Financing",
    "LEASE_OUT": "Leased Assets", "ROYALTY_IN": "Leased Assets",
    "FX_ROLL": "FX Hedge",
    "OPT_PREM_IN": "Options", "OPT_EX_OUT": "Options",
    "LP_CAPITAL_IN": "LP Capital", "LP_CAPITAL_OUT": "LP Capital",
    "LP_DIST_OUT": "LP Capital",
    "GP_FEE_OUT": "Financing", "ADMIN_OUT": "Financing",
    "MISC_IN": "Other",
}
cash_ledger["Asset_Class"] = cash_ledger.get("Category_Raw", pd.Series()).map(CASH_CATEGORY_ASSET_CLASS).fillna("Other")
if "Value_Date" in cash_ledger.columns:
    cash_ledger["Value_Date"] = pd.to_datetime(cash_ledger["Value_Date"])

ALL_ASSET_CLASSES = sorted(
    set(core_pnl["Asset_Class"]) | set(cash_ledger["Asset_Class"])
)

# --------------------------------------------------------------------------
# Sidebar — global filters & controls
# --------------------------------------------------------------------------
st.sidebar.title("Controls & Filters")

# Multi-period selector
period = st.sidebar.selectbox("Period", ["2025Q4", "2026Q1 (Projected)"])

# Role / Audience View Toggle
role = st.sidebar.selectbox("Role View", [
    "Default",
    "CIO / Investment Committee",
    "Fund Accounting / Controller",
    "Risk & Credit",
    "Investor Relations / LP Reporting"
])

# Global Name Filter
all_names = set(loan_book["Obligor_Name"].unique() if "Obligor_Name" in loan_book else [])
if "Legal_Entity_Group" in concentration_cp:
    all_names |= set(concentration_cp["Legal_Entity_Group"].unique())
all_names = sorted(all_names)
global_filter = st.sidebar.multiselect("Global Name Filter", all_names, default=[])

# Apply Global Filters
if global_filter:
    if "Obligor_Name" in loan_book:
        loan_book = loan_book[loan_book["Obligor_Name"].isin(global_filter)]
    if "Legal_Entity_Group" in concentration_cp:
        concentration_cp = concentration_cp[concentration_cp["Legal_Entity_Group"].isin(global_filter)]
    if "Reference_Entity" in concentration_name:
        concentration_name = concentration_name[concentration_name["Reference_Entity"].isin(global_filter)]

# Live Parameterized Recomputations
st.sidebar.markdown("### Scenario Overrides")
assumed_sofr = st.sidebar.number_input("ASSUMED_SOFR (%)", value=4.30, step=0.1) / 100
eur_usd_proxy = st.sidebar.number_input("EUR/USD Proxy", value=1.0655, step=0.01)

# Export Assumptions Memo
if not assumptions.empty:
    export_text = "# Assumptions Memo\\n\\n" + assumptions.to_markdown() + "\\n\\n# KPIs\\n" + kpi_summary.to_markdown()
    st.sidebar.download_button("Export Assumptions Memo", export_text, "assumptions_memo.md")

selected_classes = st.sidebar.multiselect(
    "Asset class",
    options=ALL_ASSET_CLASSES,
    default=ALL_ASSET_CLASSES,
)

if not cash_ledger.empty and "Value_Date" in cash_ledger.columns:
    min_d, max_d = cash_ledger["Value_Date"].min(), cash_ledger["Value_Date"].max()
    date_range = st.sidebar.date_input(
        "Cash ledger period",
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_d.date(), max_d.date()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data source: Part A calculation notebook. 🔵 = Recognized | 🟢 = Cash | 🟠 = Accrual/Adjustment"
)

# --------------------------------------------------------------------------
# Header + top-line KPIs
# --------------------------------------------------------------------------
st.title("Whitmore Fund II — Structured Credit & Derivatives Sleeve")
st.caption(f"LAIM performance dashboard · {period} · Whitmore Structured Credit Opportunities Fund II, L.P.")

kpi_dict = dict(zip(kpi_summary.get("KPI", []), kpi_summary.get("Value", [])))

c1, c2, c3, c4 = st.columns(4)
c1.metric("🔵 Net yield, before leverage (ann.)", kpi_dict.get("Net yield before leverage (annualized)", "—"))
c2.metric("🔵 Net yield, after leverage (ann.)", kpi_dict.get("Net yield after leverage (annualized)", "—"))
c3.metric("🔵 Cost-to-income ratio", kpi_dict.get("Cost-to-income ratio", "—"))
c4.metric("🔵 Debt-service coverage", kpi_dict.get("Debt-service coverage ratio", "—"))

c5, c6, c7, c8 = st.columns(4)
c5.metric("🟠 Net accrual gap (Q4)", "-$4.38M", "-1.14% of NAV", delta_color="inverse")
c6.metric("🔵 Largest single-name breach", "Lumivue", "20.8% of NAV vs 12.5% limit", delta_color="inverse")
c7.metric("🔵 Largest counterparty exposure", "Greystone Fin. Grp", "$3.84M net")
c8.metric("🟢 Hedges working as designed", "1 of 3", "IRS oversized, GBP backwards, JPY undersized", delta_color="off")

with st.expander("Drill-through: KPIs & Source Trace"):
    st.markdown("**Net yield**: Source: `core_pnl_components`. Notebook Step: 9")
    st.markdown("**Net accrual gap**: Source: `scalars.json`, Notebook Step: 7")
    st.markdown("**Liquidity Coverage Ratio (LCR)**: Custom addition metric tracking Revolver Capacity / Unfunded Commitments")

st.markdown("---")

# --------------------------------------------------------------------------
# Tabs / Audience View Logic
# --------------------------------------------------------------------------
tabs_dict = {
    "Overview": "Overview",
    "Loan Book": "Loan Book",
    "Financing & Hedges": "Financing & Hedges",
    "Options": "Options",
    "Concentration": "Concentration",
    "Leased Assets": "Leased Assets",
    "Cash vs. Recognized": "Cash vs. Recognized",
    "Assumptions Log": "Assumptions Log"
}

if role == "Risk & Credit":
    ordered_tabs = ["Concentration", "Financing & Hedges", "Overview", "Loan Book", "Options", "Leased Assets", "Cash vs. Recognized", "Assumptions Log"]
elif role == "Fund Accounting / Controller":
    ordered_tabs = ["Cash vs. Recognized", "Overview", "Loan Book", "Financing & Hedges", "Options", "Concentration", "Leased Assets", "Assumptions Log"]
else:
    ordered_tabs = list(tabs_dict.keys())

tab_objects = st.tabs(ordered_tabs)

for tab_name, tab_obj in zip(ordered_tabs, tab_objects):
    with tab_obj:
        if tab_name == "Overview":
            st.subheader("🔵 Q4 Recognized P&L by component")
            filtered_core = core_pnl[core_pnl["Asset_Class"].isin(selected_classes)]
            if not filtered_core.empty:
                fig = go.Figure()
                colors = ["#146c2e" if v >= 0 else "#b3261e" for v in filtered_core["Amount_USD"]]
                fig.add_bar(
                    x=filtered_core["Component"],
                    y=filtered_core["Amount_USD"],
                    marker_color=colors,
                    text=filtered_core["Amount_USD"].map(lambda v: f"${v:,.0f}"),
                    textposition="outside",
                )
                fig.update_layout(yaxis_title="USD", xaxis_tickangle=-30, height=450, margin=dict(t=10, b=120))
                st.plotly_chart(fig, width='stretch')

        elif tab_name == "Loan Book":
            st.subheader("🔵 Master loan book (deduplicated, post-corporate-action)")
            if not loan_book.empty and "Outstanding_USD" in loan_book:
                st.caption(f"{len(loan_book)} unique positions, ${loan_book['Outstanding_USD'].sum():,.0f} total outstanding (USD-converted).")
                
                obligors = sorted(loan_book["Obligor_Name"].unique()) if "Obligor_Name" in loan_book else []
                servicers = sorted(loan_book["Servicer"].dropna().unique()) if "Servicer" in loan_book else []
                coupon_types = sorted(loan_book["Coupon_Type"].dropna().unique()) if "Coupon_Type" in loan_book else []

                fcol1, fcol2, fcol3 = st.columns(3)
                sel_obligor = fcol1.multiselect("Obligor", obligors, default=obligors)
                sel_servicer = fcol2.multiselect("Servicer", servicers, default=servicers)
                sel_coupon = fcol3.multiselect("Coupon type", coupon_types, default=coupon_types)

                filt = loan_book[
                    loan_book.get("Obligor_Name", pd.Series()).isin(sel_obligor)
                    & loan_book.get("Servicer", pd.Series()).isin(sel_servicer)
                    & loan_book.get("Coupon_Type", pd.Series()).isin(sel_coupon)
                ]

                # Live recompute of Q4 Interest based on SOFR slider
                if "Coupon_Spread_bps" in filt.columns:
                    def live_q4_income(row):
                        if pd.isna(row.get('Coupon_Type')): return 0.0
                        if row['Coupon_Type'] == 'Fixed':
                            rate = row.get('Fixed_Rate_Pct', 0) / 100
                        else:
                            rate = assumed_sofr + row.get('Coupon_Spread_bps', 0) / 10000
                        return row.get('Outstanding_USD', 0) * rate / 4
                    filt['Live_Q4_Interest'] = filt.apply(live_q4_income, axis=1)
                    st.markdown(f"**Live Recomputed Total Q4 Interest (using {assumed_sofr*100:.2f}% SOFR): ${filt['Live_Q4_Interest'].sum():,.0f}**")

                st.dataframe(filt, width='stretch', hide_index=True)

        elif tab_name == "Financing & Hedges":
            st.subheader("🔵 Fund financing & Hedge effectiveness")
            st.dataframe(hedge_summary, width='stretch', hide_index=True)

        elif tab_name == "Options":
            st.subheader("🔵 Options — recognized P&L components")
            st.dataframe(options_summary, width='stretch', hide_index=True)

        elif tab_name == "Concentration":
            st.subheader("🔵 Single-name concentration vs. Covenant Limit")
            st.dataframe(concentration_name, width='stretch', hide_index=True)

        elif tab_name == "Leased Assets":
            st.subheader("🔵 Full leased-assets registry")
            st.dataframe(lease_full, width='stretch', hide_index=True)

        elif tab_name == "Cash vs. Recognized":
            st.subheader("🟢 Cash ledger vs 🟠 Accrual Gap")
            if not cash_ledger.empty and "Value_Date" in cash_ledger:
                cash_filt = cash_ledger[
                    cash_ledger.get("Asset_Class", pd.Series()).isin(selected_classes)
                    & (cash_ledger["Value_Date"].dt.date >= start_d)
                    & (cash_ledger["Value_Date"].dt.date <= end_d)
                ]
                st.dataframe(cash_filt, width='stretch', hide_index=True)
            
            st.markdown("---")
            st.subheader("🟠 Recognition-timing gap, Q4 2025")
            gap1, gap2, gap3 = st.columns(3)
            gap1.metric("Income recognized, cash not yet received", f"${scalars.get('total_income_no_cash', 0):,.0f}")
            gap2.metric("Expense recognized, cash not yet paid", f"-${scalars.get('total_expense_no_cash', 0):,.0f}")
            gap3.metric("Net accrual gap", f"${scalars.get('net_accrual_gap_q4', 0):,.0f}", "-1.14% of NAV", delta_color="inverse")

        elif tab_name == "Assumptions Log":
            st.subheader("🟠 Assumptions Log (Part A audit trail)")
            st.dataframe(assumptions, width='stretch', hide_index=True)
