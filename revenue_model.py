"""
90-Day Revenue Management Model — Extended
Best Western Plus Alexandria / Fort Belvoir (Property Code 47093)

Features:
- Left sidebar: upload 90 days of data (CSV/Excel)
- Calendar heatmap view
- AI-assisted rate recommendations with reasoning
- Demand signals: airline arrivals (DCA/IAD) + visitor/event index
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import io

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="BW+ Alexandria RevMgt",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

TOTAL_ROOMS = 132
START_DATE = datetime(2026, 9, 1)
DAYS = 90
BASE_ADR_WEEKDAY = 109
BASE_ADR_WEEKEND = 139
BASE_OCC = 0.68

DOW_MULT = {0: 1.05, 1: 1.10, 2: 1.08, 3: 0.95,
            4: 0.85, 5: 1.15, 6: 0.90}

EVENTS = {
    (2026, 10, 10): 1.35,
    (2026, 10, 11): 1.30,
    (2026, 11, 11): 1.25,
    (2026, 11, 26): 1.40,
    (2026, 11, 27): 1.45,
}

# -------------------------------------------------------------------
# DEMAND SIGNALS (simulated airline + visitor index)
# Replace with real API pulls (e.g., FAA, Cirium, STR, Arrivalist)
# -------------------------------------------------------------------
@st.cache_data
def build_demand_signals():
    dates = [START_DATE + timedelta(days=i) for i in range(DAYS)]
    rng = np.random.default_rng(42)

    # Airline arrivals (DCA + IAD combined, index 0–100)
    airline = 60 + 25 * np.sin(np.linspace(0, 3 * np.pi, DAYS)) + rng.normal(0, 6, DAYS)
    airline = np.clip(airline, 20, 100)

    # Visitor / tourism index
    visitor = 55 + 20 * np.cos(np.linspace(0, 2 * np.pi, DAYS)) + rng.normal(0, 5, DAYS)
    visitor = np.clip(visitor, 20, 100)

    # Search interest (Google Trends-style)
    search = 50 + 15 * np.sin(np.linspace(0, 4 * np.pi, DAYS)) + rng.normal(0, 7, DAYS)
    search = np.clip(search, 15, 100)

    return pd.DataFrame({
        "Date": dates,
        "Airline_Arrivals_Idx": airline.round(1),
        "Visitor_Index": visitor.round(1),
        "Search_Interest_Idx": search.round(1),
    })

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Best_Western_logo.svg/320px-Best_Western_logo.svg.png",
    width=160
)
st.sidebar.title("Revenue Controls")

# --- Data Upload ---
st.sidebar.markdown("### 📤 Upload 90-Day Data")
st.sidebar.caption("CSV or Excel. Expected columns: Date, Occupancy, ADR, Rooms_Sold (optional).")

uploaded = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])

uploaded_df = None
if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            uploaded_df = pd.read_csv(uploaded)
        else:
            uploaded_df = pd.read_excel(uploaded)
        uploaded_df["Date"] = pd.to_datetime(uploaded_df["Date"])
        st.sidebar.success(f"Loaded {len(uploaded_df)} rows.")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")

st.sidebar.markdown("---")

# --- Rate adjustment ---
rate_adjust = st.sidebar.slider(
    "Global Rate Adjustment (%)",
    min_value=-20, max_value=30, value=0, step=1
)

# --- AI assist toggle ---
use_ai = st.sidebar.toggle("🤖 Enable AI-Assisted Rates", value=True)
ai_aggressiveness = st.sidebar.slider(
    "AI Aggressiveness", 0.5, 1.5, 1.0, 0.1,
    help="Multiplier on AI-recommended rate lift."
)

# --- Demand signal weights ---
st.sidebar.markdown("### 📈 Demand Signal Weights")
w_airline = st.sidebar.slider("Airline Arrivals", 0.0, 1.0, 0.4, 0.05)
w_visitor = st.sidebar.slider("Visitor Index", 0.0, 1.0, 0.35, 0.05)
w_search  = st.sidebar.slider("Search Interest", 0.0, 1.0, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption(f"Model window: {START_DATE.strftime('%b %d')} – "
                   f"{(START_DATE + timedelta(days=DAYS-1)).strftime('%b %d, %Y')}")

# -------------------------------------------------------------------
# BASE MODEL
# -------------------------------------------------------------------
@st.cache_data
def build_model():
    records = []
    for i in range(DAYS):
        date = START_DATE + timedelta(days=i)
        dow = date.weekday()
        base_occ = BASE_OCC * DOW_MULT.get(dow, 1.0)
        event_key = (date.year, date.month, date.day)
        event_mult = EVENTS.get(event_key, 1.0)
        forecast_occ = min(base_occ * event_mult, 0.98)

        if dow == 5:
            base_rate = BASE_ADR_WEEKEND
        elif dow in (4, 6):
            base_rate = BASE_ADR_WEEKEND * 0.9
        else:
            base_rate = BASE_ADR_WEEKDAY

        if forecast_occ > 0.85:
            base_rate *= 1.25
        elif forecast_occ > 0.75:
            base_rate *= 1.10
        base_rate *= event_mult

        adr = round(base_rate)
        rooms_sold = int(TOTAL_ROOMS * forecast_occ)
        revenue = rooms_sold * adr
        revpar = round(revenue / TOTAL_ROOMS, 2)

        records.append({
            "Date": date,
            "Day": date.strftime("%a"),
            "Month": date.strftime("%b"),
            "Occupancy": forecast_occ,
            "Rooms Sold": rooms_sold,
            "ADR": adr,
            "RevPAR": revpar,
            "Revenue": revenue,
            "Event": "⚡" if event_mult > 1.0 else "",
        })
    df = pd.DataFrame(records)
    df["DateStr"] = df["Date"].dt.strftime("%b %d")
    return df

# -------------------------------------------------------------------
# AI-ASSISTED RATE ENGINE
# -------------------------------------------------------------------
def ai_recommend(df, signals, w_airline, w_visitor, w_search, aggressiveness):
    """
    Rule-based 'AI' recommender. Blends:
      - base demand (occupancy forecast)
      - day-of-week pattern
      - demand signals (airline, visitor, search)
      - event compression
    Produces recommended ADR + reasoning.
    """
    merged = df.merge(signals, on="Date", how="left")

    # Normalize signals to 0–1
    for col in ["Airline_Arrivals_Idx", "Visitor_Index", "Search_Interest_Idx"]:
        merged[col + "_n"] = (merged[col] - merged[col].min()) / \
                             (merged[col].max() - merged[col].min() + 1e-9)

    # Composite demand score (0–1)
    merged["Demand_Score"] = (
        w_airline * merged["Airline_Arrivals_Idx_n"] +
        w_visitor * merged["Visitor_Index_n"] +
        w_search  * merged["Search_Interest_Idx_n"]
    )

    # Combined pressure = occupancy + demand signal + event
    merged["Pressure"] = (
        0.55 * merged["Occupancy"] +
        0.30 * merged["Demand_Score"] +
        0.15 * (merged["Event"] == "⚡").astype(float)
    )

    # Recommended rate: base ADR scaled by pressure vs. 0.65 baseline
    lift = (merged["Pressure"] - 0.65) * 0.6 * aggressiveness
    merged["AI_ADR"] = (merged["ADR"] * (1 + lift)).round(0)

    # Reasoning string
    def reason(row):
        parts = []
        if row["Event"] == "⚡":
            parts.append("event compression")
        if row["Airline_Arrivals_Idx"] > 75:
            parts.append("high airline arrivals")
        if row["Visitor_Index"] > 70:
            parts.append("strong visitor demand")
        if row["Search_Interest_Idx"] > 70:
            parts.append("elevated search interest")
        if row["Day"] in ("Tue", "Sat"):
            parts.append("peak DOW")
        if not parts:
            parts.append("baseline demand")
        return ", ".join(parts).capitalize()

    merged["AI_Reason"] = merged.apply(reason, axis=1)
    return merged

# -------------------------------------------------------------------
# LOAD & ENRICH
# -------------------------------------------------------------------
df = build_model()
signals = build_demand_signals()

# If user uploaded data, merge to override occupancy/ADR
if uploaded_df is not None:
    df = df.merge(
        uploaded_df[["Date"] + [c for c in ["Occupancy", "ADR", "Rooms_Sold"]
                                if c in uploaded_df.columns]],
        on="Date", how="left", suffixes=("", "_up")
    )
    if "Occupancy_up" in df:
        df["Occupancy"] = df["Occupancy_up"].fillna(df["Occupancy"])
    if "ADR_up" in df:
        df["ADR"] = df["ADR_up"].fillna(df["ADR"])
    df["Rooms Sold"] = (df["Occupancy"] * TOTAL_ROOMS).astype(int)
    df["Revenue"] = df["Rooms Sold"] * df["ADR"]
    df["RevPAR"] = (df["Revenue"] / TOTAL_ROOMS).round(2)

ai_df = ai_recommend(df, signals, w_airline, w_visitor, w_search, ai_aggressiveness)

# Apply global rate adjustment + AI
final_adr = ai_df["AI_ADR"] if use_ai else ai_df["ADR"]
ai_df["Final_ADR"] = (final_adr * (1 + rate_adjust / 100)).round(0)

elasticity = -0.8
ai_df["Final_Occupancy"] = (
    ai_df["Occupancy"] * (1 + elasticity * (ai_df["Final_ADR"] / ai_df["ADR"] - 1))
).clip(0.3, 1.0)
ai_df["Final_Rooms"] = (ai_df["Final_Occupancy"] * TOTAL_ROOMS).astype(int)
ai_df["Final_Revenue"] = ai_df["Final_Rooms"] * ai_df["Final_ADR"]
ai_df["Final_RevPAR"] = (ai_df["Final_Revenue"] / TOTAL_ROOMS).round(2)

# KPIs
total_rev = ai_df["Final_Revenue"].sum()
base_rev = df["Revenue"].sum()
avg_adr = ai_df["Final_ADR"].mean()
avg_occ = ai_df["Final_Occupancy"].mean()
avg_revpar = ai_df["Final_RevPAR"].mean()
rev_pct = (total_rev - base_rev) / base_rev * 100

# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
st.title("🏨 Best Western Plus Alexandria / Fort Belvoir")
st.subheader("90-Day Revenue Model — Upload · Calendar · AI Rates · Demand Signals")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"${total_rev:,.0f}", delta=f"{rev_pct:+.1f}%")
k2.metric("Avg ADR", f"${avg_adr:,.0f}")
k3.metric("Avg Occupancy", f"{avg_occ:.1%}")
k4.metric("Avg RevPAR", f"${avg_revpar:,.2f}")
k5.metric("Rooms Sold", f"{ai_df['Final_Rooms'].sum():,}")

st.markdown("---")

# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Calendar View",
    "🤖 AI Rate Recommender",
    "✈️ Demand Signals",
    "📈 Demand & Rate Curve",
    "🎯 Scenario Planner",
])

# -------------------------------------------------------------------
# TAB 1: CALENDAR VIEW
# -------------------------------------------------------------------
with tab1:
    st.markdown("### 90-Day Occupancy Calendar")

    cal_df = ai_df.copy()
    cal_df["Week"] = cal_df["Date"].dt.isocalendar().week
    cal_df["DOW"] = cal_df["Date"].dt.weekday

    # Pivot for heatmap
    pivot = cal_df.pivot_table(
        index="Week", columns="DOW", values="Final_Occupancy", aggfunc="mean"
    )
    pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        labels=dict(color="Occupancy"),
        text_auto=".0%"
    )
    fig.update_layout(height=550, title="Occupancy Heatmap by Week")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Calendar Rate Table")
    cal_table = ai_df[["DateStr", "Day", "Final_Occupancy", "Final_ADR",
                       "Final_RevPAR", "Event"]].rename(columns={
        "DateStr": "Date", "Final_Occupancy": "Occ",
        "Final_ADR": "ADR", "Final_RevPAR": "RevPAR"
    })
    st.dataframe(
        cal_table.style.format({
            "Occ": "{:.1%}", "ADR": "${:.0f}", "RevPAR": "${:.2f}"
        }).background_gradient(subset=["Occ"], cmap="RdYlGn"),
        height=420, use_container_width=True
    )

# -------------------------------------------------------------------
# TAB 2: AI RATE RECOMMENDER
# -------------------------------------------------------------------
with tab2:
    st.markdown("### 🤖 AI-Assisted Rate Recommendations")
    st.caption(
        "AI blends occupancy forecast, day-of-week patterns, demand signals "
        "(airline, visitor, search), and event compression into a recommended ADR."
    )

    show_cols = ["DateStr", "Day", "Occupancy", "ADR", "AI_ADR",
                 "Final_ADR", "AI_Reason", "Event"]
    ai_view = ai_df[show_cols].rename(columns={
        "DateStr": "Date", "AI_ADR": "AI Rate",
        "Final_ADR": "Final Rate", "AI_Reason": "AI Reasoning"
    })

    st.dataframe(
        ai_view.style.format({
            "Occupancy": "{:.1%}", "ADR": "${:.0f}",
            "AI Rate": "${:.0f}", "Final Rate": "${:.0f}"
        }).background_gradient(subset=["AI Rate"], cmap="Reds"),
        height=500, use_container_width=True
    )

    st.markdown("### AI Lift Distribution")
    lift = (ai_df["AI_ADR"] - ai_df["ADR"])
    fig_lift = px.histogram(lift, nbins=25, title="AI Rate Lift vs Base ADR ($)",
                            color_discrete_sequence=["#2E86AB"])
    fig_lift.update_layout(xaxis_title="Lift ($)", height=350)
    st.plotly_chart(fig_lift, use_container_width=True)

    st.info(
        "**How to use:** Review the AI Reasoning column. If AI lifts a date "
        "you consider low-risk, accept it. If AI over-lifts a soft date, "
        "override manually in your RMS or reduce AI Aggressiveness in the sidebar."
    )

# -------------------------------------------------------------------
# TAB 3: DEMAND SIGNALS
# -------------------------------------------------------------------
with tab3:
    st.markdown("### ✈️ Demand Signals — Airline, Visitors, Search")

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(
        x=signals["Date"], y=signals["Airline_Arrivals_Idx"],
        name="Airline Arrivals (DCA+IAD)", line=dict(color="#2E86AB", width=3)
    ))
    fig_sig.add_trace(go.Scatter(
        x=signals["Date"], y=signals["Visitor_Index"],
        name="Visitor Index", line=dict(color="#E94F37", width=3)
    ))
    fig_sig.add_trace(go.Scatter(
        x=signals["Date"], y=signals["Search_Interest_Idx"],
        name="Search Interest", line=dict(color="#F6AE2D", width=2, dash="dash")
    ))
    fig_sig.update_layout(
        title="External Demand Signals (Indexed 0–100)",
        yaxis_title="Index", height=420, hovermode="x unified"
    )
    st.plotly_chart(fig_sig, use_container_width=True)

    st.markdown("### Signal Weights (from sidebar)")
    weights = pd.DataFrame({
        "Signal": ["Airline Arrivals", "Visitor Index", "Search Interest"],
        "Weight": [w_airline, w_visitor, w_search]
    })
    fig_w = px.pie(weights, names="Signal", values="Weight", hole=0.5,
                   title="Composite Demand Score Composition")
    st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("### Signal Data (Raw)")
    st.dataframe(signals, height=300, use_container_width=True)

# -------------------------------------------------------------------
# TAB 4: DEMAND & RATE CURVE
# -------------------------------------------------------------------
with tab4:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ai_df["Date"], y=ai_df["Final_Occupancy"],
        name="Occupancy", marker_color="#2E86AB", opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=ai_df["Date"], y=ai_df["Final_ADR"],
        name="Final ADR", mode="lines+markers",
        marker=dict(size=5, color="#E94F37"), line=dict(width=3),
        yaxis="y2"
    ))
    fig.add_trace(go.Scatter(
        x=ai_df["Date"], y=ai_df["AI_ADR"],
        name="AI ADR", mode="lines", line=dict(width=2, dash="dot",
        color="#F6AE2D"), yaxis="y2"
    ))
    fig.update_layout(
        title="Occupancy vs Final ADR vs AI ADR",
        yaxis=dict(title="Occupancy", tickformat=".0%", range=[0, 1]),
        yaxis2=dict(title="ADR ($)", overlaying="y", side="right"),
        height=450, hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# TAB 5: SCENARIO PLANNER
# -------------------------------------------------------------------
with tab5:
    st.markdown("### Rate Scenario Planner")
    st.caption(f"Current: **{rate_adjust:+d}%** global adjustment, "
               f"AI {'ON' if use_ai else 'OFF'} (aggressiveness {ai_aggressiveness}).")

    c1, c2, c3 = st.columns(3)
    c1.metric("Base Revenue", f"${base_rev:,.0f}")
    c2.metric("Final Revenue", f"${total_rev:,.0f}",
              delta=f"{total_rev - base_rev:+,.0f}")
    c3.metric("Revenue Change", f"{rev_pct:+.1f}%")

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=df["Date"], y=df["Revenue"].cumsum(),
        name="Base", fill="tozeroy", line=dict(color="#6c757d", dash="dash")
    ))
    fig_comp.add_trace(go.Scatter(
        x=ai_df["Date"], y=ai_df["Final_Revenue"].cumsum(),
        name="Final (AI + Adjustment)", fill="tozeroy",
        line=dict(color="#2E86AB", width=3)
    ))
    fig_comp.update_layout(
        title="Cumulative Revenue: Base vs Final",
        height=420, hovermode="x unified"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Model: Best Western Plus Alexandria/Fort Belvoir | 132 rooms | "
    "Elasticity -0.8 | AI = rule-based demand blender | "
    "Signals are simulated; replace with real feeds (Cirium, Arrivalist, Google Trends)."
)
