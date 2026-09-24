"""
90-Day Revenue Management Model — Extended Edition
Best Western Plus Alexandria / Fort Belvoir (Property Code 47093)
Model Window: October 1 – December 29, 2026

NEW in this version:
- Pace & Booking Curve tab (pickup vs STLY simulation)
- Competitive Rate Index (comp set positioning)
- Group Displacement Calculator
- Real demand signal integration guide (Arrivalist, Cirium, pytrends)
- October 2026 calendar with holidays pre-loaded
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar

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
START_DATE = datetime(2026, 10, 1)   # October 2026
DAYS = 90
BASE_ADR_WEEKDAY = 109
BASE_ADR_WEEKEND = 139
BASE_OCC = 0.68

# Day-of-week multipliers (Mon=0)
DOW_MULT = {0: 1.05, 1: 1.10, 2: 1.08, 3: 0.95,
            4: 0.85, 5: 1.15, 6: 0.90}

# October–December 2026 Events & Holidays
EVENTS = {
    (2026, 10, 12): 1.25,   # Columbus Day / Indigenous Peoples' Day
    (2026, 10, 31): 1.20,   # Halloween
    (2026, 11, 11): 1.30,   # Veterans Day (Fort Belvoir demand)
    (2026, 11, 26): 1.45,   # Thanksgiving
    (2026, 11, 27): 1.50,   # Black Friday weekend
    (2026, 12, 24): 1.35,   # Christmas Eve
    (2026, 12, 25): 1.40,   # Christmas Day
    (2026, 12, 31): 1.45,   # New Year's Eve
}

# Competitive set (simulated rates — replace with real comp data)
COMP_SET = {
    "Hampton Inn Alexandria": 125,
    "Courtyard Alexandria": 145,
    "Best Western Plus (This Hotel)": None,
    "Sleep Inn & Suites": 99,
    "Holiday Inn Express": 119,
}

# -------------------------------------------------------------------
# DEMAND SIGNALS
# -------------------------------------------------------------------
@st.cache_data
def build_demand_signals():
    """Simulated demand signals. Replace with real API feeds."""
    dates = [START_DATE + timedelta(days=i) for i in range(DAYS)]
    rng = np.random.default_rng(42)

    # Airline: DCA + IAD combined arrivals index (0-100)
    airline = 60 + 25 * np.sin(np.linspace(0, 3 * np.pi, DAYS)) + rng.normal(0, 6, DAYS)
    airline = np.clip(airline, 20, 100)

    # Visitor index: tourism/leisure demand
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
    merged = df.merge(signals, on="Date", how="left")

    for col in ["Airline_Arrivals_Idx", "Visitor_Index", "Search_Interest_Idx"]:
        merged[col + "_n"] = (merged[col] - merged[col].min()) / \
                             (merged[col].max() - merged[col].min() + 1e-9)

    merged["Demand_Score"] = (
        w_airline * merged["Airline_Arrivals_Idx_n"] +
        w_visitor * merged["Visitor_Index_n"] +
        w_search  * merged["Search_Interest_Idx_n"]
    )

    merged["Pressure"] = (
        0.55 * merged["Occupancy"] +
        0.30 * merged["Demand_Score"] +
        0.15 * (merged["Event"] == "⚡").astype(float)
    )

    lift = (merged["Pressure"] - 0.65) * 0.6 * aggressiveness
    merged["AI_ADR"] = (merged["ADR"] * (1 + lift)).round(0)

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
# PACE & BOOKING CURVE SIMULATION
# -------------------------------------------------------------------
def build_pace_curve(df):
    """Simulate booking pickup vs STLY for pace analysis."""
    rng = np.random.default_rng(7)
    pace = df[["Date", "Occupancy"]].copy()

    # Simulate pickup at 90/60/30/14/7/0 days out
    horizons = [90, 60, 30, 14, 7, 0]
    for h in horizons:
        # Earlier days = lower pickup, closer = higher
        fill_pct = 1 - (h / 100) * 0.8
        noise = rng.normal(0, 0.05, len(pace))
        pace[f"Pickup_{h}d"] = (pace["Occupancy"] * fill_pct + noise).clip(0, 1)

    # STLY (same time last year) = pace * 0.95 (slightly softer)
    pace["STLY_Occ"] = (pace["Occupancy"] * 0.95).clip(0, 1)

    return pace

# -------------------------------------------------------------------
# COMPETITIVE RATE INDEX
# -------------------------------------------------------------------
def build_comp_index(df):
    """Compare this hotel's rate to comp set."""
    comp = df[["Date", "ADR", "Occupancy"]].copy()

    # Simulate comp set rates with slight variation
    rng = np.random.default_rng(99)
    for name, base in COMP_SET.items():
        if base is not None:
            variation = rng.normal(0, 8, len(comp))
            comp[f"Comp_{name}"] = (base + variation).round(0)

    # Comp set average (excluding this hotel)
    comp_cols = [c for c in comp.columns if c.startswith("Comp_")]
    comp["Comp_Set_Avg"] = comp[comp_cols].mean(axis=1).round(0)
    comp["Rate_Position"] = (comp["ADR"] - comp["Comp_Set_Avg"]) / comp["Comp_Set_Avg"] * 100

    return comp

# -------------------------------------------------------------------
# GROUP DISPLACEMENT CALCULATOR
# -------------------------------------------------------------------
def calc_displacement(group_rooms, group_adr, nights, dates_list, forecast_df):
    """
    Simplified displacement analysis.
    Returns accept/reject recommendation with math breakdown.
    """
    total_group_rev = group_rooms * group_adr * nights

    # Average forecasted ADR on those dates
    date_forecast = forecast_df[forecast_df["Date"].isin(dates_list)]
    if len(date_forecast) == 0:
        return None

    avg_transient_adr = date_forecast["ADR"].mean()
    avg_occ = date_forecast["Occupancy"].mean()

    # Displaced rooms = min(group_rooms, rooms that would have sold)
    available_transient_rooms = (1 - avg_occ) * TOTAL_ROOMS
    displaced_rooms = max(0, group_rooms - available_transient_rooms)
    displaced_rev = displaced_rooms * avg_transient_adr * nights

    # Net contribution (simplified — no F&B for select-service)
    variable_cost_per_room = 25  # housekeeping + amenities
    group_cost = group_rooms * nights * variable_cost_per_room
    net = total_group_rev - displaced_rev - group_cost

    # Breakeven ADR
    breakeven_adr = (displaced_rev + group_cost) / (group_rooms * nights) if group_rooms * nights > 0 else 0

    return {
        "total_group_rev": total_group_rev,
        "displaced_rev": displaced_rev,
        "group_cost": group_cost,
        "net_contribution": net,
        "breakeven_adr": round(breakeven_adr, 0),
        "accept": net > 0 and group_adr >= breakeven_adr,
        "avg_transient_adr": round(avg_transient_adr, 0),
        "avg_occ": round(avg_occ, 3),
    }

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Best_Western_logo.svg/320px-Best_Western_logo.svg.png",
    width=160
)
st.sidebar.title("Revenue Controls")

st.sidebar.markdown("### 📤 Upload 90-Day Data")
st.sidebar.caption("CSV or Excel. Columns: Date, Occupancy, ADR, Rooms_Sold (optional).")

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

rate_adjust = st.sidebar.slider("Global Rate Adjustment (%)",
                                min_value=-20, max_value=30, value=0, step=1)

use_ai = st.sidebar.toggle("🤖 Enable AI-Assisted Rates", value=True)
ai_aggressiveness = st.sidebar.slider("AI Aggressiveness", 0.5, 1.5, 1.0, 0.1)

st.sidebar.markdown("### 📈 Demand Signal Weights")
w_airline = st.sidebar.slider("Airline Arrivals", 0.0, 1.0, 0.4, 0.05)
w_visitor = st.sidebar.slider("Visitor Index", 0.0, 1.0, 0.35, 0.05)
w_search  = st.sidebar.slider("Search Interest", 0.0, 1.0, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption(f"Model window: {START_DATE.strftime('%b %d')} – "
                   f"{(START_DATE + timedelta(days=DAYS-1)).strftime('%b %d, %Y')}")

# -------------------------------------------------------------------
# LOAD & ENRICH
# -------------------------------------------------------------------
df = build_model()
signals = build_demand_signals()

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

final_adr = ai_df["AI_ADR"] if use_ai else ai_df["ADR"]
ai_df["Final_ADR"] = (final_adr * (1 + rate_adjust / 100)).round(0)

elasticity = -0.8
ai_df["Final_Occupancy"] = (
    ai_df["Occupancy"] * (1 + elasticity * (ai_df["Final_ADR"] / ai_df["ADR"] - 1))
).clip(0.3, 1.0)
ai_df["Final_Rooms"] = (ai_df["Final_Occupancy"] * TOTAL_ROOMS).astype(int)
ai_df["Final_Revenue"] = ai_df["Final_Rooms"] * ai_df["Final_ADR"]
ai_df["Final_RevPAR"] = (ai_df["Final_Revenue"] / TOTAL_ROOMS).round(2)

pace_df = build_pace_curve(ai_df)
comp_df = build_comp_index(ai_df)

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
st.subheader("90-Day Revenue Model — October 2026 | Upload · Calendar · AI · Pace · Comp · Displacement")

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📅 October Calendar",
    "🤖 AI Rate Recommender",
    "✈️ Demand Signals",
    "📈 Pace & Booking Curve",
    "🏆 Competitive Rate Index",
    "🎯 Group Displacement",
    "📊 Scenario Planner",
    "🔌 Real Data Integration",
])

# -------------------------------------------------------------------
# TAB 1: OCTOBER CALENDAR
# -------------------------------------------------------------------
with tab1:
    st.markdown("### October–December 2026 Occupancy Calendar")

    cal_df = ai_df.copy()
    cal_df["Week"] = cal_df["Date"].dt.isocalendar().week
    cal_df["DOW"] = cal_df["Date"].dt.weekday

    pivot = cal_df.pivot_table(index="Week", columns="DOW",
                               values="Final_Occupancy", aggfunc="mean")
    pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = px.imshow(pivot, color_continuous_scale="RdYlGn", aspect="auto",
                    labels=dict(color="Occupancy"), text_auto=".0%")
    fig.update_layout(height=550, title="Occupancy Heatmap — Oct 2026 Onward")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Daily Rate Table (October 2026)")
    oct_df = ai_df[ai_df["Date"].dt.month == 10]
    cal_table = oct_df[["DateStr", "Day", "Final_Occupancy", "Final_ADR",
                        "Final_RevPAR", "Event"]].rename(columns={
        "DateStr": "Date", "Final_Occupancy": "Occ",
        "Final_ADR": "ADR", "Final_RevPAR": "RevPAR"
    })
    st.dataframe(
        cal_table.style.format({"Occ": "{:.1%}", "ADR": "${:.0f}",
                                "RevPAR": "${:.2f}"}).background_gradient(
            subset=["Occ"], cmap="RdYlGn"),
        height=420, use_container_width=True
    )

# -------------------------------------------------------------------
# TAB 2: AI RATE RECOMMENDER
# -------------------------------------------------------------------
with tab2:
    st.markdown("### 🤖 AI-Assisted Rate Recommendations")
    st.caption("AI blends occupancy, DOW patterns, demand signals, and events.")

    show_cols = ["DateStr", "Day", "Occupancy", "ADR", "AI_ADR",
                 "Final_ADR", "AI_Reason", "Event"]
    ai_view = ai_df[show_cols].rename(columns={
        "DateStr": "Date", "AI_ADR": "AI Rate",
        "Final_ADR": "Final Rate", "AI_Reason": "AI Reasoning"
    })
    st.dataframe(ai_view.style.format({
        "Occupancy": "{:.1%}", "ADR": "${:.0f}",
        "AI Rate": "${:.0f}", "Final Rate": "${:.0f}"
    }).background_gradient(subset=["AI Rate"], cmap="Reds"),
        height=500, use_container_width=True)

    lift = (ai_df["AI_ADR"] - ai_df["ADR"])
    fig_lift = px.histogram(lift, nbins=25, title="AI Rate Lift vs Base ADR ($)",
                            color_discrete_sequence=["#2E86AB"])
    fig_lift.update_layout(xaxis_title="Lift ($)", height=350)
    st.plotly_chart(fig_lift, use_container_width=True)

# -------------------------------------------------------------------
# TAB 3: DEMAND SIGNALS
# -------------------------------------------------------------------
with tab3:
    st.markdown("### ✈️ Demand Signals — Airline, Visitors, Search")

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=signals["Date"], y=signals["Airline_Arrivals_Idx"],
        name="Airline Arrivals (DCA+IAD)", line=dict(color="#2E86AB", width=3)))
    fig_sig.add_trace(go.Scatter(x=signals["Date"], y=signals["Visitor_Index"],
        name="Visitor Index", line=dict(color="#E94F37", width=3)))
    fig_sig.add_trace(go.Scatter(x=signals["Date"], y=signals["Search_Interest_Idx"],
        name="Search Interest", line=dict(color="#F6AE2D", width=2, dash="dash")))
    fig_sig.update_layout(title="External Demand Signals (Indexed 0–100)",
                          yaxis_title="Index", height=420, hovermode="x unified")
    st.plotly_chart(fig_sig, use_container_width=True)

    weights = pd.DataFrame({
        "Signal": ["Airline Arrivals", "Visitor Index", "Search Interest"],
        "Weight": [w_airline, w_visitor, w_search]
    })
    fig_w = px.pie(weights, names="Signal", values="Weight", hole=0.5,
                   title="Composite Demand Score Composition")
    st.plotly_chart(fig_w, use_container_width=True)

    st.dataframe(signals, height=300, use_container_width=True)

# -------------------------------------------------------------------
# TAB 4: PACE & BOOKING CURVE
# -------------------------------------------------------------------
with tab4:
    st.markdown("### 📈 Pace & Booking Curve — Pickup vs STLY")
    st.caption("Shows how bookings accumulate over time vs same time last year.")

    # Select a sample date to show pace
    sample_idx = st.selectbox("Select a stay date to inspect pace:",
                              range(0, DAYS, 7),
                              format_func=lambda i: pace_df.iloc[i]["Date"].strftime("%b %d"))

    horizons = [90, 60, 30, 14, 7, 0]
    row = pace_df.iloc[sample_idx]
    pickup_vals = [row[f"Pickup_{h}d"] for h in horizons]

    fig_pace = go.Figure()
    fig_pace.add_trace(go.Scatter(x=horizons, y=pickup_vals,
        name="This Year Pickup", mode="lines+markers",
        line=dict(color="#2E86AB", width=3), marker=dict(size=10)))
    fig_pace.add_trace(go.Scatter(x=horizons, y=[row["STLY_Occ"]] * len(horizons),
        name="STLY Final Occ", mode="lines",
        line=dict(color="#E94F37", width=2, dash="dash")))
    fig_pace.update_layout(
        title=f"Booking Curve: {row['Date'].strftime('%b %d, %Y')}",
        xaxis_title="Days Before Arrival", yaxis_title="Occupancy",
        yaxis_tickformat=".0%", height=400, hovermode="x unified"
    )
    st.plotly_chart(fig_pace, use_container_width=True)

    # Pace table
    pace_view = pace_df[["Date", "Occupancy", "Pickup_90d", "Pickup_60d",
                         "Pickup_30d", "Pickup_14d", "Pickup_7d", "STLY_Occ"]].copy()
    pace_view["Date"] = pace_view["Date"].dt.strftime("%b %d")
    pace_view["Pace_vs_STLY"] = (pace_view["Occupancy"] - pace_view["STLY_Occ"]) * 100
    st.dataframe(pace_view.style.format({
        "Occupancy": "{:.1%}", "Pickup_90d": "{:.1%}", "Pickup_60d": "{:.1%}",
        "Pickup_30d": "{:.1%}", "Pickup_14d": "{:.1%}", "Pickup_7d": "{:.1%}",
        "STLY_Occ": "{:.1%}", "Pace_vs_STLY": "{:+.1f}pts"
    }).background_gradient(subset=["Pace_vs_STLY"], cmap="RdYlGn"),
        height=400, use_container_width=True)

    st.info("**How to use:** If pace is ahead of STLY, hold rate. If behind, consider targeted promotions.")

# -------------------------------------------------------------------
# TAB 5: COMPETITIVE RATE INDEX
# -------------------------------------------------------------------
with tab5:
    st.markdown("### 🏆 Competitive Rate Index")
    st.caption("Your rate position vs the Alexandria/Fort Belvoir comp set.")

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=comp_df["Date"], y=comp_df["ADR"],
        name="This Hotel", line=dict(color="#2E86AB", width=3)))
    fig_comp.add_trace(go.Scatter(x=comp_df["Date"], y=comp_df["Comp_Set_Avg"],
        name="Comp Set Avg", line=dict(color="#E94F37", width=2, dash="dash")))
    fig_comp.update_layout(title="Rate Position vs Comp Set",
                           yaxis_title="ADR ($)", height=400, hovermode="x unified")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Rate position distribution
    fig_pos = px.histogram(comp_df["Rate_Position"], nbins=30,
        title="Rate Position vs Comp Set (%)",
        color_discrete_sequence=["#2E86AB"])
    fig_pos.update_layout(xaxis_title="% Above/Below Comp Set", height=350)
    st.plotly_chart(fig_pos, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Rate Position", f"{comp_df['Rate_Position'].mean():+.1f}%")
    col2.metric("Days Above Comp", f"{(comp_df['Rate_Position'] > 0).sum()}")
    col3.metric("Days Below Comp", f"{(comp_df['Rate_Position'] < 0).sum()}")

    st.dataframe(comp_df.head(20).style.format({
        "ADR": "${:.0f}", "Comp_Set_Avg": "${:.0f}", "Rate_Position": "{:+.1f}%"
    }), height=300, use_container_width=True)

# -------------------------------------------------------------------
# TAB 6: GROUP DISPLACEMENT
# -------------------------------------------------------------------
with tab6:
    st.markdown("### 🎯 Group Displacement Calculator")
    st.caption("Evaluate group requests: accept, reject, or counter-offer.")

    col1, col2, col3 = st.columns(3)
    group_rooms = col1.number_input("Group Rooms Requested", 5, 100, 30)
    group_adr = col2.number_input("Group ADR ($)", 50, 300, 99)
    nights = col3.number_input("Nights", 1, 7, 2)

    st.markdown("**Select group dates:**")
    date_options = [d.strftime("%b %d") for d in ai_df["Date"].head(30)]
    selected_dates = st.multiselect("Group stay dates", date_options,
                                    default=date_options[10:10+nights])

    if st.button("Calculate Displacement") and len(selected_dates) > 0:
        selected_date_objs = [datetime.strptime(f"2026-{d}", "%Y-%b %d")
                              for d in selected_dates]
        result = calc_displacement(group_rooms, group_adr, len(selected_dates),
                                   selected_date_objs, ai_df)

        if result:
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Group Revenue", f"${result['total_group_rev']:,.0f}")
            c2.metric("Displaced Revenue", f"${result['displaced_rev']:,.0f}")
            c3.metric("Net Contribution", f"${result['net_contribution']:,.0f}")
            c4.metric("Breakeven ADR", f"${result['breakeven_adr']}")

            if result["accept"]:
                st.success(f"✅ ACCEPT — Net contribution positive and group ADR "
                           f"(${group_adr}) exceeds breakeven (${result['breakeven_adr']}).")
            elif group_adr < result["breakeven_adr"]:
                st.warning(f"⚠️ COUNTER — Group ADR (${group_adr}) is below breakeven "
                           f"(${result['breakeven_adr']}). Counter at ${result['breakeven_adr']}+.")
            else:
                st.error("❌ REJECT — Displacement cost exceeds group value.")

            st.markdown("**Assumptions:**")
            st.caption(f"Avg transient ADR on those dates: ${result['avg_transient_adr']} | "
                       f"Avg occupancy: {result['avg_occ']:.1%} | "
                       f"Variable cost/room: $25")

# -------------------------------------------------------------------
# TAB 7: SCENARIO PLANNER
# -------------------------------------------------------------------
with tab7:
    st.markdown("### Rate Scenario Planner")
    c1, c2, c3 = st.columns(3)
    c1.metric("Base Revenue", f"${base_rev:,.0f}")
    c2.metric("Final Revenue", f"${total_rev:,.0f}",
              delta=f"{total_rev - base_rev:+,.0f}")
    c3.metric("Revenue Change", f"{rev_pct:+.1f}%")

    fig_comp2 = go.Figure()
    fig_comp2.add_trace(go.Scatter(x=df["Date"], y=df["Revenue"].cumsum(),
        name="Base", fill="tozeroy", line=dict(color="#6c757d", dash="dash")))
    fig_comp2.add_trace(go.Scatter(x=ai_df["Date"], y=ai_df["Final_Revenue"].cumsum(),
        name="Final (AI + Adjustment)", fill="tozeroy",
        line=dict(color="#2E86AB", width=3)))
    fig_comp2.update_layout(title="Cumulative Revenue: Base vs Final",
                            height=420, hovermode="x unified")
    st.plotly_chart(fig_comp2, use_container_width=True)

# -------------------------------------------------------------------
# TAB 8: REAL DATA INTEGRATION
# -------------------------------------------------------------------
with tab8:
    st.markdown("### 🔌 Real Data Integration Guide")
    st.caption("Replace simulated signals with live feeds for production use.")

    st.markdown("""
    #### 1. Airline Arrivals (DCA / IAD)
    **Source:** Cirium, OAG, or FAA ASPM
    - Cirium publishes airline schedules with passenger volumes [citation:2]
    - MWAA reports DCA domestic activity +4.5% and IAD international trends [citation:12]
    - **API:** Cirium offers Schedules API; OAG has Connections API

    #### 2. Visitor / Tourism Index
    **Source:** Arrivalist
    - Arrivalist provides "Trip Volume Arrivals by Day, Week, Month and Quarter" [citation:1]
    - Mobile geo-location panel of 120M devices, balanced to U.S. population [citation:6]
    - **API:** Arrivalist Air Intelligence Hub for airport-specific insights [citation:6]

    #### 3. Search Interest
    **Source:** Google Trends
    - Use `pytrends` library to pull Alexandria/Fort Belvoir search volume
    - Compare "hotel near Fort Belvoir" vs "Alexandria hotel" queries
    - **Python:** `pip install pytrends`

    #### 4. Convention / Event Calendar
    **Source:** Alexandria CVB or Visit Alexandria
    - Pull forward calendar with event names, dates, attendance estimates [citation:11]
    - Treat attendance as a prior — update as actual pickup comes in

    #### 5. Block Pickup
    **Source:** Your PMS / Passkey
    - Daily block pickup by group is the highest-value input [citation:11]
    - Feed directly into pace curve for real-time displacement analysis

    #### 6. Competitive Rates
    **Source:** OTA scraping or rate intelligence platforms
    - Monitor Hampton, Courtyard, Holiday Inn Express Alexandria
    - Use rate position to sit "one rung above the market" on compression [citation:11]
    """)

    st.info("**Next step:** Contact Arrivalist (info@arrivalist.com) and Cirium for API access. "
            "For Google Trends, install `pytrends` and add to requirements.")

# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Model: Best Western Plus Alexandria/Fort Belvoir | 132 rooms | October 2026 | "
    "Elasticity -0.8 | AI = rule-based demand blender | "
    "Comp set and demand signals are simulated. Replace with live feeds for production."
)
