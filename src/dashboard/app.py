"""
Customer Segmentation Dashboard
Author: Pratik
Date: 2024

Run with:  streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import json

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card h2 { margin: 0; }
    .metric-card p  { margin: 4px 0 0 0; opacity: 0.85; }
    .segment-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85em;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load all processed data files."""
    data = {}

    segments_path = PROJECT_ROOT / "data/processed/customer_segments_advanced.csv"
    if segments_path.exists():
        data["customers"] = pd.read_csv(segments_path)
        data["source"] = "advanced"
    else:
        fallback = PROJECT_ROOT / "data/processed/customer_segments.csv"
        if fallback.exists():
            data["customers"] = pd.read_csv(fallback)
            data["source"] = "basic"
        else:
            data["customers"] = None
            data["source"] = "none"

    rec_path = PROJECT_ROOT / "data/processed/segment_recommendations.csv"
    if rec_path.exists():
        data["recommendations"] = pd.read_csv(rec_path)
    else:
        data["recommendations"] = None

    summary_path = PROJECT_ROOT / "reports/advanced_clustering_summary.json"
    if not summary_path.exists():
        summary_path = PROJECT_ROOT / "reports/clustering_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data["summary"] = json.load(f)
    else:
        data["summary"] = None

    return data


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
SEGMENT_COLORS = {
    "Champions": "#2ecc71",
    "Loyal High Value": "#3498db",
    "Loyal Customers": "#3498db",
    "Potential Loyalists": "#9b59b6",
    "At Risk": "#e67e22",
    "Hibernating / Low Value": "#e74c3c",
    "Occasional Buyers": "#1abc9c",
    "Needs Attention": "#f39c12",
    "Low Value": "#95a5a6",
    "Lost/Low Value": "#e74c3c",
}


def get_segment_color(segment: str) -> str:
    return SEGMENT_COLORS.get(segment, "#34495e")


def safe_col(df: pd.DataFrame, col: str, default=None):
    """Return column if it exists, else default."""
    return df[col] if col in df.columns else default


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(data: dict):
    st.sidebar.title("🎯 Customer Segments")
    st.sidebar.markdown("---")

    if data["customers"] is None:
        st.sidebar.error("No customer data found. Run notebooks 01–06 first.")
        st.stop()

    st.sidebar.success(f"Data source: **{data['source']}** segmentation")
    st.sidebar.info(f"Total customers: **{len(data['customers']):,}**")

    segment_col = "Advanced_Segment" if "Advanced_Segment" in data["customers"].columns else "Segment"
    segments = sorted(data["customers"][segment_col].dropna().unique())

    selected = st.sidebar.multiselect(
        "Filter segments",
        options=segments,
        default=segments,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "📊 Executive Overview",
            "🔍 Segment Deep Dive",
            "💰 Revenue Analysis",
            "👤 Customer Explorer",
            "📋 Recommendations",
        ],
    )

    return segment_col, selected, page


# ---------------------------------------------------------------------------
# Page: Executive Overview
# ---------------------------------------------------------------------------
def page_executive_overview(df: pd.DataFrame, segment_col: str, selected: list):
    st.header("📊 Executive Overview")

    filtered = df[df[segment_col].isin(selected)]

    # KPI row
    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(filtered)
    total_revenue = filtered["Monetary"].sum() if "Monetary" in filtered.columns else 0
    avg_frequency = filtered["Frequency"].mean() if "Frequency" in filtered.columns else 0
    avg_recency = filtered["Recency"].mean() if "Recency" in filtered.columns else 0

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Revenue", f"£{total_revenue:,.0f}")
    col3.metric("Avg Frequency", f"{avg_frequency:.1f} orders")
    col4.metric("Avg Recency", f"{avg_recency:.0f} days")

    st.markdown("---")

    # Row 1: Segment distribution + RFM radar
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Segment Distribution")
        seg_counts = filtered[segment_col].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        fig_pie = px.pie(
            seg_counts,
            values="Count",
            names="Segment",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            hole=0.4,
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(height=420, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Segment RFM Comparison")
        rfm_cols = ["Recency", "Frequency", "Monetary"]
        available = [c for c in rfm_cols if c in filtered.columns]

        if available:
            group = filtered.groupby(segment_col)[available].mean().reset_index()

            # Normalize for radar
            norm = group.copy()
            for c in available:
                cmin, cmax = norm[c].min(), norm[c].max()
                norm[c] = (norm[c] - cmin) / (cmax - cmin) if cmax > cmin else 0.5

            # Invert recency so higher = better
            if "Recency" in norm.columns:
                norm["Recency"] = 1 - norm["Recency"]

            fig_radar = go.Figure()
            for _, row in norm.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[c] for c in available] + [row[available[0]]],
                    theta=available + [available[0]],
                    fill="toself",
                    name=row[segment_col],
                    opacity=0.6,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=420,
                margin=dict(t=30, b=20),
                showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # Row 2: RFM distributions
    st.subheader("RFM Distributions")
    d1, d2, d3 = st.columns(3)

    for col_widget, metric, color in zip(
        [d1, d2, d3],
        ["Recency", "Frequency", "Monetary"],
        ["#3498db", "#e67e22", "#2ecc71"],
    ):
        with col_widget:
            if metric in filtered.columns:
                fig = px.histogram(
                    filtered,
                    x=metric,
                    color=segment_col,
                    nbins=40,
                    color_discrete_map=SEGMENT_COLORS,
                    opacity=0.75,
                    title=metric,
                )
                fig.update_layout(
                    height=320,
                    margin=dict(t=40, b=10),
                    showlegend=False,
                    barmode="overlay",
                )
                st.plotly_chart(fig, use_container_width=True)

    # Row 3: Heatmap
    st.markdown("---")
    st.subheader("Segment Profile Heatmap")

    profile_cols = [
        "Recency", "Frequency", "Monetary",
        "Avg_Order_Value", "Unique_Products",
        "Avg_Items_Per_Order", "Customer_Lifespan_Days",
    ]
    available_profile = [c for c in profile_cols if c in filtered.columns]

    if available_profile:
        heatmap_data = filtered.groupby(segment_col)[available_profile].mean()

        # Normalize per column
        norm_heat = (heatmap_data - heatmap_data.min()) / (
            heatmap_data.max() - heatmap_data.min()
        ).replace(0, 1)

        if "Recency" in norm_heat.columns:
            norm_heat["Recency"] = 1 - norm_heat["Recency"]

        fig_heat = go.Figure(data=go.Heatmap(
            z=norm_heat.values,
            x=norm_heat.columns,
            y=norm_heat.index,
            text=heatmap_data.round(1).values,
            texttemplate="%{text}",
            colorscale="YlGnBu",
            colorbar=dict(title="Strength"),
        ))
        fig_heat.update_layout(
            height=350,
            margin=dict(t=20, b=60),
            xaxis_title="Metric",
            yaxis_title="Segment",
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Segment Deep Dive
# ---------------------------------------------------------------------------
def page_segment_deep_dive(df: pd.DataFrame, segment_col: str, selected: list):
    st.header("🔍 Segment Deep Dive")

    filtered = df[df[segment_col].isin(selected)]
    segments = sorted(filtered[segment_col].dropna().unique())

    target = st.selectbox("Select a segment to explore", segments)
    seg_df = filtered[filtered[segment_col] == target]

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", f"{len(seg_df):,}")
    if "Monetary" in seg_df.columns:
        col2.metric("Total Revenue", f"£{seg_df['Monetary'].sum():,.0f}")
        col3.metric("Avg Order Value", f"£{seg_df['Monetary'].mean():,.2f}")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"{target} — Recency vs Monetary")
        if "Recency" in seg_df.columns and "Monetary" in seg_df.columns:
            fig = px.scatter(
                seg_df,
                x="Recency",
                y="Monetary",
                color="Frequency" if "Frequency" in seg_df.columns else None,
                color_continuous_scale="Viridis",
                opacity=0.6,
                hover_data=[segment_col],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader(f"{target} — RFM Score Distribution")
        score_cols = [c for c in ["R_Score", "F_Score", "M_Score"] if c in seg_df.columns]
        if score_cols:
            score_melted = seg_df[score_cols].melt(var_name="Metric", value_name="Score")
            fig = px.box(
                score_melted,
                x="Metric",
                y="Score",
                color="Metric",
                points="outliers",
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader(f"{target} — Summary Statistics")

    stat_cols = [
        "Recency", "Frequency", "Monetary",
        "Avg_Order_Value", "Unique_Products",
        "Avg_Items_Per_Order", "Customer_Lifespan_Days",
        "Purchase_Frequency",
    ]
    available = [c for c in stat_cols if c in seg_df.columns]
    if available:
        st.dataframe(
            seg_df[available].describe().T.style.format("{:.2f}"),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Page: Revenue Analysis
# ---------------------------------------------------------------------------
def page_revenue_analysis(df: pd.DataFrame, segment_col: str, selected: list):
    st.header("💰 Revenue Analysis")

    filtered = df[df[segment_col].isin(selected)]

    if "Monetary" not in filtered.columns:
        st.warning("Monetary column not found.")
        return

    seg_rev = filtered.groupby(segment_col).agg(
        Customers=("Monetary", "count"),
        Total_Revenue=("Monetary", "sum"),
        Avg_Revenue=("Monetary", "mean"),
        Median_Revenue=("Monetary", "median"),
    ).reset_index()

    seg_rev["Revenue_Share_%"] = (
        seg_rev["Total_Revenue"] / seg_rev["Total_Revenue"].sum() * 100
    ).round(1)

    seg_rev = seg_rev.sort_values("Total_Revenue", ascending=False)

    st.dataframe(seg_rev.style.format({
        "Total_Revenue": "£{:,.0f}",
        "Avg_Revenue": "£{:,.2f}",
        "Median_Revenue": "£{:,.2f}",
        "Revenue_Share_%": "{:.1f}%",
    }), use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Revenue Share by Segment")
        fig = px.bar(
            seg_rev,
            x=segment_col,
            y="Revenue_Share_%",
            color=segment_col,
            color_discrete_map=SEGMENT_COLORS,
            text="Revenue_Share_%",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Customer Count vs Revenue")
        fig = px.scatter(
            seg_rev,
            x="Customers",
            y="Total_Revenue",
            size="Total_Revenue",
            color=segment_col,
            color_discrete_map=SEGMENT_COLORS,
            text=segment_col,
            size_max=60,
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Revenue Distribution by Segment")
    fig_violin = px.violin(
        filtered,
        x=segment_col,
        y="Monetary",
        color=segment_col,
        color_discrete_map=SEGMENT_COLORS,
        box=True,
        points="outliers",
    )
    fig_violin.update_layout(height=400, showlegend=False, yaxis_title="Monetary (£)")
    st.plotly_chart(fig_violin, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Customer Explorer
# ---------------------------------------------------------------------------
def page_customer_explorer(df: pd.DataFrame, segment_col: str, selected: list):
    st.header("👤 Customer Explorer")

    filtered = df[df[segment_col].isin(selected)]

    c1, c2 = st.columns([1, 2])

    with c1:
        search_id = st.text_input("Search Customer ID", "")
        n_rows = st.slider("Rows to display", 10, 200, 50)

    display_cols = [
        "Customer ID", segment_col,
        "Recency", "Frequency", "Monetary",
        "Avg_Order_Value", "Unique_Products",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    if search_id:
        result = filtered[
            filtered["Customer ID"].astype(str).str.contains(search_id, case=False, na=False)
        ]
    else:
        result = filtered

    with c2:
        st.dataframe(
            result[display_cols].head(n_rows).style.format({
                "Monetary": "£{:,.2f}",
                "Avg_Order_Value": "£{:,.2f}",
            }),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Top 20 Customers by Revenue")

    if "Monetary" in filtered.columns:
        top = filtered.nlargest(20, "Monetary")[display_cols]
        st.dataframe(
            top.style.format({
                "Monetary": "£{:,.2f}",
                "Avg_Order_Value": "£{:,.2f}",
            }),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Page: Recommendations
# ---------------------------------------------------------------------------
def page_recommendations(data: dict, segment_col: str, selected: list):
    st.header("📋 Segment Recommendations")

    recs = data.get("recommendations")

    if recs is None or recs.empty:
        st.info("No recommendation data found. Run notebook 06 first.")
        return

    recs_filtered = recs[recs["Segment"].isin(selected)]

    for _, row in recs_filtered.iterrows():
        segment = row["Segment"]
        color = get_segment_color(segment)

        st.markdown(f"""
        <div style="border-left: 5px solid {color}; padding: 12px 18px; 
                    background: #f8f9fa; border-radius: 6px; margin-bottom: 12px;">
            <h3 style="margin:0; color:{color};">{segment}</h3>
            <p><b>Customers:</b> {int(row['Customers']):,} &nbsp;|&nbsp; 
               <b>Share:</b> {row['Customer_Share_%']:.1f}% &nbsp;|&nbsp; 
               <b>Revenue:</b> {row.get('Revenue_Share_%', 0):.1f}%</p>
            <p><b>Profile:</b> {row['Profile']}</p>
            <p><b>Goal:</b> {row['Business_Goal']}</p>
            <p><b>Actions:</b> {row['Recommended_Actions']}</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data = load_data()
    segment_col, selected, page = render_sidebar(data)

    if data["customers"] is None:
        return

    df = data["customers"]

    if page == "📊 Executive Overview":
        page_executive_overview(df, segment_col, selected)
    elif page == "🔍 Segment Deep Dive":
        page_segment_deep_dive(df, segment_col, selected)
    elif page == "💰 Revenue Analysis":
        page_revenue_analysis(df, segment_col, selected)
    elif page == "👤 Customer Explorer":
        page_customer_explorer(df, segment_col, selected)
    elif page == "📋 Recommendations":
        page_recommendations(data, segment_col, selected)


if __name__ == "__main__":
    main()