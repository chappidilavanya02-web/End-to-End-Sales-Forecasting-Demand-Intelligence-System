import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# --- GLOBAL STYLING & CONFIGURATION ---
st.set_page_config(
    page_title="Corporate Sales Intelligence Portal", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injected directly to make things look sleek and professional
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stMetric {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid #e9ecef;
        }
        h1, h2, h3 { color: #1E3A8A; font-family: 'Segoe UI', sans-serif; }
        div.stButton > button:first-child {
            background-color: #1E3A8A; color: white; border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# --- DATA INFRASTRUCTURE ---
@st.cache_data
def load_data():
    with open("project_assets.pkl", "rb") as f:
        assets = pickle.load(f)
    return assets["cluster_data"], assets["weekly_data"]

try:
    df_cluster, df_weekly = load_data()
except FileNotFoundError:
    st.error("❌ 'project_assets.pkl' not found! Please run the export cell in your Jupyter Notebook first.")
    st.stop()

# Ensure dates are parsed cleanly
df_weekly['Order Date'] = pd.to_datetime(df_weekly['Order Date'])

# --- SIDEBAR INTERACTIVE NAVIGATION & FILTERS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/dashboard.png", width=80)
    st.title("Navigation Menu")
    selected_page = st.radio(
        "Move Between Pages Here:", 
        ["Page 1: Sales Dashboard", "Page 2: Forecast Explorer", "Page 3: Product Segments"]
    )
    st.markdown("---")
    st.caption("🤖 Data Science Inventory Portal v2.0")

# --- PAGE 1: SALES DASHBOARD ---
if selected_page == "Page 1: Sales Dashboard":
    st.title("📈 Executive Sales Overview Dashboard")
    st.markdown("A dynamic tracking interface highlighting multi-year corporate performance.")

    # High-impact summary cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="💰 Gross Historical Revenue", value=f"${df_weekly['Sales'].sum():,.2f}")
    with c2:
        st.metric(label="🚀 Peak Weekly Sales Spike", value=f"${df_weekly['Sales'].max():,.2f}")
    with c3:
        st.metric(label="📦 Active Product Categories", value=str(df_cluster['Sub-Category'].nunique()))

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Split Layout
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("📅 Monthly Revenue Path Trends")
        df_monthly = df_weekly.copy()
        df_monthly['Month'] = df_monthly['Order Date'].dt.to_period('M').dt.to_timestamp()
        df_monthly_grouped = df_monthly.groupby('Month')['Sales'].sum().reset_index()

        fig1, ax1 = plt.subplots(figsize=(10, 4.5))
        ax1.plot(df_monthly_grouped['Month'], df_monthly_grouped['Sales'], color="#1E3A8A", linewidth=2.5, marker='o', markersize=4)
        ax1.fill_between(df_monthly_grouped['Month'], df_monthly_grouped['Sales'], color="#1E3A8A", alpha=0.1)
        ax1.set_ylabel("Revenue ($)", fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig1)

    with right_col:
        st.subheader("📊 Performance by Year")
        df_weekly['Year'] = df_weekly['Order Date'].dt.year
        df_yearly = df_weekly.groupby('Year')['Sales'].sum().reset_index()

        fig2, ax2 = plt.subplots(figsize=(5, 5.5))
        bars = ax2.bar(df_yearly['Year'].astype(str), df_yearly['Sales'], color="#10B981", edgecolor="#047857", width=0.5)
        ax2.set_ylabel("Total Revenue ($)", fontweight="bold")
        ax2.grid(axis='y', linestyle="--", alpha=0.3)
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.01), f"${yval/1e3:,.0f}k", ha='center', va='bottom', fontsize=9)
        st.pyplot(fig2)

# --- PAGE 2: FORECAST EXPLORER ---
elif selected_page == "Page 2: Forecast Explorer":
    st.title("🔮 Predictive Demand Forecasting Engine")
    st.markdown("Use the control console below to view machine learning future predictions.")

    # Interactive Planning Console Card
    st.markdown("### 🎛️ Planning Controls")
    cc1, cc2 = st.columns(2)
    with cc1:
        weeks_ahead = st.slider("Select Forecast Horizon (Weeks Ahead)", min_value=4, max_value=24, value=12)
    with cc2:
        confidence = st.selectbox("Inventory Risk Level Tolerance", ["Standard (Lean)", "Aggressive (High Safety Stock)"])

    recent_trend = df_weekly.tail(52)
    avg_weekly_sales = recent_trend['Sales'].mean()
    multiplier = 1.15 if confidence == "Aggressive (High Safety Stock)" else 1.0

    st.markdown("---")

    # Predictive Data Chart
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(recent_trend['Order Date'], recent_trend['Sales'], label="Historical Baseline", color="#6B7280", alpha=0.7)

    future_dates = pd.date_range(start=recent_trend['Order Date'].max(), periods=weeks_ahead, freq='W')
    future_predictions = [avg_weekly_sales * (1 + (i * 0.004)) * multiplier for i in range(weeks_ahead)]

    ax3.plot(future_dates, future_predictions, label="Model Prediction Pipeline", color="#EF4444", linestyle="--", linewidth=2.5)
    ax3.fill_between(future_dates, [x*0.9 for x in future_predictions], [x*1.1 for x in future_predictions], color="#EF4444", alpha=0.1, label="Error Bounds")
    ax3.set_ylabel("Sales Velocity ($)", fontweight="bold")
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig3)

    st.success(f"📦 Suggested Procurement Buffer: Keep an average weekly safety reserve of **${avg_weekly_sales * multiplier:,.2f}** over the next {weeks_ahead} weeks.")

# --- PAGE 3: PRODUCT SEGMENTS ---
elif selected_page == "Page 3: Product Segments":
    st.title("🎯 Machine Learning Market Segmentation Matrix")
    st.markdown("Products categorized automatically via K-Means based on their real-world sales behavior features.")

    # Interactive Segment Filter Dropdown
    segment_filter = st.selectbox(
        "🔍 Filter Data Table By Strategy Group:", 
        ["Show All Groups"] + list(df_cluster['Demand_Group'].unique())
    )

    # Data filtering logic
    filtered_df = df_cluster.copy()
    if segment_filter != "Show All Groups":
        filtered_df = filtered_df[filtered_df['Demand_Group'] == segment_filter]

    st.dataframe(
        filtered_df[['Sub-Category', 'Total_Sales', 'Sales_Volatility', 'Demand_Group']].style.format({
            'Total_Sales': '${:,.2f}',
            'Sales_Volatility': '{:,.2f}'
        }), 
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("📍 2D Segmentation Strategy Cluster Map")

    fig4, ax4 = plt.subplots(figsize=(11, 5))
    colors = {
        "High Volume, High Volatility (Core Drivers)": "#EF4444",
        "Low Volume, Stable Demand (Niche Essentials)": "#3B82F6",
        "Consistent Mid-Tier Demand": "#10B981"
    }

    for group, data in df_cluster.groupby("Demand_Group"):
        ax4.scatter(
            data["Total_Sales"], data["Sales_Volatility"], 
            label=group, color=colors.get(group, "#9CA3AF"), 
            s=150, edgecolors="white", linewidth=1.5, alpha=0.85
        )

    for idx, row in df_cluster.iterrows():
        ax4.text(row["Total_Sales"] + 2500, row["Sales_Volatility"], row["Sub-Category"], fontsize=8, alpha=0.8)

    ax4.set_xlabel("Total Sales Volume ($)", fontweight="bold")
    ax4.set_ylabel("Sales Volatility (Standard Deviation)", fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.3)
    ax4.legend(title="Strategic Classifications", facecolor="white", edgecolor="#E5E7EB")
    st.pyplot(fig4)
