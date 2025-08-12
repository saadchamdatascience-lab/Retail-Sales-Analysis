# streamlit_retail_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import plotly.figure_factory as ff # For heatmap

# Optional: Import Prophet for forecasting
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly  # <-- ADDED THIS IMPORT
    PROPHET_INSTALLED = True
except ImportError:
    PROPHET_INSTALLED = False

# ---- Page Configuration and Styling ----
st.set_page_config(
    page_title="Retail Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the header and KPIs
st.markdown(
    """
    <style>
    .header {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand {
        font-weight: 700;
        font-size: 24px;
        color: #2c3e50;
    }
    .kpi-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #3498db;
    }
    .small {
        color: #6b7280;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 5px;
    }
    .big {
        font-weight: 700;
        font-size: 24px;
        color: #1a237e;
    }
    </style>
    """, unsafe_allow_html=True
)

# ---- Header / Hero Section ----
st.markdown('<div class="header"><div class="brand">📈 Retail Insights & Forecasting</div></div>', unsafe_allow_html=True)
st.markdown("**Upload your sales data (CSV/Excel)** to view interactive visualizations, key metrics, and future demand forecasts.")
st.markdown("---")

# ---- Sidebar: Upload & Controls ----
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
use_sample = st.sidebar.checkbox("Use sample data instead", value=False)

# Forecasting controls
if PROPHET_INSTALLED:
    forecast_enable = st.sidebar.checkbox("Enable Forecasting", value=False)
    forecast_days = st.sidebar.number_input(
        "Forecast horizon (days)",
        min_value=7, max_value=365, value=30,
        disabled=not forecast_enable
    )
else:
    st.sidebar.warning("`prophet` library not found. Forecasting is disabled.")
    forecast_enable = False

# ---- Data Loading Helpers ----
@st.cache_data
def read_file(file):
    """Reads a CSV or Excel file and returns a pandas DataFrame."""
    try:
        if isinstance(file, str):
            # For a file path
            return pd.read_csv(file)
        # For a file upload object
        if file.name.endswith('.csv'):
            return pd.read_csv(io.BytesIO(file.read()))
        else:
            return pd.read_excel(io.BytesIO(file.read()))
    except Exception as e:
        st.error(f"Could not read dataset: {e}")
        return pd.DataFrame()

@st.cache_data
def get_sample_data():
    """Generates a realistic sample dataset with a simpler structure."""
    rng = pd.date_range(end=pd.Timestamp.today(), periods=365, freq='D')
    products = ["Beauty", "Clothing", "Electronics", "Grocery", "Home", "Tools", "Books"]
    
    data = {
        "Transaction ID": np.random.randint(100000, 999999, size=len(rng)*3),
        "Orderdate": np.tile(rng.strftime("%Y-%m-%d"), 3),
        "Customer ID": np.random.choice([f"CUST{i}" for i in range(1000, 9999)], size=len(rng)*3),
        "Gender": np.random.choice(["Male", "Female"], size=len(rng)*3),
        "Age": np.random.randint(18, 65, size=len(rng)*3),
        "Product Category": np.random.choice(products, size=len(rng)*3),
        "Quantity": np.random.poisson(4, size=len(rng)*3) + 1,
        "Price per Unit": np.random.uniform(20, 1500, size=len(rng)*3).round(2),
    }

    df = pd.DataFrame(data)
    df["Total Amount"] = df["Quantity"] * df["Price per Unit"]
    return df

# ---- Data Loading Logic ----
if uploaded_file and not use_sample:
    df = read_file(uploaded_file)
elif use_sample:
    df = get_sample_data()
else:
    df = pd.DataFrame()
    st.info("Please upload a file or enable 'Use sample data' in the sidebar to get started.")

if df.empty:
    st.stop()

# ---- Data Cleaning and Preprocessing ----
df.columns = [c.strip() for c in df.columns]

# Auto-detect date and sales columns
date_col_candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "timestamp", "orderdate"])]
sales_col_candidates = [c for c in df.columns if any(k in c.lower() for k in ["total", "amount", "sales", "revenue"])]

date_col = st.sidebar.selectbox("Select Date Column", options=date_col_candidates, index=0)
sales_col = st.sidebar.selectbox("Select Sales Column", options=sales_col_candidates, index=0)
product_col = st.sidebar.selectbox("Select Product Column", options=['None'] + df.columns.tolist())

# Process selected columns
if date_col and sales_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    df = df.dropna(subset=[date_col, sales_col])
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = initial_rows - len(df)

    st.success(f"Data processed successfully! Dropped {dropped} duplicate rows.")

    # ---- KPIs ----
    st.header("Key Performance Indicators")
    with st.container():
        k1, k2, k3, k4 = st.columns(4)
        total_sales = int(df[sales_col].sum())
        total_transactions = len(df)
        avg_order = round(df[sales_col].mean(), 2)
        unique_products = df[product_col].nunique() if product_col != 'None' else 0

        k1.markdown(f'<div class="kpi-card"><div class="small">Total Sales</div><div class="big">₹{total_sales:,}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-card"><div class="small">Transactions</div><div class="big">{total_transactions:,}</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-card"><div class="small">Avg Order Value</div><div class="big">₹{avg_order:,}</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi-card"><div class="small">Unique Products</div><div class="big">{unique_products}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ---- Visualizations ----
    st.header("Visualizations")

    # Sales Over Time
    with st.expander("Sales Trends Over Time", expanded=True):
        ts = df.groupby(date_col)[sales_col].sum().reset_index().sort_values(date_col)
        fig_ts = px.line(ts, x=date_col, y=sales_col, title="Total Sales Over Time", markers=True)
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # Product Sales (Bar and Pie)
    if product_col != 'None':
        with st.expander("Top Selling Products", expanded=False):
            st.subheader("Analysis of Product Sales")
            prod_sales = df.groupby(product_col)[sales_col].sum().reset_index().nlargest(10, sales_col)
            
            col_bar, col_pie = st.columns(2)
            with col_bar:
                fig_bar = px.bar(
                    prod_sales,
                    x=product_col,
                    y=sales_col,
                    title="Top 10 Selling Products (Bar Chart)",
                    labels={product_col: 'Product', sales_col: 'Total Sales'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_pie:
                fig_pie = px.pie(
                    prod_sales,
                    values=sales_col,
                    names=product_col,
                    title="Top 10 Selling Products (Pie Chart)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

    # Advanced EDA plots
    with st.expander("Advanced Exploratory Data Analysis (EDA) Plots", expanded=False):
        
        # Histograms
        st.subheader("Sales Distribution (Histogram)")
        fig_hist = px.histogram(df, x=sales_col, nbins=30, title="Distribution of Sales")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numerical_df = df.select_dtypes(include=np.number)
        corr_matrix = numerical_df.corr()
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix of Numerical Features"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Pair Plot
        st.subheader("Pair Plot")
        st.info("Warning: This plot can be memory-intensive for large datasets. It visualizes relationships between all numerical columns.")
        try:
            fig_pair = px.scatter_matrix(
                numerical_df,
                title="Pair Plot of Numerical Features"
            )
            st.plotly_chart(fig_pair, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate pair plot. Error: {e}. Try with fewer numerical columns.")

    st.markdown("---")
    
    # ---- Demand Forecasting (Prophet) ----
    if forecast_enable and PROPHET_INSTALLED:
        st.header("Demand Forecasting")
        st.markdown("Forecast future sales based on past performance using the Prophet library.")
        if st.button("Generate Forecast"):
            with st.spinner("Running forecasting model... This may take a moment."):
                df_ts = df.groupby(date_col)[sales_col].sum().reset_index().rename(
                    columns={date_col: "ds", sales_col: "y"}
                )
                df_ts = df_ts.dropna()
                
                m = Prophet()
                m.fit(df_ts)
                future = m.make_future_dataframe(periods=int(forecast_days))
                forecast = m.predict(future)

                st.success(f"Forecast for the next {int(forecast_days)} days generated successfully!")
                
                fig_forecast = plot_plotly(m, forecast)
                fig_forecast.update_layout(title_text=f"Future Sales Forecast for {forecast_days} Days")
                st.plotly_chart(fig_forecast, use_container_width=True)

                st.subheader("Forecasted Data")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days))
    else:
        if PROPHET_INSTALLED:
            st.info("Forecasting is disabled. Check the sidebar to enable it.")
        else:
            st.warning("Forecasting requires the `prophet` library. Please install it with `pip install prophet`.")

    st.markdown("---")
    # ---- Data Download ----
    def to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Cleaned Data (CSV)",
        to_csv_bytes(df),
        "cleaned_retail_data.csv",
        "text/csv"
    )

else:
    st.warning("Please ensure your data has both a Date and a Sales column, and that you have selected them correctly in the sidebar.")
