import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Retail Analytics & Forecasting", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .hero {
        text-align: center;
        padding: 80px 20px;
        background: linear-gradient(90deg, #4f46e5, #6366f1);
        color: white;
        border-radius: 12px;
        margin-bottom: 40px;
    }
    .hero h1 {
        font-size: 3.2rem;
        font-weight: 700;
        margin-bottom: 15px;
    }
    .hero p {
        font-size: 1.25rem;
        font-weight: 400;
        margin-bottom: 25px;
        color: rgba(255,255,255,0.9);
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .cta-buttons {
        display: flex;
        justify-content: center;
        gap: 16px;
    }
    .cta-buttons a {
        background: white;
        color: #4f46e5;
        padding: 14px 28px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .cta-buttons a:hover {
        background: #f4f4f4;
    }
    .cta-buttons a.secondary {
        background: transparent;
        border: 2px solid white;
        color: white;
    }
    .cta-buttons a.secondary:hover {
        background: rgba(255,255,255,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ---------- HERO SECTION ----------
st.markdown("""
<div class="hero">
    <h1>Smart Retail Analytics Dashboard</h1>
    <p>AI-powered insights to analyze your sales trends and forecast future demand. 
    Simply upload your CSV file and see your data come alive.</p>
    <div class="cta-buttons">
        <a href="https://yourdomain.com/register" target="_blank">Start Free Trial</a>
        <a class="secondary" href="https://yourdomain.com/demo" target="_blank">Watch Demo</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- MAIN CONTENT ----------
st.markdown("## 📊 Retail Sales Analysis & Forecasting")

uploaded_file = st.file_uploader("Upload your CSV sales data", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            sales_time = df.groupby('Date')['Sales'].sum().reset_index()
            fig = px.line(sales_time, x='Date', y='Sales', title="Sales Over Time")
            st.plotly_chart(fig, use_container_width=True)

        if 'Category' in df.columns:
            cat_sales = df.groupby('Category')['Sales'].sum().reset_index()
            fig = px.bar(cat_sales, x='Category', y='Sales', title="Sales by Category")
            st.plotly_chart(fig, use_container_width=True)

        if 'Date' in df.columns and 'Sales' in df.columns:
            st.subheader("📈 Demand Forecast")
            forecast_df = df.groupby('Date')['Sales'].sum().reset_index()
            forecast_df.columns = ['ds', 'y']
            
            model = Prophet()
            model.fit(forecast_df)

            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            fig_forecast = plot_plotly(model, forecast)
            st.plotly_chart(fig_forecast, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload your CSV file to see analysis and forecasting.")
