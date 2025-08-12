import streamlit as st

# Page config
st.set_page_config(page_title="RetailAnalytics", page_icon="📊", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f9fbfd;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 50px;
            background-color: white;
            box-shadow: 0px 1px 4px rgba(0,0,0,0.05);
        }
        .logo {
            font-weight: bold;
            font-size: 20px;
            color: #0b5394;
            display: flex;
            align-items: center;
        }
        .logo:before {
            content: "\\1F4CA";
            margin-right: 8px;
        }
        .btn {
            background-color: #007bff;
            color: white;
            padding: 10px 18px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 14px;
        }
        .btn:hover {
            background-color: #0056b3;
        }
        .hero {
            text-align: center;
            padding: 60px 20px;
        }
        .hero h1 {
            font-size: 38px;
            font-weight: bold;
            color: #0096a1;
        }
        .hero p {
            font-size: 16px;
            color: #555;
            max-width: 600px;
            margin: auto;
        }
        .hero-buttons {
            margin-top: 20px;
        }
        .hero-buttons a {
            margin: 0 5px;
            padding: 10px 18px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 500;
        }
        .btn-primary {
            background-color: #0096a1;
            color: white;
        }
        .btn-secondary {
            background-color: white;
            color: black;
            border: 1px solid #ccc;
        }
        .features {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 40px;
        }
        .feature-box {
            background: white;
            padding: 20px;
            border-radius: 12px;
            width: 250px;
            text-align: center;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
        }
        .feature-icon {
            font-size: 30px;
            color: #0096a1;
            margin-bottom: 10px;
        }
        .powered {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 10px 15px;
            border-radius: 12px;
            font-size: 12px;
            color: #666;
            box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <div class="logo">RetailAnalytics</div>
    <a href="#" class="btn">Get Started →</a>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <h1>Smart Retail Analytics Dashboard</h1>
    <p>Harness the power of AI-driven demand forecasting and comprehensive sales analytics to optimize your retail operations and boost profitability.</p>
    <div class="hero-buttons">
        <a href="#" class="btn-primary">Start Free Trial →</a>
        <a href="#" class="btn-secondary">Watch Demo</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Features Section
st.markdown("""
<div class="features">
    <div class="feature-box">
        <div class="feature-icon">📊</div>
        <h4>Real-time Analytics</h4>
        <p>Track sales performance across all channels</p>
    </div>
    <div class="feature-box">
        <div class="feature-icon">🧠</div>
        <h4>AI Forecasting</h4>
        <p>Predict demand with machine learning</p>
    </div>
    <div class="feature-box">
        <div class="feature-icon">📈</div>
        <h4>Growth Insights</h4>
        <p>Identify opportunities and trends</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Powered By
st.markdown("""
<div class="powered">Powered by vly.ai</div>
""", unsafe_allow_html=True)
