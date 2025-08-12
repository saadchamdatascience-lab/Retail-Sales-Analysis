import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("Retail Sales Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Show raw data
    st.subheader("📊 Raw Data")
    st.dataframe(df.head())
    
    # Basic stats
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())
    
    # Total sales over time
    sales_trend = df.groupby("Date")["Total Amount"].sum().reset_index()
    fig = px.line(sales_trend, x="Date", y="Total Amount", title="Total Sales Over Time")
    st.plotly_chart(fig)
    
    # Sales by category
    category_sales = df.groupby("Product Category")["Total Amount"].sum().reset_index()
    fig2 = px.bar(category_sales, x="Product Category", y="Total Amount", title="Sales by Product Category")
    st.plotly_chart(fig2)
    
    # Gender-wise sales
    gender_sales = df.groupby("Gender")["Total Amount"].sum().reset_index()
    fig3 = px.pie(gender_sales, names="Gender", values="Total Amount", title="Sales by Gender")
    st.plotly_chart(fig3)
    
    # Age group sales
    bins = [0, 18, 25, 35, 50, 100]
    labels = ["<18", "18-25", "26-35", "36-50", "50+"]
    df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    age_sales = df.groupby("Age Group")["Total Amount"].sum().reset_index()
    fig4 = px.bar(age_sales, x="Age Group", y="Total Amount", title="Sales by Age Group")
    st.plotly_chart(fig4)
