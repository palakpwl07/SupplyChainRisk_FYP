import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set dashboard title
st.set_page_config(page_title="Supply Chain Risk Dashboard", layout="wide")
st.title("📊 Supply Chain Risk Assessment Dashboard")

# --- Mock data ---
st.sidebar.header("📌 Filter Options")
region = st.sidebar.selectbox("Select Region", ["Global", "Asia", "Europe", "Americas"])
time_range = st.sidebar.slider("Select Time Range (Months)", 1, 12, 6)

# Sample supplier data
data = pd.DataFrame({
    'Supplier': [f'Supplier {i}' for i in range(1, 11)],
    'Country': np.random.choice(['India', 'Germany', 'China', 'USA', 'France'], 10),
    'Risk Score': np.round(np.random.uniform(0.2, 1.0, 10), 2),
    'Inventory Buffer (days)': np.random.randint(3, 20, 10),
    'Lead Time Variability': np.round(np.random.uniform(1.0, 5.0, 10), 2),
    'Alt Supplier Available': np.random.choice(['Yes', 'No'], 10)
})

# Display key metrics
st.markdown("### 🔍 Key Risk Metrics")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric(label="Average Risk Score", value=f"{data['Risk Score'].mean():.2f}")
kpi2.metric(label="% High Risk Suppliers", value=f"{(data['Risk Score'] > 0.7).mean()*100:.1f}%")
kpi3.metric(label="With Alt Supplier", value=f"{(data['Alt Supplier Available'] == 'Yes').mean()*100:.1f}%")

# Risk Score Bar Chart
st.markdown("### 📌 Supplier Risk Score")
fig_risk = px.bar(data, x='Supplier', y='Risk Score', color='Country', title="Risk Score by Supplier")
st.plotly_chart(fig_risk, use_container_width=True)

# Feature Importance Mock Data
st.markdown("### 🧠 Feature Importance (Model Insight)")
features = pd.DataFrame({
    'Feature': ['Event Impact Score', 'Hazard Exposure', 'Lead Time Variability', 'Inventory Buffer', 'Alt Supplier Availability'],
    'Importance': [0.28, 0.24, 0.18, 0.15, 0.15]
})
fig_feat = px.bar(features, x='Importance', y='Feature', orientation='h', title="Top Predictive Features", color='Importance', color_continuous_scale='Blues')
st.plotly_chart(fig_feat, use_container_width=True)

# Supplier Table
st.markdown("### 🧾 Supplier Risk Table")
st.dataframe(data.style.background_gradient(cmap='Reds', subset=['Risk Score']))
