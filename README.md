
# 🚀 Supply Chain Risk Intelligence Platform

Welcome to the **Supply Chain Risk Assessment & Supplier Suitability Dashboard**, a powerful visual analytics and ML-powered tool designed to help organizations proactively monitor, assess, and mitigate supply chain disruptions.

This interactive platform consists of **three integrated modules** that provide end-to-end insights using **ensemble machine learning**, **geopolitical risk data**, and **supplier-specific reliability metrics**.

> ⚠️ **Deployment Note**: Due to the large size of the dataset (hundreds of MBs across INFORM and supplier logs), the dashboard **cannot be deployed online**. However, this repository includes:
>
> ✅ 3 annotated **screenshots**  
> 📂 Full **source code** for local deployment

---

## 📷 Visual Snapshots

| 📌 Dashboard | 🔍 Description |
|-------------|----------------|
| **Supply Chain Performance Dashboard** | Presents disruption likelihood, delay probability, and supplier route risks in a single view. It includes a risk distribution bar chart, correlation heatmap, and interactive scatter plots. Ideal for identifying which risk factors are interrelated. |
| **INFORM Risk Index Map** | Global choropleth visualizing country-specific disaster vulnerability scores. Built with INFORM/EMDAT datasets, it supports comparison across up to 10 countries. |
| **Supplier Suitability Chatbot** | AI-powered chatbot lets you input supplier features (like cargo condition or delivery delays) and returns a risk score with interpretability. Great for sourcing and procurement teams. |

---

## 🧭 Module Overview

### 📊 1. Supply Chain Performance Dashboard

This module visualizes key metrics for any selected product (e.g., P0005), such as:
- 📈 **Disruption Likelihood Score**
- ⏳ **Delay Probability**
- 🚚 **Route Risk & Supplier Reliability**
- 📉 **Delivery Time Deviations**

**Widgets & Plots:**
- Risk classification histogram (High / Moderate / Low)
- Heatmap of correlated risk factors
- Scatter plots to explore delivery patterns

🧠 **Purpose**: Quickly identify which routes, suppliers, or variables pose the most threat to operational continuity.

🖼️ **Screenshot**:  
![Supply Chain Dashboard Screenshot](./Screenshot%202025-05-02%20104052.png)

---

### 🌍 2. Global INFORM Risk Index Dashboard

Interactive choropleth visualization based on INFORM’s global disaster and conflict datasets.

**Functionality**:
- Select risk indicator (e.g., **Physical exposure to tsunami**)
- Choose year (e.g., 2025)
- Compare up to 10 countries by score
- See **Global Average**, **Highest**, and **Lowest** countries

🧠 **Purpose**: Adds geopolitical and climate exposure risk context to supplier selection and transport route decisions.

🖼️ **Screenshot**:  
![INFORM Risk Dashboard Screenshot](./Screenshot%202025-05-02%20104130.png)

---

### 🤖 3. Supplier Suitability Chatbot

This chatbot helps assess a supplier's risk level using interpretable inputs like:
- 📦 Cargo Condition Score
- ⏱️ Delivery Time Deviation
- 💰 Shipping Costs
- 📈 Historical Demand

Outputs a **Risk Score (0–1)** and classifies it as Low / Moderate / High.

🧠 **Purpose**: Empower sourcing teams to perform intelligent supplier evaluations in seconds.


🖼️ **Screenshot**:  
![Supplier Chatbot Screenshot](./Screenshot%202025-05-02%20141118.png)



