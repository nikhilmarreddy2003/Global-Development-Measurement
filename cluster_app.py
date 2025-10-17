#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# ===========================================
# STREAMLIT CONFIG
# ===========================================
st.set_page_config(page_title="DBSCAN Clustering Interactive Dashboard", layout="wide")
st.title("🌀 Interactive DBSCAN Clustering Dashboard")

st.markdown("""
Explore how **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** behaves 
by adjusting its parameters interactively and viewing the resulting clusters live!

---
""")

# ===========================================
# LOAD SCALER (OPTIONAL)
# ===========================================
try:
    scaler = joblib.load("scaler.pkl")
    st.sidebar.success("✅ Scaler loaded successfully.")
except Exception:
    st.sidebar.warning("Scaler not found — using new StandardScaler.")
    scaler = StandardScaler()

# ===========================================
# DATA UPLOAD
# ===========================================
st.sidebar.header("📂 Load Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ Data loaded successfully.")
else:
    st.sidebar.info("No file uploaded — using a sample dataset.")
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=500, centers=10, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])

st.write("### 🧾 Data Preview")
st.dataframe(df.head())

# ===========================================
# 🧹 DATA CLEANING
# ===========================================
st.markdown("### ⚙️ Data Preparation & Cleaning")

df_clean = df.copy()

# Step 1: Clean messy numeric strings
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace('₹', '', regex=False)
            .str.replace('$', '', regex=False)
            .str.strip()
        )
        df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')

# Step 2: Drop non-numeric columns
non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    st.warning(f"⚠️ Dropping non-numeric columns: {non_numeric_cols}")
    df_clean = df_clean.drop(columns=non_numeric_cols)

# Step 3: Ensure numeric columns exist
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("❌ No numeric columns found after cleaning. Please upload numeric data.")
    st.stop()

X = df_clean[numeric_cols]

# Step 4: Handle missing values
missing_count = X.isnull().sum().sum()
if missing_count > 0:
    st.warning(f"⚠️ Found {missing_count} missing numeric values — filling with column medians.")
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
else:
    st.info("✅ No missing numeric values found.")

# Step 5: Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
st.success(f"✅ Data cleaned & scaled successfully. Using {len(numeric_cols)} numeric columns for clustering.")
st.write("### 🧮 Cleaned Numeric Data Sample")
st.dataframe(X.head())

# ===========================================
# SIDEBAR PARAMETERS
# ===========================================
st.sidebar.header("⚙️ DBSCAN Parameters")
eps = st.sidebar.slider("Epsilon (eps):", 0.1, 5.0, 2.3, 0.1)
min_samples = st.sidebar.slider("Min Samples:", 2, 20, 6, 1)

# ===========================================
# RUN DBSCAN
# ===========================================
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)
df_clean["Cluster"] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
noise_pct = n_noise / len(labels) * 100

# Create cluster names
df_clean["Cluster Name"] = df_clean["Cluster"].apply(lambda x: f"Cluster {x}" if x != -1 else "Noise")

# ===========================================
# CLUSTER SUMMARY
# ===========================================
st.markdown("## 📊 Cluster Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Clusters Formed", n_clusters)
col2.metric("Noise Points", n_noise)
col3.metric("Noise %", f"{noise_pct:.2f}%")

cluster_summary = (
    df_clean["Cluster Name"].value_counts()
    .rename_axis("Cluster Name")
    .reset_index(name="Count")
    .assign(**{"% of Total": lambda x: (x["Count"] / len(df_clean) * 100).round(2)})
    .sort_values("Cluster Name")
)
st.dataframe(cluster_summary)

# ===========================================
# EVALUATION METRICS
# ===========================================
st.markdown("## 🧮 Evaluation Metrics")

mask = labels != -1
if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
    silhouette = silhouette_score(X_scaled[mask], labels[mask])
    dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
    chi = calinski_harabasz_score(X_scaled[mask], labels[mask])
else:
    silhouette, dbi, chi = np.nan, np.nan, np.nan

metrics_df = pd.DataFrame({
    "Metric": ["Silhouette (↑)", "Davies–Bouldin (↓)", "Calinski–Harabasz (↑)"],
    "Value": [round(silhouette, 3), round(dbi, 3), round(chi, 3)],
    "Meaning": [
        "Separation & compactness of clusters.",
        "Lower = better separation.",
        "Higher = better-defined clusters."
    ]
})
st.table(metrics_df)

# ===========================================
# PCA VISUALIZATION
# ===========================================
st.markdown("## 🎨 Cluster Visualization (PCA Projection)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Cluster"] = labels
df_pca["Cluster Name"] = df_clean["Cluster Name"]

fig, ax = plt.subplots(figsize=(8, 6))
for cluster_id in sorted(df_pca["Cluster"].unique()):
    subset = df_pca[df_pca["Cluster"] == cluster_id]
    label = "Noise (-1)" if cluster_id == -1 else f"Cluster {cluster_id}"
    color = "gray" if cluster_id == -1 else None
    ax.scatter(subset["PC1"], subset["PC2"], s=40, label=label, alpha=0.7, color=color)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples}) — {n_clusters} Clusters")
ax.legend()
st.pyplot(fig)

st.markdown("""
**Interpretation Tips:**
- Adjust `eps` to control neighborhood radius.
- Adjust `min_samples` to control density strictness.
- Gray points represent **noise/outliers**.
""")

# ===========================================
# 🔍 VIEW EACH CLUSTER SEPARATELY
# ===========================================
st.markdown("## 🔎 Explore Individual Clusters")

if n_clusters > 0:
    cluster_options = sorted([c for c in df_clean["Cluster"].unique() if c != -1])
    selected_cluster = st.selectbox("Select Cluster to Explore:", cluster_options)

    cluster_data = df_clean[df_clean["Cluster"] == selected_cluster]
    cluster_name = f"Cluster {selected_cluster}"

    st.write(f"### 📂 {cluster_name} — {len(cluster_data)} Samples")

    st.write("#### 📊 Cluster Summary Statistics")
    st.dataframe(cluster_data.describe().T.style.format("{:.2f}"))

    # PCA scatter (highlight selected cluster)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for cluster_id in sorted(df_pca["Cluster"].unique()):
        subset = df_pca[df_pca["Cluster"] == cluster_id]
        color = "lightgray" if cluster_id != selected_cluster else None
        alpha = 0.2 if cluster_id != selected_cluster else 0.9
        label = f"Cluster {cluster_id}" if cluster_id == selected_cluster else None
        ax2.scatter(subset["PC1"], subset["PC2"], s=50, alpha=alpha, color=color, label=label)

    highlight_subset = df_pca[df_pca["Cluster"] == selected_cluster]
    ax2.scatter(highlight_subset["PC1"], highlight_subset["PC2"], s=70, color="red", alpha=0.8, label=cluster_name)
    ax2.set_title(f"{cluster_name} in PCA Space")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    ax2.legend()
    st.pyplot(fig2)

    with st.expander("📋 Show Raw Data for This Cluster"):
        st.dataframe(cluster_data)
else:
    st.info("No clusters found (try adjusting eps or min_samples).")

# ===========================================
# 🔮 PREDICT NEW DATA
# ===========================================
st.sidebar.markdown("---")
st.sidebar.header("🔮 Predict Cluster for New Data")

def assign_dbscan_label(dbscan_obj, X_existing, new_point_scaled, eps):
    if not hasattr(dbscan_obj, "core_sample_indices_") or len(dbscan_obj.core_sample_indices_) == 0:
        return -1
    core_idx = dbscan_obj.core_sample_indices_
    core_samples = X_existing[core_idx]
    nbrs = NearestNeighbors(n_neighbors=1).fit(core_samples)
    dist, idx = nbrs.kneighbors(new_point_scaled.reshape(1, -1), return_distance=True)
    nearest_dist = dist[0, 0]
    core_pos = core_idx[idx[0, 0]]
    label_of_core = dbscan_obj.labels_[core_pos]
    return int(label_of_core) if nearest_dist <= eps else -1

user_input = []
for col in X.columns:
    val = st.sidebar.number_input(f"{col}", value=float(df_clean[col].mean()))
    user_input.append(val)

if st.sidebar.button("Predict Cluster"):
    new_scaled = scaler.transform([user_input])
    label = assign_dbscan_label(dbscan, X_scaled, new_scaled[0], eps)
    if label == -1:
        st.sidebar.warning("⚠️ Predicted as noise (-1).")
    else:
        st.sidebar.success(f"✅ Predicted Cluster: {label}")

# ===========================================
# LEARNING SECTION
# ===========================================
st.markdown("""
---
## 🧠 Learn DBSCAN
**1️⃣ What makes DBSCAN unique?**
- Detects clusters of arbitrary shape.
- Identifies noise/outliers automatically.
- No need to predefine number of clusters (unlike K-Means).

**2️⃣ Key Parameters:**
- `eps`: radius of neighborhood (cluster width)
- `min_samples`: min points in a dense region

**3️⃣ Metrics Meaning:**
- Silhouette ↑ → better separation & compactness
- Davies–Bouldin ↓ → better distinctness
- Calinski–Harabasz ↑ → better-defined clusters
---
> 💡 Try adjusting `eps` and `min_samples` on the left panel to see how DBSCAN adapts dynamically!
""")


