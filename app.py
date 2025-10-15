# app.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
import scipy.cluster.hierarchy as sch

# --- Load dataset ---
data = fetch_california_housing(as_frame=True)
df = data.frame

st.title("California Housing Interactive Dashboard")

# --- Sidebar filters ---
income_filter = st.slider("Median Income", float(df.MedInc.min()), float(df.MedInc.max()), (1.0, 5.0))
age_filter = st.slider("House Age", int(df.HouseAge.min()), int(df.HouseAge.max()), (10, 30))

filtered_df = df[(df.MedInc >= income_filter[0]) & (df.MedInc <= income_filter[1]) &
                 (df.HouseAge >= age_filter[0]) & (df.HouseAge <= age_filter[1])]

st.write("Filtered Data:", filtered_df.shape[0], "rows")

# --- Visualization: Distribution ---
st.subheader("House Value Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_df["MedHouseVal"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# --- Hierarchical Clustering ---
st.subheader("Hierarchical Clustering (Sample of 500 rows)")
sample_df = df.sample(500, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sample_df.drop(columns=["MedHouseVal"]))

# Dendrogram
fig, ax = plt.subplots(figsize=(8, 4))
sch.dendrogram(sch.linkage(X_scaled, method="ward"), no_labels=True)
st.pyplot(fig)

# Agglomerative Clustering visualization
agg = AgglomerativeClustering(n_clusters=4)
clusters = agg.fit_predict(X_scaled)
fig, ax = plt.subplots()
sns.scatterplot(x=sample_df["Longitude"], y=sample_df["Latitude"], hue=clusters, palette="Set2", s=10, ax=ax)
st.pyplot(fig)

# --- GridSearchCV: Ridge Regression ---
st.subheader("GridSearchCV: Ridge Regression")
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge()
param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}
grid_ridge = GridSearchCV(ridge, param_grid, scoring="r2", cv=5)
grid_ridge.fit(X_train, y_train)

st.write("Best Ridge Parameters:", grid_ridge.best_params_)
st.write("Best R² Score:", grid_ridge.best_score_)

# --- GridSearchCV: Decision Tree ---
st.subheader("GridSearchCV: Decision Tree (Binary Classification)")
median_val = y.median()
y_bin = np.where(y > median_val, 1, 0)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.2, random_state=42)

param_grid_tree = {"max_depth": [3, 5, 7, 10], "min_samples_split": [2, 5, 10]}
grid_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_tree, scoring="accuracy", cv=5)
grid_tree.fit(X_train_bin, y_train_bin)

st.write("Best Decision Tree Parameters:", grid_tree.best_params_)
st.write("Best Accuracy:", grid_tree.best_score_)
