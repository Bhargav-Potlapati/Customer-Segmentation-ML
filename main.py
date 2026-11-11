
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/marketing_campaign.csv", sep="\t")

sns.set_theme(style="whitegrid")

# --------------------------------
# 1. Income Distribution
# --------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Income",
    bins=40,
    kde=True
)

plt.title("Customer Income Distribution")
plt.xlabel("Income")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# --------------------------------
# 2. Income Outliers
# --------------------------------

plt.figure(figsize=(8, 4))

sns.boxplot(data=df, x="Income")

plt.title("Income Outlier Detection")
plt.xlabel("Income")
plt.tight_layout()
plt.show()


# --------------------------------
# 3. Product Spending
# --------------------------------

spending_columns = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

product_spending = df[spending_columns].sum().sort_values(
    ascending=False
)

plt.figure(figsize=(10, 5))

sns.barplot(
    x=product_spending.index,
    y=product_spending.values
)

plt.title("Total Spending by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Spending")
plt.xticks(rotation=35)
plt.tight_layout()
plt.show()


# --------------------------------
# 4. Purchase Channels
# --------------------------------

purchase_columns = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases"
]

channel_totals = df[purchase_columns].sum()

plt.figure(figsize=(8, 5))

sns.barplot(
    x=channel_totals.index,
    y=channel_totals.values
)

plt.title("Customer Purchase Channels")
plt.xlabel("Purchase Channel")
plt.ylabel("Total Purchases")
plt.tight_layout()
plt.show()


# --------------------------------
# 5. Customer Recency
# --------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Recency",
    bins=30,
    kde=True
)

plt.title("Customer Purchase Recency")
plt.xlabel("Days Since Last Purchase")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


print("ALL VISUALIZATIONS COMPLETED!")


# STEP 3: FEATURE ENGINEERING

# 1. Check customer registration dates
df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    dayfirst=True,
    errors="coerce"
)

print("\nCUSTOMER REGISTRATION DATE RANGE:")
print("Earliest date:", df["Dt_Customer"].min())
print("Latest date:", df["Dt_Customer"].max())

# 2. Check birth year range
print("\nBIRTH YEAR INFORMATION:")
print("Earliest birth year:", df["Year_Birth"].min())
print("Latest birth year:", df["Year_Birth"].max())

# 3. Check unusual birth years
print("\nCUSTOMERS BORN BEFORE 1920:")
print(df[df["Year_Birth"] < 1920][["Year_Birth", "Income"]])


# STEP 3.4: CREATE NEW FEATURES

# 1. Calculate approximate customer age
df["Age"] = 2014 - df["Year_Birth"]

# 2. Calculate total spending across all product categories
spending_columns = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

df["Total_Spending"] = df[spending_columns].sum(axis=1)

# 3. Calculate total purchases across all shopping channels
purchase_columns = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases"
]

df["Total_Purchases"] = df[purchase_columns].sum(axis=1)

# 4. Calculate total household members who are children or teenagers
df["Total_Household"] = df["Kidhome"] + df["Teenhome"]

# 5. Calculate average spending per purchase
df["Average_Spending"] = (
    df["Total_Spending"] /
    df["Total_Purchases"].replace(0, float("nan"))
)

# 6. Calculate customer registration tenure in months
reference_date = pd.Timestamp("2014-06-29")

df["Customer_Tenure_Months"] = (
    (reference_date - df["Dt_Customer"]).dt.days / 30.44
)

# 7. Flag unusual ages instead of deleting customers
df["Unusual_Age"] = df["Age"] > 100

# 8. Display newly created features
new_features = [
    "Age",
    "Total_Spending",
    "Total_Purchases",
    "Total_Household",
    "Average_Spending",
    "Customer_Tenure_Months",
    "Unusual_Age"
]

print("\nNEW FEATURES CREATED:")
print(df[new_features].head(10))

print("\nNEW FEATURE SUMMARY:")
print(df[new_features].describe())

print("\nUNUSUAL AGE RECORDS:")
print(df[df["Unusual_Age"]][["Year_Birth", "Age"]])

print("\nFEATURE ENGINEERING COMPLETED!")


# STEP 3.5: DATA QUALITY INVESTIGATION

# 1. Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# 2. Check customers with zero purchases
print("\nCUSTOMERS WITH ZERO PURCHASES:")
print(df[df["Total_Purchases"] == 0][
    ["Total_Spending", "Total_Purchases", "Average_Spending"]
])

# 3. Check unusually high income
print("\nHIGHEST CUSTOMER INCOMES:")
print(df.nlargest(10, "Income")[["Income", "Age"]])

# 4. Check unusually high spending
print("\nHIGHEST CUSTOMER SPENDING:")
print(df.nlargest(10, "Total_Spending")[
    ["Total_Spending", "Total_Purchases", "Average_Spending"]
])

# 5. Check age anomalies
print("\nCUSTOMERS WITH AGE ABOVE 100:")
print(df[df["Age"] > 100][["Year_Birth", "Age", "Income"]])

print("\nDATA QUALITY INVESTIGATION COMPLETED!")


# STEP 4.1: INVESTIGATE EXTREME INCOME

# Display the customer with the highest income
print("\nCUSTOMER WITH HIGHEST INCOME:")
print(df.loc[df["Income"].idxmax()].to_string())

# Check income distribution
print("\nINCOME PERCENTILES:")
print(df["Income"].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

# Calculate median income
income_median = df["Income"].median()

print("\nMEDIAN INCOME:", income_median)

# Count missing income values
print("\nMISSING INCOME COUNT:", df["Income"].isnull().sum())


# STEP 4.2: CREATE CLEANED DATASET

# 1. Create a separate copy
df_clean = df.copy()

print("\nORIGINAL DATASET SHAPE:", df.shape)

# 2. Remove customers with suspicious ages
df_clean = df_clean[df_clean["Age"] <= 100].copy()

# 3. Fill missing income with median income
income_median = df_clean["Income"].median()

df_clean["Income"] = df_clean["Income"].fillna(income_median)

# 4. Calculate the 99th percentile of income
income_99th = df_clean["Income"].quantile(0.99)

# 5. Create a capped income feature
df_clean["Income_Capped"] = df_clean["Income"].clip(
    upper=income_99th
)

# 6. Display cleaning results
print("\nCLEANED DATASET SHAPE:", df_clean.shape)

print("\nMISSING VALUES AFTER CLEANING:")
print(df_clean.isnull().sum())

print("\nINCOME BEFORE AND AFTER CAPPING:")
print("Original maximum income:", df_clean["Income"].max())
print("Capped maximum income:", df_clean["Income_Capped"].max())

print("\nCLEANED DATASET PREVIEW:")
print(df_clean.head())

print("\nDATA CLEANING COMPLETED!")


# STEP 4.3: SELECT FEATURES FOR CLUSTERING

numeric_features = [
    "Income_Capped",
    "Age",
    "Recency",
    "Total_Spending",
    "Total_Purchases",
    "Total_Household",
    "Customer_Tenure_Months",
    "NumDealsPurchases",
    "NumWebVisitsMonth"
]

categorical_features = [
    "Education",
    "Marital_Status"
]

# Select only the required columns
X = df_clean[numeric_features + categorical_features].copy()

print("\nSELECTED FEATURES:")
print(X.head())

print("\nFEATURE DATASET SHAPE:", X.shape)


# STEP 4.4: ONE-HOT ENCODING

X_encoded = pd.get_dummies(
    X,
    columns=categorical_features,
    dtype=int
)

print("\nENCODED DATASET:")
print(X_encoded.head())

print("\nENCODED DATASET SHAPE:", X_encoded.shape)


# STEP 4.5: HANDLE MISSING VALUES

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X_encoded)

print("\nMISSING VALUES AFTER IMPUTATION:")
print(pd.DataFrame(X_imputed).isnull().sum().sum())


# STEP 4.6: FEATURE SCALING

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

print("\nSCALED DATASET SHAPE:", X_scaled.shape)

print("\nFIRST FIVE SCALED ROWS:")
print(X_scaled[:5])

print("\nFEATURE PREPROCESSING COMPLETED!")


# STEP 5: FIND THE BEST NUMBER OF CLUSTERS

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Store evaluation results
inertia_values = []
silhouette_values = []

# Test cluster counts from 2 to 10
cluster_range = range(2, 11)

for k in cluster_range:

    # Create K-Means model
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Train the model
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calculate Elbow score
    inertia_values.append(kmeans.inertia_)

    # Calculate Silhouette score
    score = silhouette_score(
        X_scaled,
        cluster_labels,
        sample_size=1500,
        random_state=42
    )

    silhouette_values.append(score)

    print(
        f"Clusters: {k} | "
        f"Inertia: {kmeans.inertia_:.2f} | "
        f"Silhouette Score: {score:.4f}"
    )


# Plot both evaluation graphs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Method
axes[0].plot(
    list(cluster_range),
    inertia_values,
    marker="o"
)

axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters")
axes[0].set_ylabel("Inertia")
axes[0].set_xticks(list(cluster_range))


# Silhouette Score
axes[1].plot(
    list(cluster_range),
    silhouette_values,
    marker="o"
)

axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(list(cluster_range))

plt.tight_layout()
plt.show()

print("\nCLUSTER EVALUATION COMPLETED!")


# STEP 5.4: COMPARE DIFFERENT CLUSTER COUNTS

candidate_clusters = [5, 6, 9, 10]

profile_features = [
    "Income_Capped",
    "Age",
    "Recency",
    "Total_Spending",
    "Total_Purchases",
    "Total_Household",
    "Customer_Tenure_Months",
    "NumDealsPurchases",
    "NumWebVisitsMonth"
]

for k in candidate_clusters:

    print(f"\n{'=' * 45}")
    print(f"ANALYSIS FOR {k} CLUSTERS")
    print("=" * 45)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    # Create a copy for customer analysis
    profile_data = df_clean.reset_index(drop=True).copy()
    profile_data["Cluster"] = labels

    # Number of customers in each group
    print("\nCUSTOMERS PER CLUSTER:")
    print(profile_data["Cluster"].value_counts().sort_index())

    # Average characteristics of each group
    print("\nCUSTOMER CLUSTER PROFILES:")
    print(
        profile_data.groupby("Cluster")[profile_features]
        .mean()
        .round(2)
    )

print("\nCLUSTER COMPARISON COMPLETED!")


# STEP 5.5: DETAILED ANALYSIS OF 6 CUSTOMER SEGMENTS

# Train the selected K-Means model
final_kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10
)

# Assign cluster labels
df_clean = df_clean.reset_index(drop=True)
df_clean["Cluster"] = final_kmeans.fit_predict(X_scaled)

# Features for customer profiling
business_features = [
    "Income_Capped",
    "Age",
    "Recency",
    "Total_Spending",
    "Total_Purchases",
    "Total_Household",
    "NumDealsPurchases",
    "NumWebVisitsMonth"
]

# Display all columns without hiding any
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Calculate average customer characteristics
cluster_profile = (
    df_clean.groupby("Cluster")[business_features]
    .mean()
    .round(2)
)

# Calculate customer count
cluster_size = df_clean["Cluster"].value_counts().sort_index()

# Calculate percentage of customers
cluster_percentage = (
    df_clean["Cluster"].value_counts(normalize=True).sort_index() * 100
).round(2)

print("\nCUSTOMER COUNT PER CLUSTER:")
print(cluster_size)

print("\nCUSTOMER PERCENTAGE PER CLUSTER:")
print(cluster_percentage)

print("\nDETAILED CUSTOMER PROFILES:")
print(cluster_profile)

print("\nSIX-CLUSTER MODEL COMPLETED!")


# STEP 6: CUSTOMER SEGMENT VISUALIZATION

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# 1. Number of customers in each cluster
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df_clean,
    x="Cluster",
    hue="Cluster",
    palette="viridis",
    legend=False
)

plt.title("Number of Customers in Each Cluster")
plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# 2. Average spending by cluster
plt.figure(figsize=(9, 5))

sns.barplot(
    data=df_clean,
    x="Cluster",
    y="Total_Spending",
    hue="Cluster",
    palette="magma",
    legend=False
)

plt.title("Average Customer Spending by Cluster")
plt.xlabel("Customer Cluster")
plt.ylabel("Average Spending")
plt.tight_layout()
plt.show()


# 3. Average income by cluster
plt.figure(figsize=(9, 5))

sns.barplot(
    data=df_clean,
    x="Cluster",
    y="Income_Capped",
    hue="Cluster",
    palette="crest",
    legend=False
)

plt.title("Average Customer Income by Cluster")
plt.xlabel("Customer Cluster")
plt.ylabel("Average Income")
plt.tight_layout()
plt.show()


# 4. Average purchases by cluster
plt.figure(figsize=(9, 5))

sns.barplot(
    data=df_clean,
    x="Cluster",
    y="Total_Purchases",
    hue="Cluster",
    palette="flare",
    legend=False
)

plt.title("Average Purchases by Cluster")
plt.xlabel("Customer Cluster")
plt.ylabel("Average Purchases")
plt.tight_layout()
plt.show()

print("All customer segment visualizations completed!")


# STEP 7: SAVE TRAINED MODEL AND PREPROCESSING OBJECTS

import os
import joblib

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save trained KMeans model
joblib.dump(
    final_kmeans,
    "models/customer_segmentation_kmeans.pkl"
)

# Save scaler
joblib.dump(
    scaler,
    "models/customer_scaler.pkl"
)

# Save imputer
joblib.dump(
    imputer,
    "models/customer_imputer.pkl"
)

# Save feature column names and order
joblib.dump(
    X_encoded.columns.tolist(),
    "models/customer_feature_columns.pkl"
)

# Save customer data with cluster labels
df_clean.to_csv(
    "models/customer_segments.csv",
    index=False
)

print("\nMODEL SAVING COMPLETED!")

print("Saved files:")
print("1. customer_segmentation_kmeans.pkl")
print("2. customer_scaler.pkl")
print("3. customer_imputer.pkl")
print("4. customer_feature_columns.pkl")
print("5. customer_segments.csv")



# STEP 8: TEST THE SAVED MODEL (CORRECTED)

import joblib
import pandas as pd

# Load saved objects
loaded_model = joblib.load(
    "models/customer_segmentation_kmeans.pkl"
)

loaded_scaler = joblib.load(
    "models/customer_scaler.pkl"
)

loaded_imputer = joblib.load(
    "models/customer_imputer.pkl"
)

loaded_features = joblib.load(
    "models/customer_feature_columns.pkl"
)

# Select one existing customer
sample_customer = X.iloc[[0]].copy()

# Apply the SAME encoding used during training
sample_encoded = pd.get_dummies(
    sample_customer,
    columns=categorical_features,
    dtype=int
)

# Ensure the columns and order match training
sample_encoded = sample_encoded.reindex(
    columns=loaded_features,
    fill_value=0
)

# Apply saved imputer
sample_imputed = loaded_imputer.transform(sample_encoded)

# Apply saved scaler
sample_scaled = loaded_scaler.transform(sample_imputed)

# Predict cluster
predicted_cluster = loaded_model.predict(sample_scaled)[0]

# Get original cluster
actual_cluster = df_clean.iloc[0]["Cluster"]

print("\nSAVED MODEL TEST")
print("-------------------------")
print("Predicted Cluster:", predicted_cluster)
print("Original Cluster:", actual_cluster)

if predicted_cluster == actual_cluster:
    print("SUCCESS: Prediction matches original cluster!")
else:
    print("ERROR: Prediction does not match!")
