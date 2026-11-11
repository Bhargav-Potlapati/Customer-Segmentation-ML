
import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Customer Segmentation System")

st.write(
    "Discover customer segments using Machine Learning "
    "and KMeans clustering."
)

st.divider()

# -----------------------------------
# 2. LOAD SAVED MODEL
# -----------------------------------

@st.cache_resource
def load_models():

    model = joblib.load(
        "models/customer_segmentation_kmeans.pkl"
    )

    scaler = joblib.load(
        "models/customer_scaler.pkl"
    )

    imputer = joblib.load(
        "models/customer_imputer.pkl"
    )

    features = joblib.load(
        "models/customer_feature_columns.pkl"
    )

    return model, scaler, imputer, features


model, scaler, imputer, features = load_models()

# -----------------------------------
# 3. LOAD CUSTOMER DATA
# -----------------------------------

@st.cache_data
def load_customer_data():

    df = pd.read_csv(
        "models/customer_segments.csv"
    )

    return df


df = load_customer_data()

# -----------------------------------
# 4. DASHBOARD OVERVIEW
# -----------------------------------

st.header("📊 Customer Analytics Dashboard")

total_customers = len(df)

total_clusters = df["Cluster"].nunique()

average_spending = df["Total_Spending"].mean()

average_income = df["Income_Capped"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Customer Segments",
    total_clusters
)

col3.metric(
    "Average Spending",
    f"{average_spending:,.0f}"
)

col4.metric(
    "Average Income",
    f"{average_income:,.0f}"
)

st.divider()

# -----------------------------------
# 5. CUSTOMER DISTRIBUTION
# -----------------------------------

st.subheader("Customer Distribution by Cluster")

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)

# -----------------------------------
# 6. CLUSTER PERCENTAGES
# -----------------------------------

st.subheader("Customer Percentage by Cluster")

cluster_percentages = (
    df["Cluster"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

st.bar_chart(cluster_percentages)

# -----------------------------------
# 7. AVERAGE SPENDING
# -----------------------------------

st.subheader("Average Spending by Cluster")

spending_by_cluster = (
    df.groupby("Cluster")["Total_Spending"]
    .mean()
)

st.bar_chart(spending_by_cluster)

# -----------------------------------
# 8. AVERAGE INCOME
# -----------------------------------

st.subheader("Average Income by Cluster")

income_by_cluster = (
    df.groupby("Cluster")["Income_Capped"]
    .mean()
)

st.bar_chart(income_by_cluster)

# -----------------------------------
# 9. CLUSTER PROFILE
# -----------------------------------

st.subheader("Explore Individual Customer Segments")

selected_cluster = st.selectbox(
    "Select Customer Cluster",
    sorted(df["Cluster"].unique())
)

cluster_data = df[
    df["Cluster"] == selected_cluster
]

st.write(
    f"Customers in Cluster {selected_cluster}"
)

profile_col1, profile_col2, profile_col3 = st.columns(3)

profile_col1.metric(
    "Customers",
    len(cluster_data)
)

profile_col2.metric(
    "Average Spending",
    f"{cluster_data['Total_Spending'].mean():,.0f}"
)

profile_col3.metric(
    "Average Income",
    f"{cluster_data['Income_Capped'].mean():,.0f}"
)

st.subheader("Customer Segment Details")

st.dataframe(
    cluster_data[
        [
            "Income_Capped",
            "Age",
            "Total_Spending",
            "Total_Purchases",
            "Total_Household",
            "Cluster"
        ]
    ].head(20),
    use_container_width=True
)

st.divider()

# -----------------------------------
# 10. CUSTOMER PREDICTION
# -----------------------------------

st.header("🔮 Predict a New Customer Segment")

st.write(
    "Enter customer information below to predict "
    "which existing customer segment they belong to."
)

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:

        income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=50000.0
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40
        )

        recency = st.number_input(
            "Days Since Last Purchase",
            min_value=0,
            value=30
        )

        total_spending = st.number_input(
            "Total Product Spending",
            min_value=0.0,
            value=500.0
        )

        total_purchases = st.number_input(
            "Total Purchases",
            min_value=0,
            value=10
        )

    with col2:

        household = st.number_input(
            "Children and Teenagers at Home",
            min_value=0,
            value=1
        )

        tenure = st.number_input(
            "Customer Tenure (Months)",
            min_value=0.0,
            value=12.0
        )

        deals = st.number_input(
            "Deals Purchases",
            min_value=0,
            value=2
        )

        web_visits = st.number_input(
            "Website Visits Per Month",
            min_value=0,
            value=5
        )

        education = st.selectbox(
            "Education",
            [
                "Graduation",
                "PhD",
                "Master",
                "Basic",
                "2n Cycle"
            ]
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Married",
                "Together",
                "Single",
                "Divorced",
                "Widow",
                "Alone",
                "Absurd",
                "YOLO"
            ]
        )

    submitted = st.form_submit_button(
        "Predict Customer Segment"
    )

# -----------------------------------
# 11. PREDICTION PROCESS
# -----------------------------------

if submitted:

    customer = pd.DataFrame([{

        "Income_Capped": min(income, 94440.32),

        "Age": age,

        "Recency": recency,

        "Total_Spending": total_spending,

        "Total_Purchases": total_purchases,

        "Total_Household": household,

        "Customer_Tenure_Months": tenure,

        "NumDealsPurchases": deals,

        "NumWebVisitsMonth": web_visits,

        "Education": education,

        "Marital_Status": marital_status

    }])

    customer_encoded = pd.get_dummies(
        customer,
        columns=["Education", "Marital_Status"],
        dtype=int
    )

    customer_encoded = customer_encoded.reindex(
        columns=features,
        fill_value=0
    )

    customer_imputed = imputer.transform(
        customer_encoded
    )

    customer_scaled = scaler.transform(
        customer_imputed
    )

    prediction = model.predict(
        customer_scaled
    )[0]

    segment_names = {

        0: "Older, Higher-Spending Customers",

        1: "Lower-Spending Customers",

        2: "Lower-to-Moderate Spending Customers",

        3: "Moderate-to-High Spending Customers",

        4: "High-Value Customers",

        5: "Lower-Spending, Frequent Web Visitors"

    }

    st.divider()

    st.success(
        "Customer Segment Predicted Successfully!"
    )

    st.subheader(
        f"Customer belongs to Cluster {prediction}"
    )

    st.title(
        segment_names[prediction]
    )

    st.info(
        "Segment names are descriptive interpretations "
        "of the training data, not separate model predictions."
    )
