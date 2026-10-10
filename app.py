from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Future Sales Analyzer",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "module1"
    / "data"
    / "processed"
    / "ecommerce_sales_cleaned.csv"
)

SALES_MODEL_PATH = PROJECT_ROOT / "lightgbm_sales_model.pkl"

RETURN_MODEL_PATH = (
    PROJECT_ROOT
    / "module5"
    / "powerbi"
    / "return_risk_model.pkl"
)


# --------------------------------------------------
# 2. LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_sales_model():
    return joblib.load(SALES_MODEL_PATH)


@st.cache_resource
def load_return_model():
    return joblib.load(RETURN_MODEL_PATH)


# --------------------------------------------------
# 3. APP HEADER AND NAVIGATION
# --------------------------------------------------
st.title("Future Sales Analyzer")
st.write("Welcome to our E-Commerce Sales Analytics Project!")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Sales Prediction",
        "Return Risk Prediction",
        "Sales Forecast",
        "Power BI Dashboard",
        "Project Information"
    ]
)


# --------------------------------------------------
# 4. HOME - HISTORICAL SALES ANALYTICS
# --------------------------------------------------
if page == "Home":
    st.header("Historical Sales Analytics")

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Net Sales",
            f"{df['net_sales'].sum():,.2f}"
        )
        col2.metric(
            "Total Orders",
            f"{len(df):,}"
        )
        col3.metric(
            "Average Order Sales",
            f"{df['net_sales'].mean():,.2f}"
        )

        df["order_date"] = pd.to_datetime(
            df["order_date"],
            errors="coerce"
        )

        monthly_sales = (
            df.dropna(subset=["order_date"])
            .groupby(
                df.dropna(subset=["order_date"])["order_date"]
                .dt.to_period("M")
            )["net_sales"]
            .sum()
        )

        st.subheader("Monthly Sales Trend")
        st.line_chart(monthly_sales)

        category_sales = (
            df.groupby("category")["net_sales"]
            .sum()
            .sort_values(ascending=False)
        )

        st.subheader("Category-wise Sales")
        st.bar_chart(category_sales)

    else:
        st.error("Processed dataset not found. Check the dataset path.")


# --------------------------------------------------
# 5. SALES PREDICTION
# --------------------------------------------------
elif page == "Sales Prediction":
    st.header("Sales Prediction")
    st.write("Enter order details to predict net sales.")

    price = st.number_input(
        "Price", min_value=0.0, value=100.0
    )
    discount = st.number_input(
        "Discount (0 to 1)",
        min_value=0.0,
        max_value=1.0,
        value=0.1
    )
    quantity = st.number_input(
        "Quantity", min_value=1, value=1
    )
    delivery_time_days = st.number_input(
        "Delivery Time (Days)", min_value=0, value=3
    )
    shipping_cost = st.number_input(
        "Shipping Cost", min_value=0.0, value=10.0
    )
    customer_age = st.number_input(
        "Customer Age", min_value=1, value=25
    )
    year = st.number_input(
        "Year", min_value=2020, max_value=2100, value=2025
    )
    month = st.number_input(
        "Month", min_value=1, max_value=12, value=1
    )
    day = st.number_input(
        "Day", min_value=1, max_value=31, value=15
    )
    day_of_week = st.number_input(
        "Day of Week (0-6)", min_value=0, max_value=6, value=2
    )
    week_of_year = st.number_input(
        "Week of Year", min_value=1, max_value=53, value=3
    )

    discount_percent = discount * 100

    if st.button("Predict Sales"):
        try:
            sales_model = load_sales_model()

            input_data = pd.DataFrame([{
                "price": price,
                "discount": discount,
                "quantity": quantity,
                "delivery_time_days": delivery_time_days,
                "shipping_cost": shipping_cost,
                "customer_age": customer_age,
                "year": year,
                "month": month,
                "day": day,
                "day_of_week": day_of_week,
                "week_of_year": week_of_year,
                "discount_percent": discount_percent
            }])

            prediction = sales_model.predict(input_data)[0]

            st.success(
                f"Predicted Net Sales: {prediction:,.2f}"
            )

        except Exception as error:
            st.error(f"Sales prediction error: {error}")


# --------------------------------------------------
# 6. RETURN RISK PREDICTION
# --------------------------------------------------
elif page == "Return Risk Prediction":
    st.header("Return Risk Prediction")
    st.write("Enter order details to estimate return risk.")

    price = st.number_input(
        "Product Price", min_value=0.0,
        value=100.0, key="rr_price"
    )
    discount = st.number_input(
        "Discount (0 to 1)", min_value=0.0,
        max_value=1.0, value=0.1, key="rr_discount"
    )
    quantity = st.number_input(
        "Quantity", min_value=1,
        value=1, key="rr_quantity"
    )
    customer_age = st.number_input(
        "Customer Age", min_value=1,
        value=25, key="rr_age"
    )

    category = st.selectbox(
        "Category",
        ["Grocery", "Clothing", "Electronics",
         "Home", "Beauty", "Fashion", "Sports",
         "Toys", "Other"],
        key="rr_category"
    )

    payment_method = st.selectbox(
        "Payment Method",
        ["Credit Card", "Debit Card", "UPI",
         "Cash", "PayPal", "Other"],
        key="rr_payment"
    )

    region = st.selectbox(
        "Region",
        ["North", "South", "East",
         "West", "Central", "Other"],
        key="rr_region"
    )

    customer_gender = st.selectbox(
        "Customer Gender",
        ["Male", "Female", "Other"],
        key="rr_gender"
    )

    order_month = st.selectbox(
        "Order Month", list(range(1, 13)),
        key="rr_month"
    )

    order_day_of_week = st.selectbox(
        "Order Day of Week (0=Monday)",
        list(range(7)),
        key="rr_weekday"
    )

    if st.button("Estimate Return Risk"):
        try:
            return_model = load_return_model()

            input_data = pd.DataFrame([{
                "price": price,
                "discount": discount,
                "quantity": quantity,
                "customer_age": customer_age,
                "order_month": order_month,
                "order_day_of_week": order_day_of_week,
                "category": category,
                "payment_method": payment_method,
                "region": region,
                "customer_gender": customer_gender
            }])

            risk_score = return_model.predict_proba(
                input_data
            )[0][1]

            st.metric(
                "Estimated Return Risk",
                f"{risk_score:.1%}"
            )

            st.caption(
                "This is a model score, not a guarantee. "
                "The current model has limited return-detection "
                "performance."
            )

        except Exception as error:
            st.error(f"Return prediction error: {error}")


# --------------------------------------------------
# 7. FUTURE SALES FORECAST
# --------------------------------------------------
elif page == "Sales Forecast":
    st.header("Future Sales Forecast")

    forecast_path = (
        PROJECT_ROOT
        / "module4"
        / "future_sales_predictions.csv"
    )

    if forecast_path.exists():
        forecast_df = pd.read_csv(forecast_path)

        st.subheader("Predicted Monthly Sales")
        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        if (
            "order_date" in forecast_df.columns
            and "predicted_net_sales" in forecast_df.columns
        ):
            chart_data = forecast_df.copy()
            chart_data["order_date"] = pd.to_datetime(
                chart_data["order_date"]
            )
            chart_data = chart_data.set_index("order_date")

            st.line_chart(
                chart_data["predicted_net_sales"]
            )

        elif (
            "order_date" in forecast_df.columns
            and "net_sales" in forecast_df.columns
        ):
            chart_data = forecast_df.copy()
            chart_data["order_date"] = pd.to_datetime(
                chart_data["order_date"]
            )
            chart_data = chart_data.set_index("order_date")

            st.line_chart(chart_data["net_sales"])

    else:
        st.warning(
            "Forecast file not found. Generate the forecast first."
        )


# --------------------------------------------------
# 8. POWER BI DASHBOARD ANALYTICS
# --------------------------------------------------
elif page == "Power BI Dashboard":
    st.header("Power BI Dashboard")
    st.write(
        "Explore the analytics reports of Future Sales Analyzer."
    )

    dashboard_page = st.selectbox(
        "Choose Dashboard Page",
        [
            "Executive Overview",
            "Sales Analysis",
            "Customer & Return Analysis",
            "Discount & Profit/Loss Analysis",
            "Machine Learning Prediction",
            "SQL & Technical Analysis",
            "Future Sales Forecast",
            "Return Risk Analysis"
        ]
    )

    st.subheader(dashboard_page)

    # EXECUTIVE OVERVIEW
    if dashboard_page == "Executive Overview":
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Net Sales",
                f"{df['net_sales'].sum():,.2f}"
            )
            col2.metric(
                "Total Orders",
                f"{len(df):,}"
            )
            col3.metric(
                "Average Order Sales",
                f"{df['net_sales'].mean():,.2f}"
            )

            df["order_date"] = pd.to_datetime(
                df["order_date"],
                errors="coerce"
            )

            valid_dates = df.dropna(subset=["order_date"])

            monthly_sales = (
                valid_dates.groupby(
                    valid_dates["order_date"].dt.to_period("M")
                )["net_sales"]
                .sum()
            )

            st.subheader("Monthly Sales Trend")
            st.line_chart(monthly_sales)

            category_sales = (
                df.groupby("category")["net_sales"]
                .sum()
                .sort_values(ascending=False)
            )

            st.subheader("Category-wise Sales")
            st.bar_chart(category_sales)

        else:
            st.error("Processed dataset not found.")

    # SALES ANALYSIS
    elif dashboard_page == "Sales Analysis":
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            df["order_date"] = pd.to_datetime(
                df["order_date"],
                errors="coerce"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Monthly Sales")
                monthly_sales = (
                    df.dropna(subset=["order_date"])
                    .groupby(
                        df.dropna(subset=["order_date"])
                        ["order_date"].dt.to_period("M")
                    )["net_sales"]
                    .sum()
                )
                st.line_chart(monthly_sales)

            with col2:
                st.subheader("Sales by Region")
                region_sales = (
                    df.groupby("region")["net_sales"]
                    .sum()
                    .sort_values(ascending=False)
                )
                st.bar_chart(region_sales)

            st.subheader("Sales by Category")
            category_sales = (
                df.groupby("category")["net_sales"]
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(category_sales)

            st.subheader("Sales Data")
            st.dataframe(
                df[
                    ["order_date", "category", "region", "net_sales"]
                ].head(20),
                use_container_width=True
            )
        else:
            st.error("Processed dataset not found.")

    # CUSTOMER AND RETURN ANALYSIS
    elif dashboard_page == "Customer & Return Analysis":
        st.write("Customer behavior and return analysis.")

        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)

            if "returned" in df.columns:
                st.subheader("Return Status Distribution")
                st.bar_chart(df["returned"].value_counts())

            if "customer_age" in df.columns:
                st.subheader("Customer Age Distribution")
                st.bar_chart(df["customer_age"].value_counts().sort_index())

            if "customer_gender" in df.columns:
                st.subheader("Sales by Customer Gender")
                gender_sales = (
                    df.groupby("customer_gender")["net_sales"].sum()
                )
                st.bar_chart(gender_sales)

    # DISCOUNT AND PROFIT/LOSS ANALYSIS
    elif dashboard_page == "Discount & Profit/Loss Analysis":
        st.write("Discount impact and profit/loss analysis.")

        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)

            st.subheader("Sales by Discount")
            discount_sales = (
                df.groupby("discount")["net_sales"]
                .sum()
                .sort_index()
            )
            st.line_chart(discount_sales)

            if "profit_loss_status" in df.columns:
                st.subheader("Profit/Loss Status")
                st.bar_chart(
                    df["profit_loss_status"].value_counts()
                )
            elif "profit_margin" in df.columns:
                st.subheader("Profit Margin Distribution")
                st.bar_chart(
                    df["profit_margin"].value_counts().sort_index()
                )

    # MACHINE LEARNING PREDICTION
    elif dashboard_page == "Machine Learning Prediction":
        st.write("Machine learning model information.")
        st.info(
            "Use the Sales Prediction and Return Risk Prediction "
            "pages to run predictions."
        )

        performance_path = (
    PROJECT_ROOT
    / "model_performance_summary.csv"
)

        if performance_path.exists():
            st.subheader("Model Performance")
            st.dataframe(
                pd.read_csv(performance_path),
                use_container_width=True
            )

    # SQL AND TECHNICAL ANALYSIS
    elif dashboard_page == "SQL & Technical Analysis":
        st.write("Database and technical project information.")

        st.markdown("""
        - **Database:** PostgreSQL
        - **Data Processing:** Pandas
        - **Machine Learning:** LightGBM and Scikit-learn
        - **Dashboard:** Power BI
        - **Web Application:** Streamlit
        """)

    # FUTURE SALES FORECAST
    elif dashboard_page == "Future Sales Forecast":
        forecast_path = (
            PROJECT_ROOT
            / "module4"
            / "future_sales_predictions.csv"
        )

        if forecast_path.exists():
            forecast_df = pd.read_csv(forecast_path)
            st.subheader("Future Sales Forecast")
            st.dataframe(
                forecast_df,
                use_container_width=True
            )

            if (
                "order_date" in forecast_df.columns
                and "predicted_net_sales" in forecast_df.columns
            ):
                chart_data = forecast_df.copy()
                chart_data["order_date"] = pd.to_datetime(
                    chart_data["order_date"]
                )
                st.line_chart(
                    chart_data.set_index("order_date")
                    ["predicted_net_sales"]
                )
        else:
            st.warning("Forecast file not found.")

    # RETURN RISK ANALYSIS
    elif dashboard_page == "Return Risk Analysis":
        st.write("Return risk predictions and evaluation.")

        results_path = (
            PROJECT_ROOT
            / "module5"
            / "powerbi"
            / "return_risk_threshold_predictions.csv"
        )

        if results_path.exists():
            results = pd.read_csv(results_path)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Test Orders", f"{len(results):,}")

            actual_returned = (
                results["actual_returned"] == "Yes"
            ).sum()
            col2.metric(
                "Actual Returned Orders",
                f"{actual_returned:,}"
            )

            correct_returns = (
                (results["actual_returned"] == "Yes")
                & (results["predicted_returned"] == "Yes")
            ).sum()
            col3.metric(
                "Correctly Detected Returns",
                f"{correct_returns:,}"
            )

            missed_returns = (
                (results["actual_returned"] == "Yes")
                & (results["predicted_returned"] == "No")
            ).sum()
            col4.metric(
                "Missed Returns",
                f"{missed_returns:,}"
            )

            st.subheader("Predicted Return Status")
            st.bar_chart(
                results["predicted_returned"].value_counts()
            )

            st.subheader("Actual vs Predicted Return Status")
            comparison = pd.crosstab(
                results["actual_returned"],
                results["predicted_returned"]
            )
            st.dataframe(comparison)

            st.subheader("Return Risk Predictions")
            st.dataframe(
                results.head(100),
                use_container_width=True
            )
        else:
            st.warning(
                "Return-risk prediction CSV not found. "
                "Generate the predictions first."
            )


# --------------------------------------------------
# 9. PROJECT INFORMATION
# --------------------------------------------------
elif page == "Project Information":
    st.header("Project Information")

    st.write("**Programming Language:** Python")
    st.write("**Data Analysis:** Pandas")
    st.write("**Machine Learning:** LightGBM, Scikit-learn")
    st.write("**Database:** PostgreSQL")
    st.write("**Dashboard:** Power BI")
    st.write("**Web Application:** Streamlit")

