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

          layout="wide",

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

          PROJECT_ROOT / "module5" / "powerbi" / "return_risk_model.pkl"

)



# --------------------------------------------------

# 2. LOAD MODELS AND DATA

# --------------------------------------------------

@st.cache_resource

def load_sales_model():

          return joblib.load(SALES_MODEL_PATH)





@st.cache_resource

def load_return_model():

          return joblib.load(RETURN_MODEL_PATH)





@st.cache_data

def load_dataset():

          df = pd.read_csv(DATA_PATH)

          if "order_date" in df.columns:

                    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

          return df





def show_home():
    """Home page: keep KPI cards and show a project overview instead of charts."""
    st.header("Historical Sales Analytics")
    st.write(
        "Welcome to Future Sales Analyzer — an e-commerce analytics application "
        "that helps explore customer behaviour, returns, discounts, profitability, "
        "and future sales predictions."
    )

    if not DATA_PATH.exists():
        st.error("Processed dataset not found. Check the dataset path.")
        return

    df = load_dataset().copy()
    if "net_sales" not in df.columns:
        st.error("The dataset does not contain a 'net_sales' column.")
        return

    # Keep the three KPI cards on the Home page.
    net_sales = pd.to_numeric(df["net_sales"], errors="coerce")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Net Sales", f"{net_sales.sum():,.2f}")
    col2.metric("Total Orders", f"{len(df):,}")
    col3.metric("Average Order Sales", f"{net_sales.mean():,.2f}")

    st.divider()
    st.subheader("Explore Project Features")
    st.caption(
        "Choose a section from the navigation menu to explore the analysis and prediction tools."
    )

    features = [
        ("👥 Customer Segmentation", "Group customers into high-, medium-, and low-value segments."),
        ("📦 Product Return Analysis", "Explore return patterns in the e-commerce dataset."),
        ("🏷️ Discount Impact Analysis", "Compare sales across different discount levels."),
        ("💰 Product-wise Profit & Loss", "Review available profit/loss and profit-margin information."),
        ("🛡️ Return Risk Prediction", "Estimate the return risk for an order using its details."),
        ("📈 Future Sales Prediction", "View the forecast generated from historical monthly sales."),
    ]

    for row_start in range(0, len(features), 3):
        columns = st.columns(3)
        for column, (title, description) in zip(columns, features[row_start:row_start + 3]):
            with column:
                with st.container(border=True):
                    st.markdown(f"#### {title}")
                    st.write(description)

    st.info(
        "Tip: Open **Power BI Dashboard** for detailed reports, or use **Sales Prediction**, "
        "**Return Risk Prediction**, and **Sales Forecast** from the navigation menu."
    )

def show_sales_prediction():

          st.header("Sales Prediction")

          st.write("Enter order details to predict net sales.")



          price = st.number_input("Price", min_value=0.0, value=100.0)

          discount = st.number_input(

                    "Discount (0 to 1)", min_value=0.0, max_value=1.0, value=0.1

          )

          quantity = st.number_input("Quantity", min_value=1, value=1)

          delivery_time_days = st.number_input(

                    "Delivery Time (Days)", min_value=0, value=3

          )

          shipping_cost = st.number_input("Shipping Cost", min_value=0.0, value=10.0)

          customer_age = st.number_input("Customer Age", min_value=1, value=25)

          year = st.number_input("Year", min_value=2020, max_value=2100, value=2025)

          month = st.number_input("Month", min_value=1, max_value=12, value=1)

          day = st.number_input("Day", min_value=1, max_value=31, value=15)

          day_of_week = st.number_input(

                    "Day of Week (0-6)", min_value=0, max_value=6, value=2

          )

          week_of_year = st.number_input(

                    "Week of Year", min_value=1, max_value=53, value=3

          )



          if st.button("Predict Sales"):

                    try:

                              model = load_sales_model()

                              input_data = pd.DataFrame(

                                        [

                                                  {

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

                                                            "discount_percent": discount * 100,

                                                  }

                                        ]

                              )

                              prediction = model.predict(input_data)[0]

                              st.success(f"Predicted Net Sales: {prediction:,.2f}")

                    except Exception as error:

                              st.error(f"Sales prediction error: {error}")





def show_return_risk_prediction():

          st.header("Return Risk Prediction")

          st.write("Enter order details to estimate return risk.")



          price = st.number_input("Product Price", min_value=0.0, value=100.0, key="rr_price")

          discount = st.number_input(

                    "Discount (0 to 1)", min_value=0.0, max_value=1.0,

                    value=0.1, key="rr_discount"

          )

          quantity = st.number_input("Quantity", min_value=1, value=1, key="rr_quantity")

          customer_age = st.number_input("Customer Age", min_value=1, value=25, key="rr_age")

          category = st.selectbox(

                    "Category",

                    ["BEAUTY", "ELECTRONICS", "FASHION", "GROCERY", "HOME", "SPORTS", "TOYS"],

                    key="rr_category",

          )

          payment_method = st.selectbox(

                    "Payment Method",

                    ["Credit Card", "Debit Card", "UPI", "Cash", "PayPal", "Other"],

                    key="rr_payment",

          )

          region = st.selectbox(

                    "Region", ["North", "South", "East", "West", "Central", "Other"],

                    key="rr_region",

          )

          customer_gender = st.selectbox(

                    "Customer Gender", ["Male", "Female", "Other"], key="rr_gender"

          )

          order_month = st.selectbox("Order Month", list(range(1, 13)), key="rr_month")

          order_day_of_week = st.selectbox(

                    "Order Day of Week (0=Monday)", list(range(7)), key="rr_weekday"

          )



          if st.button("Estimate Return Risk"):

                    try:

                              model = load_return_model()

                              input_data = pd.DataFrame(

                                        [

                                                  {

                                                            "price": price,

                                                            "discount": discount,

                                                            "quantity": quantity,

                                                            "customer_age": customer_age,

                                                            "order_month": order_month,

                                                            "order_day_of_week": order_day_of_week,

                                                            "category": category,

                                                            "payment_method": payment_method,

                                                            "region": region,

                                                            "customer_gender": customer_gender,

                                                  }

                                        ]

                              )

                              risk_score = model.predict_proba(input_data)[0][1]

                              st.metric("Estimated Return Risk", f"{risk_score:.1%}")

                              st.caption(

                                        "This is a model score, not a guarantee. The current model has "

                                        "limited return-detection performance."

                              )

                    except Exception as error:

                              st.error(f"Return prediction error: {error}")





def show_sales_forecast():

          st.header("Future Sales Forecast")

          forecast_path = PROJECT_ROOT / "module4" / "future_sales_predictions.csv"



          if not forecast_path.exists():

                    st.warning("Forecast file not found. Generate the forecast first.")

                    return



          forecast_df = pd.read_csv(forecast_path)

          st.subheader("Predicted Monthly Sales")

          st.dataframe(forecast_df, use_container_width=True)



          value_column = (

                    "predicted_net_sales"

                    if "predicted_net_sales" in forecast_df.columns

                    else "net_sales" if "net_sales" in forecast_df.columns else None

          )

          if "order_date" in forecast_df.columns and value_column:

                    chart_data = forecast_df.copy()

                    chart_data["order_date"] = pd.to_datetime(

                              chart_data["order_date"], errors="coerce"

                    )

                    chart_data[value_column] = pd.to_numeric(

                              chart_data[value_column], errors="coerce"

                    )

                    chart_data = chart_data.dropna(subset=["order_date", value_column])

                    if not chart_data.empty:

                              chart_data = chart_data.set_index("order_date").sort_index()

                              st.line_chart(chart_data[value_column], use_container_width=True)





def show_power_bi_dashboard():

          st.header("Power BI Dashboard")

          st.write("Explore the analytics reports of Future Sales Analyzer.")



          dashboard_page = st.selectbox(

                    "Choose Dashboard Page",

                    [

                              "Executive Overview",

                              "Sales Analysis",

                              "Customer & Return Analysis",

                                                            "Customer Segmentation",

"Discount & Profit/Loss Analysis",


                              "SQL & Technical Analysis",

                              "Future Sales Forecast",

                              "Return Risk Analysis",

                    ],

          )

          st.subheader(dashboard_page)



          if dashboard_page in ["Executive Overview", "Sales Analysis",

                                                                 "Customer & Return Analysis",

                                                                                                                                  "Customer Segmentation",

                                                                 "Discount & Profit/Loss Analysis"]:

                    if not DATA_PATH.exists():

                              st.error("Processed dataset not found.")

                              return

                    df = load_dataset().copy()



                    if dashboard_page == "Executive Overview":

                              col1, col2, col3 = st.columns(3)

                              col1.metric("Total Net Sales", f"{df['net_sales'].sum():,.2f}")

                              col2.metric("Total Orders", f"{len(df):,}")

                              col3.metric("Average Order Sales", f"{df['net_sales'].mean():,.2f}")

                              if "order_date" in df.columns:

                                        valid = df.dropna(subset=["order_date"]).copy()

                                        if not valid.empty:

                                                  monthly = (

                                                            valid.set_index("order_date").sort_index()

                                                            .resample("MS")["net_sales"].sum()

                                                  )

                                                  monthly.index = monthly.index.strftime("%Y-%m")

                                                  st.subheader("Monthly Sales Trend")

                                                  st.line_chart(monthly, use_container_width=True)

                              if "category" in df.columns:

                                        st.subheader("Category-wise Sales")

                                        st.bar_chart(df.groupby("category")["net_sales"].sum().sort_values(ascending=False))

                              # Restore the region/category sales analysis that was previously available.
                              if all(column in df.columns for column in ["region", "category", "net_sales"]):

                                        st.subheader("Region-wise Category Sales")
                                        region_category_sales = (
                                                  df.groupby(["region", "category"], as_index=False)["net_sales"]
                                                  .sum()
                                                  .sort_values("net_sales", ascending=False)
                                        )
                                        st.bar_chart(
                                                  region_category_sales,
                                                  x="category",
                                                  y="net_sales",
                                                  color="region",
                                                  use_container_width=True,
                                        )

                                        st.subheader("Highest-Selling Category in Each Region")
                                        top_category_by_region = region_category_sales.loc[
                                                  region_category_sales.groupby("region")["net_sales"].idxmax()
                                        ].sort_values("net_sales", ascending=False)
                                        top_category_by_region = top_category_by_region.rename(
                                                  columns={"region": "Region", "category": "Top Category", "net_sales": "Sales Amount"}
                                        )
                                        st.dataframe(
                                                  top_category_by_region.style.format({"Sales Amount": "{:,.2f}"}),
                                                  use_container_width=True,
                                                  hide_index=True,
                                        )

                                        st.subheader("Highest-Selling Region for Each Category")
                                        top_region_by_category = region_category_sales.loc[
                                                  region_category_sales.groupby("category")["net_sales"].idxmax()
                                        ].sort_values("net_sales", ascending=False)
                                        top_region_by_category = top_region_by_category.rename(
                                                  columns={"region": "Top Region", "category": "Category", "net_sales": "Sales Amount"}
                                        )
                                        st.dataframe(
                                                  top_region_by_category.style.format({"Sales Amount": "{:,.2f}"}),
                                                  use_container_width=True,
                                                  hide_index=True,
                                        )



                    elif dashboard_page == "Sales Analysis":

                              if "order_date" in df.columns:

                                        valid = df.dropna(subset=["order_date"]).copy()

                                        monthly = valid.set_index("order_date").sort_index().resample("MS")["net_sales"].sum()

                                        monthly.index = monthly.index.strftime("%Y-%m")

                                        st.subheader("Monthly Sales")

                                        st.line_chart(monthly, use_container_width=True)

                              if "region" in df.columns:

                                        st.subheader("Sales by Region")

                                        st.bar_chart(df.groupby("region")["net_sales"].sum().sort_values(ascending=False))

                              columns = [c for c in ["order_date", "category", "region", "net_sales"] if c in df.columns]

                              st.subheader("Sales Data")

                              st.dataframe(df[columns].head(20), use_container_width=True)

                              # Search a category and compare its net sales across every region.
                              st.divider()
                              st.subheader("Search Category-wise Sales by Region")
                              category_search = st.text_input(
                                        "Search Product Category",
                                        placeholder="Type BEAUTY, ELECTRONICS, FASHION, GROCERY, HOME, SPORTS, TOYS...",
                                        key="sales_analysis_category_search",
                              )

                              if category_search.strip():
                                        if all(column in df.columns for column in ["category", "region", "net_sales"]):
                                                  search_df = df.copy()
                                                  search_df["net_sales"] = pd.to_numeric(
                                                            search_df["net_sales"], errors="coerce"
                                                  )
                                                  matched_df = search_df[
                                                            search_df["category"].astype(str).str.contains(
                                                                      category_search.strip(),
                                                                      case=False,
                                                                      na=False,
                                                            )
                                                  ].dropna(subset=["net_sales"])

                                                  if not matched_df.empty:
                                                            region_sales = (
                                                                      matched_df.groupby("region", as_index=False)["net_sales"]
                                                                      .sum()
                                                                      .sort_values("net_sales", ascending=False)
                                                            )
                                                            st.subheader(
                                                                      f"{category_search.strip()} - Region-wise Sales"
                                                            )
                                                            st.bar_chart(
                                                                      region_sales,
                                                                      x="region",
                                                                      y="net_sales",
                                                                      x_label="Region",
                                                                      y_label="Total Net Sales",
                                                                      use_container_width=True,
                                                            )
                                                            display_region_sales = region_sales.rename(
                                                                      columns={
                                                                                "region": "Region",
                                                                                "net_sales": "Total Net Sales",
                                                                      }
                                                            )
                                                            st.dataframe(
                                                                      display_region_sales.style.format(
                                                                                {"Total Net Sales": "{:,.2f}"}
                                                                      ),
                                                                      use_container_width=True,
                                                                      hide_index=True,
                                                            )
                                                  else:
                                                            st.warning(
                                                                      "No matching category found in the dataset. "
                                                                      "Check the category name and try again."
                                                            )
                                        else:
                                                  st.error(
                                                            "The dataset must contain category, region, and net_sales columns."
                                                  )
                              else:
                                        st.info(
                                                  "Enter a category name to view its sales across all regions."
                                        )


                    elif dashboard_page == "Customer Segmentation":
                              st.write("Explore customers grouped by their purchasing value.")

                              segment_df = df.copy()
                              customer_col = "customer_id" if "customer_id" in segment_df.columns else None
                              sales_col = "net_sales" if "net_sales" in segment_df.columns else (
                                        "total_amount" if "total_amount" in segment_df.columns else None
                              )

                              # Use an existing segment column when available; otherwise
                              # derive High/Medium/Low Value segments from customer sales.
                              if "customer_segment" not in segment_df.columns:
                                        if customer_col and sales_col:
                                                  segment_df[sales_col] = pd.to_numeric(segment_df[sales_col], errors="coerce")
                                                  customer_sales = segment_df.groupby(customer_col)[sales_col].sum(min_count=1).dropna()
                                                  if len(customer_sales) >= 3:
                                                            customer_segments = pd.qcut(
                                                                      customer_sales.rank(method="first"),
                                                                      q=3,
                                                                      labels=["Low Value", "Medium Value", "High Value"]
                                                            )
                                                            segment_lookup = customer_segments.astype(str).to_dict()
                                                            segment_df["customer_segment"] = segment_df[customer_col].map(segment_lookup)
                                                  else:
                                                            st.warning("Not enough customers to create three segments.")
                                                            segment_df["customer_segment"] = "Medium Value"
                                        else:
                                                  st.warning("No customer_segment column or customer_id/sales fields were found to build segments.")
                                                  segment_df["customer_segment"] = "Unclassified"

                              segment_df = segment_df.dropna(subset=["customer_segment"]).copy()
                              if segment_df.empty:
                                        st.info("No customer segment data is available to display.")
                              else:
                                        if customer_col:
                                                  customer_counts = segment_df.groupby("customer_segment")[customer_col].nunique().rename("customers")
                                        else:
                                                  customer_counts = segment_df["customer_segment"].value_counts().rename("customers")
                                                  st.caption("Customer counts use dataset rows because customer_id is unavailable.")

                                        order_counts = segment_df["customer_segment"].value_counts().rename("total_orders")
                                        segment_summary = pd.concat([customer_counts, order_counts], axis=1).fillna(0).astype(int)
                                        segment_order = [name for name in ["High Value", "Medium Value", "Low Value"] if name in segment_summary.index]
                                        segment_order += [name for name in segment_summary.index if name not in segment_order]
                                        segment_summary = segment_summary.reindex(segment_order)

                                        chart_col1, chart_col2 = st.columns(2)
                                        with chart_col1:
                                                  st.subheader("Customer Segment Distribution")
                                                  donut_data = segment_summary.reset_index()
                                                  donut_data = donut_data.rename(columns={donut_data.columns[0]: "customer_segment"})
                                                  donut_spec = {
                                                            "mark": {"type": "arc", "innerRadius": 65},
                                                            "encoding": {
                                                                      "theta": {"field": "customers", "type": "quantitative"},
                                                                      "color": {"field": "customer_segment", "type": "nominal", "title": "Customer Segment"},
                                                                      "tooltip": [
                                                                                {"field": "customer_segment", "type": "nominal", "title": "Segment"},
                                                                                {"field": "customers", "type": "quantitative", "title": "Customers"}
                                                                      ]
                                                            },
                                                            "view": {"stroke": None}
                                                  }
                                                  st.vega_lite_chart(donut_data, donut_spec, use_container_width=True)
                                                  st.caption(f"Total customers: {int(segment_summary['customers'].sum()):,}")

                                        with chart_col2:
                                                  st.subheader("Orders by Customer Segment")
                                                  st.bar_chart(segment_summary[["total_orders"]], use_container_width=True)

                                        st.subheader("Customer Segment Summary")
                                        summary_display = segment_summary.reset_index().rename(columns={
                                                  "index": "Customer Segment",
                                                  "customers": "Customers",
                                                  "total_orders": "Total Orders"
                                        })
                                        st.dataframe(summary_display, use_container_width=True, hide_index=True)

                                        # Show the purchase amount range for each existing customer segment.
                                        if customer_col and sales_col:
                                                  customer_amounts = segment_df.copy()
                                                  customer_amounts[sales_col] = pd.to_numeric(customer_amounts[sales_col], errors="coerce")
                                                  customer_amounts = customer_amounts.dropna(subset=[customer_col, sales_col, "customer_segment"])
                                                  customer_amounts = customer_amounts.groupby([customer_col, "customer_segment"], as_index=False)[sales_col].sum()
                                                  amount_ranges = customer_amounts.groupby("customer_segment")[sales_col].agg(
                                                            Minimum_Purchase="min", Maximum_Purchase="max"
                                                  ).reset_index()
                                                  amount_ranges["Customer Segment"] = amount_ranges["customer_segment"]
                                                  amount_ranges["Minimum Purchase Amount"] = amount_ranges["Minimum_Purchase"].map(lambda value: f"{value:,.2f}")
                                                  amount_ranges["Maximum Purchase Amount"] = amount_ranges["Maximum_Purchase"].map(lambda value: f"{value:,.2f}")
                                                  amount_ranges = amount_ranges[["Customer Segment", "Minimum Purchase Amount", "Maximum Purchase Amount"]]
                                                  segment_rank = {"High Value": 0, "Medium Value": 1, "Low Value": 2}
                                                  amount_ranges["_sort"] = amount_ranges["Customer Segment"].map(segment_rank).fillna(3)
                                                  amount_ranges = amount_ranges.sort_values("_sort").drop(columns="_sort")
                                                  st.subheader("Customer Purchase Amount Range")
                                                  st.caption("Ranges are based on each customer's total purchase amount in the loaded dataset.")
                                                  st.dataframe(amount_ranges, use_container_width=True, hide_index=True)

                    elif dashboard_page == "Customer & Return Analysis":

                              st.write("Customer behavior and return analysis.")

                              if "returned" in df.columns:

                                        st.subheader("Return Status Distribution")

                                        st.bar_chart(df["returned"].value_counts())

                              if "customer_age" in df.columns:

                                        st.subheader("Customer Age Distribution")

                                        st.bar_chart(df["customer_age"].value_counts().sort_index())

                              if "customer_gender" in df.columns:

                                        st.subheader("Sales by Customer Gender")

                                        st.bar_chart(df.groupby("customer_gender")["net_sales"].sum())



                    elif dashboard_page == "Discount & Profit/Loss Analysis":

                              st.write("Discount impact and profit/loss analysis.")

                              if "discount" in df.columns:

                                        st.subheader("Sales by Discount")

                                        st.bar_chart(df.groupby("discount")["net_sales"].sum().sort_index())

                              if "profit_loss_status" in df.columns:

                                        st.subheader("Profit/Loss Status")

                                        st.bar_chart(df["profit_loss_status"].value_counts())

                              elif "profit_margin" in df.columns:

                                        st.subheader("Profit Margin Distribution")

                                        st.bar_chart(df["profit_margin"].value_counts().sort_index())



          elif dashboard_page == "SQL & Technical Analysis":

                    st.markdown(

                              """

                              - **Database:** PostgreSQL

                              - **Data Processing:** Pandas

                              - **Machine Learning:** LightGBM and Scikit-learn

                              - **Dashboard:** Power BI

                              - **Web Application:** Streamlit

                              """

                    )



          elif dashboard_page == "Future Sales Forecast":

                    show_sales_forecast()



          elif dashboard_page == "Return Risk Analysis":

                    st.write("Return risk predictions and evaluation.")

                    results_path = (

                              PROJECT_ROOT / "module5" / "powerbi"

                              / "return_risk_threshold_predictions.csv"

                    )

                    if not results_path.exists():

                              st.warning("Return-risk prediction CSV not found. Generate predictions first.")

                              return



                    results = pd.read_csv(results_path)

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("Total Test Orders", f"{len(results):,}")

                    actual_returned = (results["actual_returned"] == "Yes").sum()

                    col2.metric("Actual Returned Orders", f"{actual_returned:,}")

                    correct_returns = (

                              (results["actual_returned"] == "Yes")

                              & (results["predicted_returned"] == "Yes")

                    ).sum()

                    col3.metric("Correctly Detected Returns", f"{correct_returns:,}")

                    missed_returns = (

                              (results["actual_returned"] == "Yes")

                              & (results["predicted_returned"] == "No")

                    ).sum()

                    col4.metric("Missed Returns", f"{missed_returns:,}")

                    st.subheader("Predicted Return Status")

                    st.bar_chart(results["predicted_returned"].value_counts())

                    st.subheader("Actual vs Predicted Return Status")

                    st.dataframe(pd.crosstab(results["actual_returned"], results["predicted_returned"]))

                    st.subheader("Return Risk Predictions")

                    st.dataframe(results.head(100), use_container_width=True)





def show_project_information():
          st.header("Project Information")
          st.write(
                    "Future Sales Analyzer helps analyze e-commerce sales data, "
                    "understand customer behavior, study product returns, and predict future sales."
          )

          st.subheader("Project Features")
          features = [
                    (
                              "Customer Segmentation",
                              "Groups customers into segments based on their purchase behavior. "
                              "This helps understand different types of customers."
                    ),
                    (
                              "Product Return Analysis",
                              "Analyzes returned and non-returned orders to understand return patterns."
                    ),
                    (
                              "Discount Impact Analysis",
                              "Compares sales at different discount levels to understand how sales vary with discounts."
                    ),
                    (
                              "Product-wise Profit and Loss Analysis",
                              "Uses available profit and profit-margin data to help identify profitable and loss-making products or categories."
                    ),
                    (
                              "Sales Prediction",
                              "Predicts the estimated sales amount using the order details entered by the user."
                    ),
                    (
                              "Return Risk Prediction",
                              "Estimates whether an order may be returned based on the details entered by the user."
                    ),
                    (
                              "Future Sales Forecasting",
                              "Uses historical monthly sales data and LightGBM to estimate sales for the next six months."
                    ),
                    (
                              "Sales Analysis Dashboard",
                              "Displays sales trends and comparisons through charts and tables."
                    ),
                    (
                              "SQL and Technical Analysis",
                              "Shows the main technologies used in the project, including PostgreSQL, Pandas, LightGBM, Scikit-learn, Power BI, and Streamlit."
                    ),
          ]

          for row_start in range(0, len(features), 3):
                    columns = st.columns(3)
                    for column, (feature_name, description) in zip(columns, features[row_start:row_start + 3]):
                              with column:
                                        with st.container(border=True):
                                                  st.markdown(f"#### {feature_name}")
                                                  st.write(description)



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

                    "Project Information",

          ],

)



# --------------------------------------------------

# 4. PAGE ROUTING

# --------------------------------------------------

if page == "Home":

          show_home()

elif page == "Sales Prediction":

          show_sales_prediction()

elif page == "Return Risk Prediction":

          show_return_risk_prediction()

elif page == "Sales Forecast":

          show_sales_forecast()

elif page == "Power BI Dashboard":

          show_power_bi_dashboard()

elif page == "Project Information":

          show_project_information()
