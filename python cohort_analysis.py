# ============================
# COHORT & SALES ANALYSIS SCRIPT
# ============================

# ---- 1. Imports ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("Set2")

# ---- 2. Load data ----
file_path = r"C:\Users\Rishit\OneDrive\Desktop\sales.csv"

df = pd.read_csv(file_path)

print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---- 3. Basic cleaning & type conversion ----

# Standardize column names (strip spaces, lower-case)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Expected important columns:
# order_id, order_date, status, item_id, sku, qty_ordered, price, value,
# discount_amount, total, category, payment_method, cust_id, year, month, ...
# Adjust if your actual names differ after cleaning
print("Standardized columns:", df.columns.tolist())

# Convert order_date to datetime
# Try multiple formats, given examples like 01/10/2020 and 13/11/2020
def parse_date(col):
    return pd.to_datetime(col, errors="coerce", dayfirst=True)

df["order_date"] = parse_date(df["order_date"])

# Remove rows with invalid dates
df = df[~df["order_date"].isna()].copy()

# Numeric conversions
numeric_cols = ["qty_ordered", "price", "value", "discount_amount", "total", "age", "discount_percent"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Basic sanity cleaning: drop rows with no customer or order id
df = df.dropna(subset=["order_id", "cust_id"])

# Ensure order_id and cust_id are strings
df["order_id"] = df["order_id"].astype(str)
df["cust_id"] = df["cust_id"].astype(str)

print("Cleaned shape:", df.shape)

# ---- 4. Create time features ----

df["order_year"] = df["order_date"].dt.year
df["order_month"] = df["order_date"].dt.to_period("M")  # e.g., 2020-10
df["order_month_start"] = df["order_month"].dt.to_timestamp()

# ---- 5. Basic business KPIs ----

# Overall metrics
total_revenue = df["total"].sum()
total_orders = df["order_id"].nunique()
total_customers = df["cust_id"].nunique()

print("\n=== BASIC KPIs ===")
print(f"Total revenue: {total_revenue:,.2f}")
print(f"Total orders: {total_orders}")
print(f"Total customers: {total_customers}")

# Revenue by month
revenue_by_month = df.groupby("order_month_start")["total"].sum().reset_index()

# Orders by month
orders_by_month = df.groupby("order_month_start")["order_id"].nunique().reset_index(name="orders")

# Revenue by category
if "category" in df.columns:
    revenue_by_category = df.groupby("category")["total"].sum().sort_values(ascending=False)
    print("\nRevenue by category:")
    print(revenue_by_category)

# Revenue by payment method
if "payment_method" in df.columns:
    revenue_by_payment = df.groupby("payment_method")["total"].sum().sort_values(ascending=False)
    print("\nRevenue by payment method:")
    print(revenue_by_payment)

# ---- 6. Cohort analysis setup ----

# 6.1 Find each customer's first order month (cohort_month)
customer_first_order = (
    df.groupby("cust_id")["order_month_start"]
    .min()
    .reset_index()
    .rename(columns={"order_month_start": "cohort_month"})
)

df = df.merge(customer_first_order, on="cust_id", how="left")

# 6.2 Define cohort index (number of months since cohort_month)
def get_cohort_index(order_month, cohort_month):
    return (order_month.year - cohort_month.year) * 12 + (order_month.month - cohort_month.month) + 1

df["cohort_index"] = df.apply(
    lambda x: get_cohort_index(x["order_month_start"], x["cohort_month"]), axis=1
)

# ---- 7. Cohort: customer retention (count of unique customers) ----

cohort_data = (
    df.groupby(["cohort_month", "cohort_index"])["cust_id"]
    .nunique()
    .reset_index()
    .rename(columns={"cust_id": "num_customers"})
)

cohort_pivot = cohort_data.pivot_table(
    index="cohort_month",
    columns="cohort_index",
    values="num_customers"
)

cohort_size = cohort_pivot.iloc[:, 0]  # number of customers in month 1 by cohort
cohort_retention = cohort_pivot.divide(cohort_size, axis=0)

print("\n=== COHORT SIZE (first month customers per cohort) ===")
print(cohort_size)

print("\n=== COHORT RETENTION TABLE (fraction of customers retained) ===")
print(cohort_retention.round(3))

# ---- 8. Cohort: revenue per cohort per period ----

cohort_revenue = (
    df.groupby(["cohort_month", "cohort_index"])["total"]
    .sum()
    .reset_index()
)

cohort_revenue_pivot = cohort_revenue.pivot_table(
    index="cohort_month",
    columns="cohort_index",
    values="total"
)

print("\n=== COHORT REVENUE TABLE ===")
print(cohort_revenue_pivot.round(2))

# ---- 9. Visualizations ----

# 9.1 Revenue over time
plt.figure(figsize=(10, 5))
sns.lineplot(data=revenue_by_month, x="order_month_start", y="total", marker="o")
plt.title("Monthly Revenue Over Time")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9.2 Orders over time
plt.figure(figsize=(10, 5))
sns.lineplot(data=orders_by_month, x="order_month_start", y="orders", marker="o")
plt.title("Monthly Orders Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9.3 Revenue by category (top 10)
if "category" in df.columns:
    top_cat = (
        df.groupby("category")["total"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_cat, x="category", y="total")
    plt.title("Revenue by Category (Top 10)")
    plt.xlabel("Category")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# 9.4 Revenue by payment method
if "payment_method" in df.columns:
    pay_plot = (
        df.groupby("payment_method")["total"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    plt.figure(figsize=(8, 4))
    sns.barplot(data=pay_plot, x="payment_method", y="total")
    plt.title("Revenue by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 9.5 Cohort retention heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    cohort_retention,
    annot=True,
    fmt=".0%",
    cmap="Blues"
)
plt.title("Customer Retention by Cohort (Percentage)")
plt.xlabel("Cohort Index (Months since first purchase)")
plt.ylabel("Cohort Month")
plt.tight_layout()
plt.show()

# 9.6 Cohort revenue heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    cohort_revenue_pivot,
    annot=True,
    fmt=".0f",
    cmap="Greens"
)
plt.title("Revenue by Cohort and Cohort Index")
plt.xlabel("Cohort Index (Months since first purchase)")
plt.ylabel("Cohort Month")
plt.tight_layout()
plt.show()

# ---- 10. Additional useful insights ----

# 10.1 CLV-like metric: total revenue per customer
customer_revenue = df.groupby("cust_id")["total"].sum().reset_index(name="customer_revenue")
print("\n=== TOP 10 CUSTOMERS BY REVENUE ===")
print(customer_revenue.sort_values("customer_revenue", ascending=False).head(10))

# 10.2 Number of orders per customer
customer_orders = df.groupby("cust_id")["order_id"].nunique().reset_index(name="num_orders")
print("\n=== TOP 10 CUSTOMERS BY NUMBER OF ORDERS ===")
print(customer_orders.sort_values("num_orders", ascending=False).head(10))

# 10.3 AOV (Average Order Value) overall and by month
aov_overall = df.groupby("order_id")["total"].sum().mean()
print(f"\nAverage Order Value (Overall): {aov_overall:,.2f}")

aov_by_month = (
    df.groupby("order_month_start")["total"]
    .sum()
    .div(df.groupby("order_month_start")["order_id"].nunique())
    .reset_index(name="AOV")
)

plt.figure(figsize=(10, 5))
sns.lineplot(data=aov_by_month, x="order_month_start", y="AOV", marker="o")
plt.title("Average Order Value by Month")
plt.xlabel("Month")
plt.ylabel("AOV")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 10.4 Status distribution (completed, canceled, etc.)
if "status" in df.columns:
    status_counts = df["status"].value_counts()
    print("\n=== ORDER STATUS COUNTS ===")
    print(status_counts)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=status_counts.index, y=status_counts.values)
    plt.title("Order Status Distribution")
    plt.xlabel("Status")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 10.5 Region-wise revenue (if region column exists)
if "region" in df.columns:
    region_rev = df.groupby("region")["total"].sum().sort_values(ascending=False)
    print("\n=== REVENUE BY REGION ===")
    print(region_rev)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=region_rev.index, y=region_rev.values)
    plt.title("Revenue by Region")
    plt.xlabel("Region")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()

print("\nScript finished successfully.")
