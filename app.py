import joblib
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import matplotlib.pyplot as plt

# MySQL connection parameters
MYSQL_USER = 'root'
MYSQL_PASSWORD = ''
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_DATABASE = 'customer_order'

# Establish connection using SQLAlchemy
engine = create_engine(f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}')

# Load data from MySQL tables
customers = pd.read_sql('SELECT * FROM customers', engine)
orders = pd.read_sql('SELECT * FROM orders', engine)

# Ensure customer_id is of the same type in both DataFrames
customers['customer_id'] = customers['customer_id'].astype(str)
orders['customer_id'] = orders['customer_id'].astype(str)

# Convert order_date to datetime
orders['order_date'] = pd.to_datetime(orders['order_date'])

# Sidebar filters
st.sidebar.header("Filters")

# Date range filter
start_date, end_date = st.sidebar.date_input("Select order date range:", [orders['order_date'].min(), orders['order_date'].max()])
filtered_orders = orders[(orders['order_date'] >= pd.Timestamp(start_date)) & (orders['order_date'] <= pd.Timestamp(end_date))]

# Slider to filter customers by total amount spent
min_amount = 0
max_amount = 100000
total_amount = st.sidebar.slider("Filter customers by total amount spent:", min_value=min_amount, max_value=max_amount, value=(0, max_amount))
filtered_orders = filtered_orders[(filtered_orders['total_amount'] >= total_amount[0]) & (filtered_orders['total_amount'] <= total_amount[1])]

# Dropdown to filter by number of orders
order_count_threshold = st.sidebar.number_input("Filter customers with more than X orders:", min_value=1, value=1)
customer_order_count = filtered_orders.groupby('customer_id').size().reset_index(name='order_count')
filtered_customers = customer_order_count[customer_order_count['order_count'] > order_count_threshold]

# Merge filtered orders with customer data
filtered_data = pd.merge(filtered_orders, customers, on='customer_id')

# Filter by customers who meet the order count condition
filtered_data = filtered_data[filtered_data['customer_id'].isin(filtered_customers['customer_id'])]

# Main dashboard
st.title("Customer Order Dashboard")

# Display filtered data in a table
st.subheader("Filtered Data")
st.dataframe(filtered_data)

# Bar chart for top 10 customers by total revenue
st.subheader("Top 10 Customers by Total Revenue")
top_customers = filtered_data.groupby('customer_name')['total_amount'].sum().reset_index().sort_values(by='total_amount', ascending=False).head(10)
fig, ax = plt.subplots()
ax.bar(top_customers['customer_name'], top_customers['total_amount'])
plt.xticks(rotation=45)
st.pyplot(fig)

# Line chart for total revenue over time (grouped by month)
st.subheader("Total Revenue Over Time")
filtered_data['order_month'] = filtered_data['order_date'].dt.to_period('M')
monthly_revenue = filtered_data.groupby('order_month')['total_amount'].sum().reset_index()
fig, ax = plt.subplots()
ax.plot(monthly_revenue['order_month'].astype(str), monthly_revenue['total_amount'])
plt.xticks(rotation=45)
st.pyplot(fig)

# Summary section
st.subheader("Summary")
total_revenue = filtered_data['total_amount'].sum()
unique_customers = filtered_data['customer_id'].nunique()
total_orders = filtered_data['order_id'].nunique()

st.metric("Total Revenue", f"${total_revenue:,.2f}")
st.metric("Unique Customers", unique_customers)
st.metric("Total Orders", total_orders)

# Load data from MySQL tables
customers = pd.read_sql('SELECT * FROM customers', engine)
orders = pd.read_sql('SELECT * FROM orders', engine)


# Load the model
model = joblib.load('lr_model.pkl')

# Customer input for prediction
st.sidebar.header("Customer Repeat Purchase Prediction")
customer_id_input = st.sidebar.text_input("Customer ID")
total_orders_input = st.sidebar.number_input("Total Orders", min_value=0, step=1)
total_revenue_input = st.sidebar.number_input("Total Amount", min_value=0.0, step=100.0)

# Prepare input data for prediction
if st.sidebar.button("Predict Repeat Purchase"):
    # Create DataFrame with all necessary features
    input_data = pd.DataFrame({
        'customer_id': [customer_id_input],  # Add customer_id if needed
        'total_orders': [total_orders_input],
        'total_amount': [total_revenue_input]  # Change total_revenue to total_amount
    })

    # Check if the input data matches the expected feature names
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.sidebar.success("This customer is likely to be a repeat purchaser.")
        else:
            st.sidebar.info("This customer is less likely to be a repeat purchaser.")
    except ValueError as e:
        st.sidebar.error(f"Error in prediction: {e}")



