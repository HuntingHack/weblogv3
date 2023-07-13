import pandas as pd
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Code for website design

st.title('Web Log Analysis App')
st.subheader('By Siddharth Madhavan')

st.write('The Web Log Analysis App is a powerful tool designed to analyze and gain insights from web server logs. It provides valuable information about website traffic, user behavior, errors, and performance. With this app, you can make data-driven decisions to optimize your website, improve user experience, and enhance security.')
# starting off with an input given by the user
try:
  uploaded_file = st.file_uploader("Upload your file here...")

  log_data = pd.read_csv(uploaded_file)
except Exception:
 pass

st.write(log_data)

# Perform initial data exploration
print(log_data.head())  # Display the first few rows of the dataset
print(log_data.info())  # Get information about the dataset

# Create new columns for day, month, year, and time
log_data['Date'] = log_data['Time'].str.extract(r'\[(\d{2}/\w+/\d{4})')
log_data['Day'] = log_data['Date'].str.extract(r'(\d{2})/')
log_data['Month'] = log_data['Date'].str.extract(r'/(\w+)/')
log_data['Year'] = log_data['Date'].str.extract(r'/(\d{4})')
log_data['Time'] = log_data['Time'].str.extract(r':(\d{2}:\d{2}:\d{2})')
log_data['URL'] = log_data['URL'].str.extract(r'(\S+)\sHTTP/1\.1')


page_views = log_data['URL'].value_counts()  # Count of page views for each URL

# Data Visualization

st.subheader('Data Visualization:')
fig = px.line(page_views, title='Page Views')
st.plotly_chart(fig)

aggregated_data = log_data.groupby('Date').agg({'IP': 'count'}).reset_index()
aggregated_data.columns = ['Date', 'request_count']

aggregated_data.set_index('Date', inplace=True)
target_variable = aggregated_data['request_count']

target_variable.plot(figsize=(12, 6))
plt.xlabel('Timestamp')
plt.ylabel('Total Requests')
plt.title('Web Log Data')
plt.show()

# Step 3: Make the Time Series Stationary
# Apply differencing to remove trend and seasonality
differenced_data = target_variable.diff().dropna()

# Step 4: Determine ARIMA Parameters
# Plot the autocorrelation and partial autocorrelation functions
fig, ax = plt.subplots(figsize=(12, 6))
plot_acf(differenced_data, ax=ax, lags=30)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function')
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
plot_pacf(differenced_data, ax=ax, lags=25)
plt.xlabel('Lags')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function')
plt.show()

# Determine the order (p, d, q) of the ARIMA model based on the plots and ADF test

# Step 5: Split the Data
train_data = differenced_data[:-30]  # Use all but the last 30 data points for training
test_data = differenced_data[-30:]  # Use the last 30 data points for testing

# Step 6: Fit the ARIMA Model


order = (2, 3, 1)  # Example order, replace with the appropriate values
model = ARIMA(train_data, order=order)
model_fit = model.fit()
# Step 8: Evaluate the Model
predictions = model_fit.forecast(steps=len(test_data))
mse = mean_squared_error(test_data, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data, predictions)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Step 9: Forecast with the Model

inpu2 = st.text_input("Enter the required number of predictions:")

if st.button('Submit'):
 pred_no = inpu2.title()

future_predictions = model_fit.forecast(steps=int(pred_no))  # Example: Generate 10 future predictions
st.subheader('Predictions:')
st.success(future_predictions)

# Step 5: Create a Line Plot with Forecasted Predictions
fig = go.Figure()

# Add the historical data to the plot
fig.add_trace(go.Scatter(
    x=target_variable.index,
    y=target_variable,
    name='Historical Data'
))

# Add the forecasted predictions to the plot
fig.add_trace(go.Scatter(
    x=future_predictions.index,
    y=future_predictions,
    name='Forecasted Predictions'
))

# Customize the plot layout
fig.update_layout(
    title='Web Log Data Forecast',
    xaxis_title='Timestamp',
    yaxis_title='Total Requests'
)

# Show the plot
st.plotly_chart(fig)


# Perform analysis based on available columns
unique_ips = log_data['IP'].nunique()  # Count the number of unique IP addresses
unique_urls = log_data['URL'].nunique()  # Count the number of unique URLs
status_counts = log_data['Staus'].value_counts()  # Count the occurrences of each status code

st.header('Additional Information:')

# Print the results
st.subheader("Analysis Results:")
st.text("Unique IP Addresses:")
st.text(unique_ips)
st.text("Unique URLs:")
st.text(unique_urls)
st.text("Status Code Counts:")
st.text(status_counts)

# Perform exploratory analysis
total_requests = len(log_data)  # Total number of requests
unique_visitors = log_data['IP'].nunique()  # Number of unique IP addresses


st.subheader("Exploratory Analysis Results:")
st.text("Total Requests:")
st.text(total_requests)
st.text("Unique Visitors:")
st.text(unique_visitors)
st.text("Page Views:")
st.text(page_views.head(20))
