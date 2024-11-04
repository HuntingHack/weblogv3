import pandas as pd
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
from trubrics.integrations.streamlit import FeedbackCollector

# Set Streamlit page layout
st.set_page_config(page_title='Web Log Analysis App', layout='wide', page_icon=":fax:",)

# Function to style text
def styled_text(text, font_size=18, color='black', weight='normal', align='left'):
    return f'<p style="font-size:{font_size}px; color:{color}; font-weight:{weight}; text-align:{align};">{text}</p>'

# Function to create colorful boxes
def colored_box(title, content, color):
    return f'<div style="background-color:{color}; padding: 10px; border-radius: 5px;"><h2>{title}</h2><p>{content}</p></div>'

# Function to create expandable sections
def expandable_section(title, content):
    return st.expander(title, expanded=True).write(content)

# Code for website design
st.title('Web Log Analysis App')
st.subheader('By Siddharth Madhavan')

# Disclaimer
with st.expander('Disclaimer', expanded=True):
    st.markdown('[T&Cs](https://docs.google.com/document/d/1ej4FDVM_NPhB3BDbsRZ4ygkt738de62WUlap42gy4ZY/edit?usp=sharing)')
    st.markdown('[Privacy Policy](https://docs.google.com/document/d/1d5BDiJyufvRkGjB0BjwW3PGpdCEUKi2MYFkL4B1UKe0/edit?usp=sharing)')
    button1 = st.radio("Please agree to continue:", ('Agree', 'Disagree'))

if (button1 == 'Agree'):
    st.expander('Disclaimer', expanded=False)
    st.header('About App')
    st.write('The Web Log Analysis App is a powerful tool designed to analyze and gain insights from web server logs. It provides valuable information about website traffic, user behavior, errors, and performance. With this app, you can make data-driven decisions to optimize your website, improve user experience, and enhance security.')

    # Starting off with an input given by the user
    uploaded_file = st.file_uploader("Upload your file here...")

    if uploaded_file is not None:
        log_data = pd.read_csv(uploaded_file)

        # Perform initial data exploration
        st.write(log_data.head())  # Display the first few rows of the dataset
        #st.write(log_data.info())  # Get information about the dataset

        # Create new columns for day, month, year, and time
        log_data['Date'] = log_data['Time'].str.extract(r'\[(\d{2}/\w+/\d{4})')
        log_data['Day'] = log_data['Date'].str.extract(r'(\d{2})/')
        log_data['Month'] = log_data['Date'].str.extract(r'/(\w+)/')
        log_data['Year'] = log_data['Date'].str.extract(r'/(\d{4})')
        log_data['Time'] = log_data['Time'].str.extract(r':(\d{2}:\d{2}:\d{2})')
        log_data['URL'] = log_data['URL'].str.extract(r'(\S+)\sHTTP/1\.1')

        page_views = log_data['URL'].value_counts()  # Count of page views for each URL

        # Perform analysis based on available columns
        unique_ips = log_data['IP'].nunique()  # Count the number of unique IP addresses
        unique_urls = log_data['URL'].nunique()  # Count the number of unique URLs
        status_counts = log_data['Staus'].value_counts()  # Count the occurrences of each status code
        # Perform exploratory analysis
        total_requests = len(log_data)  # Total number of requests
        unique_visitors = log_data['IP'].nunique()  # Number of unique IP addresses


        # Exploratory Analysis Results Styling
        st.header("Exploratory Analysis Results:")

        st.write(styled_text("Analysis Results:", font_size=24, weight='bold', color='white', align='center'), unsafe_allow_html=True)

        col1, col2, col3,col4 = st.columns(4)
        with col1:
            st.write(colored_box("Unique IP Adds:", str(unique_ips), 'blue'), unsafe_allow_html=True)
        with col2:
            st.write(colored_box("Unique URLs:", str(unique_urls), 'green'), unsafe_allow_html=True)
        with col3:
            st.write(colored_box("Total Requests:", str(total_requests), 'orange'), unsafe_allow_html=True)
        with col4:
            st.write(colored_box("Unique Vistors:", str(unique_visitors), 'red'), unsafe_allow_html=True)
            
        st.subheader("Additional Informations:")
    
        st.write(colored_box("Status Code Counts:", "", 'lightgreen'), unsafe_allow_html=True)
        st.table(status_counts)

        st.write(colored_box("Page Views:", "", 'lightgreen'), unsafe_allow_html=True)
        st.table(page_views.head(20))

        # Data Visualization
        st.header('Data Visualization:')
        st.subheader("Page Views Frequency:")
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
        #differenced_data = target_variable.diff().dropna()

        # Step 4: Determining ARIMA Parameters (For DEV use ONLY)
        # Plot the autocorrelation and partial autocorrelation functions

        #fig, ax = plt.subplots(figsize=(12, 6))
        #plot_acf(differenced_data, ax=ax, lags=lags)
        #plt.xlabel('Lags')
        #plt.ylabel('Autocorrelation')
        #plt.title('Autocorrelation Function')
        #plt.show()

        #fig, ax = plt.subplots(figsize=(12, 6))
        #plot_pacf(differenced_data, ax=ax, lags=25)
        #plt.xlabel('Lags')
        #plt.ylabel('Partial Autocorrelation')
        #plt.title('Partial Autocorrelation Function')
        #plt.show()

        # Determine the order (p, d, q) of the ARIMA model based on the plots and ADF test

        # Step 5: Split the Data
        train_data = target_variable[:-100]  # Use all but the last 100 data points for training
        test_data = target_variable[-100:]  # Use the last 100 data points for testing

        # Step 6: Fit the ARIMA Model

        order = (2, 3, 1)
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

        st.subheader("Forecasting:")

        inpu2 = st.text_input("Enter the required number of predictions:")

        if inpu2:
            pred_no = inpu2.title()

            future_predictions = model_fit.forecast(steps=int(pred_no))
            st.subheader('Predictions:')

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

            st.write("In conclusion, the Web Log Analysis App offers an invaluable resource for website owners and administrators to delve into their web server logs and extract meaningful insights. By harnessing data-driven decisions derived from this analysis, website owners can effectively optimize their online platforms, elevate user experiences, and bolster the overall security of their websites. The app's intuitive interface and powerful visualizations enable users to effortlessly identify patterns, track trends, and make informed decisions to enhance the performance and user engagement of their web presence. With the Web Log Analysis App at their disposal, website administrators are empowered to take their online ventures to new heights of success and efficiency.")

            
        st.subheader("Feedback:")
        collector = FeedbackCollector(
            component_name="feedback",
            email=st.secrets["TRUBRICS_EMAIL"], 
            password=st.secrets["TRUBRICS_PASSWORD"], 
        )

        collector.st_feedback(
            feedback_type="faces",
            model="your_model_name",
            open_feedback_label="Please let us know about your suggestions!",
        )
          
    else:
        st.warning('Please upload a file to proceed.')
else:
    st.warning('Please agree to use our service!')
