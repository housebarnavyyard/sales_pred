import streamlit as st
import pandas as pd
from nixtla import NixtlaClient
import plotly.express as px

# Initialize Nixtla Client with your API key
nixtla_client = NixtlaClient(api_key='nixtla-tok-u8xxrv1Z85ERiGue41KoOsnhyQ7zHjcZi9fEV23cYLixX8iToWIUnYiXLcxcEsFX0kfazAyOKaijLQ78')

# Function to preprocess the data and forecast
def preprocess_and_forecast(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop(columns=['location_name']).drop_duplicates(subset=['Date']).set_index('Date').resample('D').ffill()

    # Generate forecast for Gross Sales for the next two weeks (14 days)
    forecast_gross_sales = nixtla_client.forecast(
        df=df.reset_index(),
        h=14,  # Forecast for the next two weeks
        time_col='Date',
        target_col='Gross_Sales',
        finetune_steps=10,
        finetune_loss="mae"
    )
    
    return df, forecast_gross_sales

# Read the sales data
file_path = '/Users/vishwanathmuthuraman/Desktop/sales_pred/sales prediction - 1.csv'  # Adjust with your file path
df = pd.read_csv(file_path)

# Preprocess data and generate forecast
df, forecast_gross_sales = preprocess_and_forecast(df)

# Combine past 30 days with the forecast for the next 14 days
df_last_30_days = df.tail(30)  # Last 30 days
forecast_gross_sales['Date'] = pd.to_datetime(forecast_gross_sales['Date'])
df_forecast = forecast_gross_sales[['Date', 'TimeGPT']].set_index('Date')

# Combine past 30 days and forecast
combined_data = pd.concat([df_last_30_days, df_forecast], axis=0)

# Sort data by date
combined_data_sorted = combined_data.sort_index(ascending=True)

# Display the data and chart in Streamlit
st.title("Sales Forecast with Streamlit")

st.subheader("Sales Data")
st.dataframe(combined_data_sorted, width=1000)

# Plotly interactive chart for sales forecast
fig = px.line(
    combined_data_sorted.reset_index(), 
    x='Date', 
    y='Gross_Sales', 
    title='Gross Sales Forecast for Next 14 Days',
    labels={'Gross_Sales': 'Predicted Gross Sales'},
    template='plotly_dark'
)

# Add hover functionality to display date and predicted sales
fig.update_traces(mode='lines+markers', hovertemplate='%{x}: %{y:.2f}')

# Make the x-axis more interactive by adding a range slider and zooming features
fig.update_xaxes(rangeslider_visible=True, showspikes=True)
fig.update_yaxes(showspikes=True)

# Add a mode for zooming, panning, and interacting with the chart
fig.update_layout(
    hovermode='x',
    spikedistance=-1,
    xaxis_title="Date",
    yaxis_title="Predicted Gross Sales",
    xaxis_showgrid=False,
    yaxis_showgrid=False,
)

# Display the Plotly chart in Streamlit
st.plotly_chart(fig)

# Filter the forecast data to show the next two weeks in a table
st.subheader("14-Day Forecast Data")
st.dataframe(forecast_gross_sales[['Date', 'TimeGPT']], width=1000)
