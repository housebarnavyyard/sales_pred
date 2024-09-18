from flask import Flask, render_template
import pandas as pd
from nixtla import NixtlaClient

app = Flask(__name__)

# Initialize Nixtla Client with your API key
nixtla_client = NixtlaClient(api_key='nixtla-tok-u8xxrv1Z85ERiGue41KoOsnhyQ7zHjcZi9fEV23cYLixX8iToWIUnYiXLcxcEsFX0kfazAyOKaijLQ78')

@app.route('/')
def index():
    # Read the sales data
    file_path = '/Users/vishwanathmuthuraman/Desktop/HB App/sales prediction - 1.csv'  # Adjust path
    df = pd.read_csv(file_path)

    # Convert Date column to datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Remove non-numerical columns
    df = df.drop(columns=['location_name'])
    df = df.drop_duplicates(subset=['Date'])
    df.set_index('Date', inplace=True)
    
    # Resample to daily frequency
    df_resampled = df.resample('D').ffill()

    # Generate forecast for Gross Sales
    forecast_gross_sales = nixtla_client.forecast(
        df=df_resampled.reset_index(),
        h=30,
        time_col='Date',
        target_col='Gross_Sales',
        finetune_steps=10,
        finetune_loss="mae"
    )
    
    # Prepare data for Chart.js
    forecast_gross_sales_df = forecast_gross_sales[['Date', 'TimeGPT']]
    forecast_data = forecast_gross_sales_df.to_dict(orient='records')
    
    # Pass data to the front-end
    return render_template('index.html', forecast_data=forecast_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
