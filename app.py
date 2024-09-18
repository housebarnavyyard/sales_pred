from flask import Flask, render_template
import pandas as pd
from nixtla import NixtlaClient
from google.cloud import storage
import io

app = Flask(__name__)

# Initialize Nixtla Client with your API key
nixtla_client = NixtlaClient(api_key='nixtla-tok-u8xxrv1Z85ERiGue41KoOsnhyQ7zHjcZi9fEV23cYLixX8iToWIUnYiXLcxcEsFX0kfazAyOKaijLQ78')

def download_csv_from_gcs(bucket_name, file_name):
    """Download a CSV file from Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_bytes()
    return io.BytesIO(data)

@app.route('/')
def index():
    # Read the sales data
    file_path = '/path/to/your/csv/file.csv'  # Adjust path
    df = pd.read_csv(file_path)

    # Preprocessing data
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop(columns=['location_name']).drop_duplicates(subset=['Date']).set_index('Date').resample('D').ffill()

    # Generate forecast for Gross Sales
    forecast_gross_sales = nixtla_client.forecast(
        df=df.reset_index(),
        h=30,
        time_col='Date',
        target_col='Gross_Sales',
        finetune_steps=10,
        finetune_loss="mae"
    )

    # Print or log the forecast data to check
    print(forecast_gross_sales)

    # Prepare data for Chart.js
    forecast_gross_sales_df = forecast_gross_sales[['Date', 'TimeGPT']]
    forecast_data = forecast_gross_sales_df.to_dict(orient='records')

    # Pass data to the front-end
    return render_template('index.html', forecast_data=forecast_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
