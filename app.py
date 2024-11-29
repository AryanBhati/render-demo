#the flask code for the app
#make changes accordingly

from flask import Flask, render_template, request

import pickle
import pandas as pd
import matplotlib.pyplot as plt

import io
import base64
import numpy as np
from statsmodels.tsa.arima.model import ARIMA



app = Flask(__name__)


# Load the ARIMA model from the pickle file
with open('arima_model.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)




#data here is the clean data used in arima model we put manually here for these countries 
historical_data = {
    'USA': [2.0, 2.3, 2.6, 2.9, 3.1, 3.1, 3.0, 3.6, 3.7, 3.9],
    'India': [3.5, 4.1, 4.2, 4.5, 4.7, 4.9, 5.5, 5.3, 5.5, 5.8],
    'China': [2.0, 2.1, 2.9, 2.7, 3.6, 3.8, 4.0, 3.8, 3.7, 4.0],
    'Canada': [3.5, 3.6, 4.3, 4.1, 4.6, 4.4, 4.5, 4.9, 5.3, 5.1],
    'Russia': [6.0, 6.2, 6.4, 6.7, 6.9, 7.1, 7.3, 7.5, 7.7, 8.0],
    'Germany': [1.8, 2.0, 2.2, 2.3, 2.5, 2.7, 2.8, 3.0, 3.2, 3.3],
    'Japan': [1.9, 1.7, 1.7, 1.8, 2.4, 2.5, 2.9, 3.2, 3.3, 3.5],
    'Australia': [2.2, 2.4, 2.5, 2.7, 2.8, 3.0, 3.2, 3.3, 3.5, 3.7],
}

# List of countries for dropdown in HTML
countries = list(historical_data.keys())

# Route for the home page with dropdown of countries
@app.route('/')
def index():
    return render_template('index.html', countries=countries)

# Route to handle prediction and graph generation
@app.route('/predict', methods=['POST'])
def predict():
    selected_country = request.form.get('country')  # Get selected country
    steps = 2  # Forecast for 2 years ahead

    # Get the historical data for the selected country
    time_series_data = historical_data[selected_country]
    
    # Fit the ARIMA model with the historical data for the selected country
    years = list(range(1970, 2023))
    model_fitted = ARIMA(time_series_data, order=(1, 1, 1))  # Adjust ARIMA parameters as needed
    model_fitted = model_fitted.fit()

    # Forecasting using the model
    forecast_values = model_fitted.predict(start=len(time_series_data), end=len(time_series_data) + steps - 1)  # Predict the next 2 steps ahead
    
    # Convert the forecasted values to a list
    forecast_list = forecast_values.tolist()
    forecast_years = list(range(2024, 2026))
    
    # Combine historical years and forecast years for the x-axis
    plot_years = years + forecast_years
    
    # Combine historical data and forecast data for the plot
    plot_data = time_series_data + forecast_values.tolist()

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(time_series_data, label='Historical Data', color='blue')  # Historical data plot
    plt.plot(range(len(time_series_data), len(time_series_data) + steps), forecast_values, label='Forecasted Data', color='red')  # Forecast plot
    plt.title(f'Inflation Forecast for {selected_country} - Next {steps} Years')
    plt.xlabel('Year')
    plt.ylabel('Inflation Rate')
    plt.legend()

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Encode the image as a base64 string
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    # Return the template with the graph URL and forecast values
    return render_template('index.html', countries=countries, forecast=forecast_list, country=selected_country, steps=steps, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
