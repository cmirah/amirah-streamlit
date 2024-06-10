import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

st.set_page_config(page_title="SIRF Prediction", page_icon="📊")
st.title("Prediction of COVID-19 Disease using SIRF Model")

def train_model(df, features, target):
    # Feature engineering and preprocessing
    X = df[features]
    y = df[target]
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train model
    model = RandomForestRegressor()
    model.fit(X_scaled, y)
    return model, scaler

def predict_value(model, scaler, inputs):
    # Scale the input data
    new_data = np.array([inputs])
    new_data_scaled = scaler.transform(new_data)
    # Perform prediction
    predicted_value = model.predict(new_data_scaled)
    return predicted_value[0]

def main():
    # Input form for new data
    prediction_date = st.date_input('Prediction Date')

    # Read the CSV file
    file_path = 'cases_malaysia.csv'
    df = pd.read_csv(file_path)
    
    # Check for None values in the DataFrame
    if df.isnull().values.any():
        st.warning("The DataFrame contains None or NaN values. Please check your CSV file and make sure it is properly formatted.")
        st.write(df[df.isnull().any(axis=1)])

    # Convert date column to datetime and remove the time component
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce').dt.date

    # Display the full DataFrame
    st.write("Full Dataset:")
    st.write(df)

    # Drop rows with None values
    df = df.dropna()

    # Sort data by date
    df = df.sort_values('date')

    # Train models
    models = {}
    scalers = {}
    targets = ['susceptible', 'infected', 'recovered', 'fatal']
    features_map = {
        'susceptible': ['infected', 'recovered', 'fatal', 'confirmed'],
        'infected': ['susceptible', 'recovered', 'fatal', 'confirmed'],
        'recovered': ['susceptible', 'infected', 'fatal', 'confirmed'],
        'fatal': ['susceptible', 'infected', 'recovered', 'confirmed']
    }
    
    for target in targets:
        models[target], scalers[target] = train_model(df, features_map[target], target)

    # Ensure prediction_date is a datetime object
    prediction_date = pd.to_datetime(prediction_date).date()

    # Get the latest data before the prediction date
    latest_data = df[df['date'] < prediction_date]
    
    if latest_data.empty:
        st.warning("No data available before the selected prediction date. Please choose a different date.")
        return

    latest_data = latest_data.iloc[-1]
    
    # Predict values
    inputs_map = {
        'susceptible': [latest_data['infected'], latest_data['recovered'], latest_data['fatal'], latest_data['confirmed']],
        'infected': [latest_data['susceptible'], latest_data['recovered'], latest_data['fatal'], latest_data['confirmed']],
        'recovered': [latest_data['susceptible'], latest_data['infected'], latest_data['fatal'], latest_data['confirmed']],
        'fatal': [latest_data['susceptible'], latest_data['infected'], latest_data['recovered'], latest_data['confirmed']]
    }

    if st.button('Predict SIRF'):
        predictions = {}
        for target in targets:
            inputs = inputs_map[target]
            predictions[target] = predict_value(models[target], scalers[target], inputs)
        
        st.success(f"Predicted values on {prediction_date} are:")
        st.write(f"Susceptible: {predictions['susceptible']:.0f}")
        st.write(f"Infected: {predictions['infected']:.0f}")
        st.write(f"Recovered: {predictions['recovered']:.0f}")
        st.write(f"Fatal: {predictions['fatal']:.0f}")

        # Plotting the SIRF curves over time
        plot_data = {
            'Date': [],
            'Susceptible': [],
            'Infected': [],
            'Recovered': [],
            'Fatal': []
        }

        # Populate plot_data for each date up to prediction_date
        current_date = latest_data['date']
        while current_date <= prediction_date:
            plot_data['Date'].append(current_date)
            plot_data['Susceptible'].append(inputs_map['susceptible'][0])
            plot_data['Infected'].append(inputs_map['infected'][0])
            plot_data['Recovered'].append(inputs_map['recovered'][0])
            plot_data['Fatal'].append(inputs_map['fatal'][0])

            # Update inputs_map for the next day's prediction
            inputs_map['susceptible'] = [predictions['susceptible'], predictions['recovered'], predictions['fatal'], predictions['confirmed']]
            inputs_map['infected'] = [predictions['infected'], predictions['recovered'], predictions['fatal'], predictions['confirmed']]
            inputs_map['recovered'] = [predictions['susceptible'], predictions['infected'], predictions['fatal'], predictions['confirmed']]
            inputs_map['fatal'] = [predictions['susceptible'], predictions['infected'], predictions['recovered'], predictions['confirmed']]

            # Predict for the next day
            for target in targets:
                predictions[target] = predict_value(models[target], scalers[target], inputs_map[target])

            current_date += pd.Timedelta(days=1)

        # Create Plotly figure
        fig = go.Figure()

        # Add traces for each category
        fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Susceptible'], mode='lines', name='Susceptible'))
        fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Infected'], mode='lines', name='Infected'))
        fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Recovered'], mode='lines', name='Recovered'))
        fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Fatal'], mode='lines', name='Fatal'))

        # Update figure layout
        fig.update_layout(
            title='SIRF Model Prediction',
            xaxis_title='Date',
            yaxis_title='Population',
            template='plotly_white'
        )

        # Display the plot
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()



