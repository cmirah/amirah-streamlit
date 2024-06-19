import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="ANN Infectious Prediction", page_icon="📊")
st.title("Prediction of COVID-19 Infectious using ANN")

# Function to train ANN model
def train_ann_model(df, features, target):
    X = df[features].values
    y = df[target].values
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build ANN model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # Train model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Evaluate model
    train_loss = model.evaluate(X_train_scaled, y_train, verbose=0)
    test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    st.write(f"Train Loss: {train_loss:.4f}")
    st.write(f"Test Loss: {test_loss:.4f}")
    
    return model, scaler

# Function to predict using trained ANN model
def predict_ann_value(model, scaler, inputs):
    # Scale the input data
    inputs_scaled = scaler.transform([inputs])
    
    # Perform prediction
    predicted_value = model.predict(inputs_scaled)
    return predicted_value[0][0]

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
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.date

    # Display the full DataFrame
    st.write("Full Dataset:")
    st.write(df)

    # Drop rows with None values
    df = df.dropna()

    # Sort data by date
    df = df.sort_values('date')

    # Train ANN models
    targets = ['infected']
    features = ['susceptible', 'recovered', 'fatal', 'confirmed']
    
    models = {}
    scalers = {}
    
    for target in targets:
        models[target], scalers[target] = train_ann_model(df, features, target)

    # Ensure prediction_date is a datetime object
    prediction_date = pd.to_datetime(prediction_date).date()

    # Get the latest data before the prediction date
    latest_data = df[df['date'] < prediction_date]
    
    if latest_data.empty:
        st.warning("No data available before the selected prediction date. Please choose a different date.")
        return

    latest_data = latest_data.iloc[-1]
    
    # Predict values
    inputs = [latest_data['susceptible'], latest_data['recovered'], latest_data['fatal'], latest_data['confirmed']]
    predicted_infected = predict_ann_value(models['infected'], scalers['infected'], inputs)
    
    # Display prediction
    st.success(f"Predicted Infected value on {prediction_date} is : {predicted_infected:.0f}")

if __name__ == '__main__':
    main()


