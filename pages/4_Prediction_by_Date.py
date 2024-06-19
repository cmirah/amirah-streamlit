import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(page_title="ANN Infectious Prediction", page_icon="📊")
st.title("Prediction of COVID-19 Infectious using ANN (PyTorch)")

# Define the neural network model
class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Initialize the model, loss function, and optimizer
    model = ANNModel(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(X_train_tensor), y_train_tensor).item()
        test_loss = criterion(model(X_test_tensor), y_test_tensor).item()
    
    st.write(f"Train Loss: {train_loss:.4f}")
    st.write(f"Test Loss: {test_loss:.4f}")
    
    return model, scaler

# Function to predict using trained ANN model
def predict_ann_value(model, scaler, inputs):
    # Scale the input data
    inputs_scaled = scaler.transform([inputs])
    
    # Convert data to PyTorch tensor
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
    
    # Perform prediction
    model.eval()
    with torch.no_grad():
        predicted_value = model(inputs_tensor).item()
    
    return predicted_value

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


