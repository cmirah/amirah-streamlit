import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="S_Prediction", page_icon="📕")
st.title("Prediction of COVID-19 Disease using SIRF Model")
st.subheader("Susceptible Prediction")

def train_model(df):
    # Feature engineering and preprocessing
    X = df[['infected', 'recovered', 'fatal']]
    y = df['susceptible']
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train model
    model = RandomForestRegressor()
    model.fit(X_train_scaled, y_train)
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    # Evaluate the model
    evaluation = {
        'MAE Train': mean_absolute_error(y_train, y_pred_train),
        'MSE Train': mean_squared_error(y_train, y_pred_train),
        'R² Train': r2_score(y_train, y_pred_train),
        'MAE Test': mean_absolute_error(y_test, y_pred_test),
        'MSE Test': mean_squared_error(y_test, y_pred_test),
        'R² Test': r2_score(y_test, y_pred_test)
    }
    return model, scaler, evaluation, y_train, y_test, y_pred_train, y_pred_test

def predict_susceptible(model, scaler, infected, recovered, fatal):
    # Scale the input data
    new_data = np.array([[infected, recovered, fatal]])
    new_data_scaled = scaler.transform(new_data)
    # Perform prediction
    predicted_susceptible = model.predict(new_data_scaled)
    return predicted_susceptible[0]

def main():
    # Input form for new data
    infected = st.number_input('Infected', value=1000)
    recovered = st.number_input('Recovered', value=500)
    fatal = st.number_input('Fatal', value=50)
    prediction_date = st.date_input('Prediction Date')

    # Read the CSV file
    file_path = 'cases_malaysia.csv'
    df = pd.read_csv(file_path)

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # Train model and get evaluation metrics
    model, scaler, evaluation, y_train, y_test, y_pred_train, y_pred_test = train_model(df)

    # Predict button
    if st.button('Predict Susceptible'):
        # Perform prediction
        prediction = predict_susceptible(model, scaler, infected, recovered, fatal)
        # Display prediction
        st.success(f'Predicted Susceptible value on {prediction_date} is : {prediction:.0f}')

    # Display evaluation metrics
    st.subheader("Model Evaluation")
    evaluation_df = pd.DataFrame(evaluation.items(), columns=['Metric', 'Value'])
    st.table(evaluation_df)

    # Plotting the loss function graph (MSE) for train and test sets
    st.subheader("Loss Function (Residuals)")
    fig, ax = plt.subplots()
    ax.plot(y_train, y_train - y_pred_train, 'o', label='Train', alpha=0.6)
    ax.plot(y_test, y_test - y_pred_test, 'o', label='Test', alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='-')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Residuals')
    ax.set_title("Loss Function (Residuals)")
    ax.legend()
    st.pyplot(fig)

    # Plotting the predicted vs actual values as a line graph
    st.subheader("Predicted vs Actual")
    fig, ax = plt.subplots()
    ax.plot(df['date'][:len(y_test)], y_test, label='Actual', alpha=0.6)
    ax.plot(df['date'][:len(y_test)], y_pred_test, label='Predicted', alpha=0.6)
    ax.set_xlabel('Date')
    ax.set_ylabel('Susceptible Values')
    ax.set_title("Susceptible Predicted vs Actual")
    ax.legend()
    st.pyplot(fig)

if __name__ == '__main__':
    main()





