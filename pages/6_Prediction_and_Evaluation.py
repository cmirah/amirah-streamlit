import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Evaluation Performance", page_icon="📜")

# Title and Image
st.title("Prediction & Evaluation")
st.image("wb2.webp", width=500)

# Function to train and evaluate the model
def train_model(df, features, target):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor()
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    evaluation = {
        'MAE Train': mean_absolute_error(y_train, y_pred_train),
        'MSE Train': mean_squared_error(y_train, y_pred_train),
        'R² Train': r2_score(y_train, y_pred_train),
        'MAE Test': mean_absolute_error(y_test, y_pred_test),
        'MSE Test': mean_squared_error(y_test, y_pred_test),
        'R² Test': r2_score(y_test, y_pred_test)
    }

    return model, scaler, evaluation, y_train, y_test, y_pred_train, y_pred_test

# Function to predict using the trained model
def predict(model, scaler, input_data):
    input_data_scaled = scaler.transform(np.array([input_data]))
    prediction = model.predict(input_data_scaled)
    return prediction[0]

def main():
    # Read the CSV file
    file_path = 'cases_malaysia.csv'
    df = pd.read_csv(file_path)

    # Initial input values for [S, I, R, F]
    initial_values = [10000000, 1000, 1000, 0]

    # Sections for each prediction (S, I, R, F)
    sections = {
        'Susceptible': {'features': ['susceptible','infected', 'recovered', 'fatal'], 'target': 'susceptible', 'initial_values': initial_values},
        'Infected': {'features': ['susceptible', 'infected','recovered', 'fatal'], 'target': 'infected', 'initial_values': initial_values},
        'Recovered': {'features': ['susceptible', 'infected','recovered', 'fatal'], 'target': 'recovered', 'initial_values': initial_values},
        'Fatal': {'features': ['susceptible', 'infected', 'recovered','fatal'], 'target': 'fatal', 'initial_values': initial_values}
    }

    # Sidebar headers for each S-I-R-F model
    selected_section = st.sidebar.selectbox("Select Prediction Model", list(sections.keys()), key="model_selection")

    for section, params in sections.items():
        expander = st.sidebar.expander(f"{section} Model", expanded=(section == selected_section))

        with expander:
            # Plotting the loss function graph (MSE) for train and test sets
            st.subheader(f"{section} Loss Function")
            model, scaler, evaluation, y_train, y_test, y_pred_train, y_pred_test = train_model(df, params['features'], params['target'])
            fig, ax = plt.subplots()
            ax.plot(y_train, y_train - y_pred_train, 'o', label='Train', alpha=0.6)
            ax.plot(y_test, y_test - y_pred_test, 'o', label='Test', alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='-')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f"Loss Function (Residuals)")
            ax.legend()
            st.pyplot(fig)

            # Plotting the predicted vs actual values as a line graph
            st.subheader(f"Predicted vs Actual")
            fig, ax = plt.subplots()
            ax.plot(y_test.values, label='Actual', alpha=0.6)
            ax.plot(y_pred_test, label='Predicted', alpha=0.6)
            ax.set_xlabel('Index')
            ax.set_ylabel(f'{section} Values')
            ax.set_title(f"{section} Predicted vs Actual")
            ax.legend()
            st.pyplot(fig)

    # Main panel for prediction and evaluation
    for section, params in sections.items():
        if selected_section == section:
            model, scaler, evaluation, y_train, y_test, y_pred_train, y_pred_test = train_model(df, params['features'], params['target'])

            # Input form for prediction
            st.subheader(f"{section} Prediction")

            inputs = [st.number_input(f'{feature}', key=f"{section}_{feature}", value=params['initial_values'][idx]) for idx, feature in enumerate(params['features'])]

            if st.button(f'Predict {section}'):
                prediction = predict(model, scaler, inputs)
                st.success(f'Predicted {section} value is : {prediction:.0f}')

            # Display evaluation metrics
            st.subheader(f"{section} Evaluation")
            evaluation_df = pd.DataFrame(evaluation.items(), columns=['Metric', 'Value'])
            st.table(evaluation_df)

if __name__ == '__main__':
    main()





