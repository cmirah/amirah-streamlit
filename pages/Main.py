import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from pyGRNN import GRNN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Title of the application
st.title("TourVis Pro: Predictive Analytics for Tourism 📊")

# Main menu bar on the sidebar
st.sidebar.title("Main Menu")

# Add menu items
menu = st.sidebar.radio(
    "Select an option",
    ("Home", "Data Overview", "Model Training", "Predictions", "About")
)

# Display content based on the selected menu item
if menu == "Home":
    st.header("Welcome to SVR & GRNN Web-based App! 👋")

    col1, col2 = st.columns([1, 2])  # Split the page into 1/4 and 3/4

    with col1:
        st.image("Robot Animation.gif", width=200)

    with col2:
        st.sidebar.success("Select a demo above.")
        st.markdown(
            """
            Welcome to our SVR & GRNN Web-Based App! Predict tourist arrivals in Malaysia with precision using 
            Support Vector Regression and Generalized Regression Neural Network technology. Our user-friendly platform 
            empowers businesses and policymakers with accurate forecasting for any selected year. Experience the future 
            of tourism prediction today!
            """
        )

elif menu == "Data Overview":
    st.header("Data Overview 📈")
    st.write("Here you can see the overview of the data used for forecasting from 2011 to 2023.")

    col1, col2 = st.columns([1, 2])  # Split the page into 1/4 and 3/4

    with col1:
        # Load and display your dataset here
        df = pd.read_excel('Malaysia-Tourism1.xlsx')
        st.dataframe(df)

    with col2:
        st.image("Data Animation.gif", width=400)

elif menu == "Model Training":
    st.header("Model Training 📉")

    def train_svr():
        st.title("1) Inbound Tourism using SVR")

        # Load data
        df = pd.read_excel('Malaysia-Tourism1.xlsx')
        st.write(df)

        # Check for null values
        st.write(df.isnull().sum())

        # Drop 'Date' column for training
        data = df.drop(['Date'], axis=1)

        # Time Series Generator
        n_input = 1
        n_output = 1
        generator = TimeseriesGenerator(data.values, data.values, length=n_input, batch_size=1)

        data_ts = pd.DataFrame(columns=['x', 'y'])
        for i in range(len(generator)):
            x, y = generator[i]
            df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
            data_ts = pd.concat([data_ts, df], ignore_index=True)

        st.write(data_ts)

        # Split Data
        data_ts[['x', 'y']] = data_ts[['x', 'y']].astype(int)
        X = np.array(data_ts['x']).reshape(-1, 1)
        Y = np.array(data_ts['y']).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Plot Actual Data
        plt.plot(Y, label='Actual Data', marker='x')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.legend()
        plt.grid(True)
        st.pyplot()

        # Scaling Dataset
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)

        # Initialize and fit the SVR model
        svr_model = SVR(kernel='rbf')
        svr_model.fit(X_train_scaled, y_train_scaled.ravel())

        # Predict and evaluate on training data
        y_pred_train = svr_model.predict(X_train_scaled)
        evaluate_model(svr_model, X_train_scaled, y_train_scaled, scaler_y, "Train")

        # Predict and evaluate on testing data
        y_pred_test = svr_model.predict(X_test_scaled)
        evaluate_model(svr_model, X_test_scaled, y_test_scaled, scaler_y, "Test")

        # Plot predictions vs actual data
        plot_predictions(svr_model, scaler_X, scaler_y, X, Y, "Actual vs SVR Prediction")

    def evaluate_model(model, X_scaled, y_scaled, scaler_y, data_type):
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_actual = scaler_y.inverse_transform(y_scaled)

        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)

        st.write(f"Mean Squared Error ({data_type}): {mse}")
        st.write(f"Root Mean Squared Error ({data_type}): {rmse}")
        st.write(f"Mean Absolute Error ({data_type}): {mae}")
        st.write(f"R-squared ({data_type}): {r2}")

    def plot_predictions(model, scaler_X, scaler_y, X, Y, title):
        x_scaled = scaler_X.transform(X)
        y_pred_scaled = model.predict(x_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

        plt.plot(Y, label='Actual Data', marker='o')
        plt.plot(y_pred, label='SVR Prediction', marker='x')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        st.pyplot()

    train_svr()

elif menu == "Predictions":
    st.header("Predictions 📊")

    def grnn_predict(X_train, y_train, X_test, sigma=0.1):
        diff = X_train - X_test[:, np.newaxis]
        distance = np.exp(-np.sum(diff ** 2, axis=2) / (2 * sigma ** 2))
        output = np.sum(distance * y_train, axis=1, keepdims=True) / np.sum(distance, axis=1, keepdims=True)
        return output

    def main():
        st.title("Inbound Tourism Forecasting")

        # Read data from CSV file
        df = pd.read_csv('Malaysia-Tourism.csv')

        # Show data
        st.subheader("Data:")
        st.write(df.head())

        # Convert 'Date' column to datetime format with dayfirst=True
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df.set_index('Date', inplace=True)
        df['NumericDate'] = df.index.map(pd.Timestamp.toordinal)

        # Features and target
        X = df['NumericDate'].values.reshape(-1, 1)
        y = df['Actual'].values

        # Feature scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Model selection
        st.subheader("Select Model:")
        model_selection = st.radio("", ("Support Vector Regression (SVR)", "General Regression Neural Network (GRNN)"))

        if model_selection == "Support Vector Regression (SVR)":
            model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
            model.fit(X_scaled, y_scaled)
        elif model_selection == "General Regression Neural Network (GRNN)":
            y_pred_scaled = grnn_predict(X_scaled, y_scaled, X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            num_months = st.number_input("Enter the number of months to forecast:", min_value=1, max_value=50)
            last_date = df.index[-1]
            next_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_months + 1)]
            next_numeric_dates = np.array([date.toordinal() for date in next_dates]).reshape(-1, 1)
            next_numeric_dates_scaled = scaler_X.transform(next_numeric_dates)
            next_predictions_scaled = grnn_predict(X_scaled, y_scaled, next_numeric_dates_scaled)
            next_predictions = scaler_y.inverse_transform(next_predictions_scaled.reshape(-1, 1)).ravel()
            plot_predictions(df, y_pred, next_dates, next_predictions, num_months)
            st.subheader("Predicted Data:")
            predictions_df = pd.DataFrame({'Date': next_dates, 'Predicted': next_predictions})
            st.write(predictions_df)
        else:
            st.warning("Invalid model selection.")

    def plot_predictions(df, y_pred, next_dates, next_predictions, num_months):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df['Actual'], label='Actual Data', marker='o')
        ax.plot(df.index, y_pred, label='Predicted Data', marker='x')
        ax.plot(next_dates, next_predictions, label=f'Next {num_months} Months Predictions', marker='s')
        ax.set_xlabel('Date')
        ax.set_ylabel('Tourism Data')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    if __name__ == "__main__":
        main()

elif menu == "About":
    st.header("About TourVis Pro 📚")
    st.markdown(
        """
        **TourVis Pro** is an advanced predictive analytics platform designed to forecast tourism trends in Malaysia. 
        Utilizing cutting-edge machine learning models, including Support Vector Regression (SVR) and Generalized 
        Regression Neural Networks (GRNN), TourVis Pro provides accurate and actionable insights for businesses, 
        policymakers, and researchers.
        
        **Key Features**:
        - Accurate prediction of tourist arrivals
        - User-friendly interface
        - Integration of advanced ML models
        - Customizable forecasting period
        
        **Contact Us**:
        For more information, please contact us at [info@tourvispro.com](mailto:info@tourvispro.com).
        """
    )
