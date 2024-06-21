import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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

# Custom TimeSeriesGenerator
def custom_timeseries_generator(data, length):
    x = []
    y = []
    for i in range(len(data) - length):
        x.append(data[i:i+length])
        y.append(data[i+length])
    return np.array(x), np.array(y)

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
            Welcome to our SVR & GRNN Web-Based App! Predict tourist arrivals in Malaysia with precision using Support Vector Regression and Generalized Regression Neural Network technology. Our user-friendly platform empowers businesses and policymakers with accurate forecasting for any selected year. Experience the future of tourism prediction today!
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
    # Add your model training code here
    def main():
        st.title("1) Inbound Tourism using SVR")

        # Reading the csv file
        df = pd.read_excel('Malaysia-Tourism1.xlsx')
        df

        df.isnull().sum()

        data = df.drop(['Date'], axis=1)
        data.head()

        # Custom TimeSeriesGenerator
        n_input = 1
        n_output = 1

        x, y = custom_timeseries_generator(data.values, n_input)

        data_ts = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
        st.write(data_ts)

        data_ts[['x', 'y']] = data_ts[['x', 'y']].astype(int)

        X = np.array(data_ts['x'])
        Y = np.array(data_ts['y'])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Plot Actual Data
        plt.plot(Y, label='Prediction Value', marker='x')
        st.write('Actual Data')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.legend()
        plt.grid(True)
        st.pyplot()

        X_train = X_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)

        X_test = X_test.reshape(-1,1)
        y_test = y_test.reshape(-1,1)

        # Scaling Dataset
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)

        X_test_scaled = scaler_X.fit_transform(X_test)
        y_test_scaled = scaler_y.fit_transform(y_test)

        # Initialize and fit the SVR model
        svr_model = SVR(kernel='rbf')
        svr_model.fit(X_train_scaled, y_train_scaled)

        # Prediksi nilai untuk data latih
        y_pred_train = svr_model.predict(X_train_scaled)

        # Menghitung Mean Squared Error (MSE) untuk data latih
        mse_train = mean_squared_error(y_train_scaled, y_pred_train)
        st.write("Mean Squared Error (Train):", mse_train)

        # Menghitung Root Mean Squared Error (RMSE) untuk data latih
        rmse_train = np.sqrt(mse_train)
        st.write("Root Mean Squared Error (Train):", rmse_train)

        # Menghitung Mean Absolute Error (MAE) untuk data latih
        mae_train = mean_absolute_error(y_train_scaled, y_pred_train)
        st.write("Mean Absolute Error (Train):", mae_train)

        # Menghitung Koefisien Determinasi (R^2) untuk data latih
        r2_train = r2_score(y_train_scaled, y_pred_train)
        st.write("R^2 (Train):", r2_train)

        y_pred_train_inv = scaler_y.inverse_transform(y_pred_train.reshape(-1,1))
        y_train_inv = scaler_y.inverse_transform(y_train_scaled.reshape(-1,1))

        mse_train = mean_squared_error(y_train_inv, y_pred_train_inv)
        st.write("Mean Squared Error (Train):", mse_train)

        rmse_train = np.sqrt(mse_train)
        st.write("Root Mean Squared Error (Train):", rmse_train)

        mae_train = mean_absolute_error(y_train_inv, y_pred_train_inv)
        st.write("Mean Absolute Error (Train):", mae_train)

        r2_train = r2_score(y_train_inv, y_pred_train_inv)
        st.write("R^2 (Train):", r2_train)

        # Plot SVR Prediction
        plt.plot(y_pred_train_inv, label='Actual Data', marker='o')
        plt.plot(y_train, label='SVR Prediction', marker='x')
        st.write('Actual Data vs SVR Prediction')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.legend()
        plt.grid(True)
        st.pyplot()

        # Prediksi nilai untuk data latih
        y_pred_test = svr_model.predict(X_test_scaled)

        mse_test = mean_squared_error(y_test_scaled, y_pred_test)
        st.write("Mean Squared Error (Test):", mse_test)

        rmse_test = np.sqrt(mse_train)
        st.write("Root Mean Squared Error (Test):", rmse_test)

        mae_test = mean_absolute_error(y_test_scaled, y_pred_test)
        st.write("Mean Absolute Error (Test):", mae_test)

        r2_test = r2_score(y_test_scaled, y_pred_test)
        st.write("R^2 (Test):", r2_test)

        y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1,1))
        y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1))

        mse_test = mean_squared_error(y_test_inv, y_pred_test_inv)
        print("Mean Squared Error (Test):", mse_test)

        rmse_test = np.sqrt(mse_test)
        print("Root Mean Squared Error (Test):", rmse_test)

        mae_test = mean_absolute_error(y_test_inv, y_pred_test_inv)
        print("Mean Absolute Error (Test):", mae_test)

        r2_test = r2_score(y_test_inv, y_pred_test_inv)
        print("R^2 (Test):", r2_test)

        # Plot SVR Prediction for test data
        plt.plot(y_pred_test_inv, label='Actual Data', marker='o')
        plt.plot(y_test, label='SVR Prediction', marker='x')
        st.write('Actual Data vs SVR Prediction')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.legend()
        plt.grid(True)
        st.pyplot()

        x_scaled = scaler_X.fit_transform(X.reshape(-1,1))
        y_pred = svr_model.predict(x_scaled)

        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1,1))

        # Plot Actual vs SVR Prediction
        plt.plot(Y, label='Actual Data', marker='o')
        plt.plot(y_pred_inv, label='SVR Prediction', marker='x')
        st.write('Actual vs SVR Prediction')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.title('Actual vs SVR Prediction')
        plt.legend()
        plt.grid(True)
        st.pyplot()

    if __name__ == "__main__":
        main()

    def grnn_predict(X_train, y_train, X_test, sigma=0.1):
        diff = X_train - X_test[:, np.newaxis]
        distance = np.exp(-np.sum(diff ** 2, axis=2) / (2 * sigma ** 2))
        output = np.sum(distance * y_train, axis=1, keepdims=True) / np.sum(distance, axis=1, keepdims=True)
        return output

    def main():
        st.title("2) Inbound Tourism using GRNN")

        df = pd.read_csv('Malaysia-Tourism.csv')
        st.subheader("Dataset")
        st.write(df)

        data = df.drop(['Date'], axis=1)

        n_input = 1
        x, y = custom_timeseries_generator(data.values, n_input)
        data_ts = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
        st.write(data_ts)

        data_ts[['x', 'y']] = data_ts[['x', 'y']].astype(int)

        X = np.array(data_ts['x'])
        Y = np.array(data_ts['y'])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Plot Actual Data
        plt.plot(Y, label='Prediction Value', marker='x')
        st.write('Actual Data')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.legend()
        plt.grid(True)
        st.pyplot()

        X_train = X_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        X_test = X_test.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)

        X_test_scaled = scaler_X.fit_transform(X_test)
        y_test_scaled = scaler_y.fit_transform(y_test)

        model_grnn = GRNN()
        model_grnn.fit(X_train_scaled, y_train_scaled)

        y_pred_train_grnn = model_grnn.predict(X_train_scaled)
        y_pred_train_grnn = scaler_y.inverse_transform(y_pred_train_grnn)

        mse_train = mean_squared_error(y_train, y_pred_train_grnn)
        rmse_train = np.sqrt(mse_train)
        mae_train = mean_absolute_error(y_train, y_pred_train_grnn)
        r2_train = r2_score(y_train, y_pred_train_grnn)

        st.write("Training Mean Squared Error:", mse_train)
        st.write("Training Root Mean Squared Error:", rmse_train)
        st.write("Training Mean Absolute Error:", mae_train)
        st.write("Training R^2:", r2_train)

        y_pred_test_grnn = model_grnn.predict(X_test_scaled)
        y_pred_test_grnn = scaler_y.inverse_transform(y_pred_test_grnn)

        mse_test = mean_squared_error(y_test, y_pred_test_grnn)
        rmse_test = np.sqrt(mse_test)
        mae_test = mean_absolute_error(y_test, y_pred_test_grnn)
        r2_test = r2_score(y_test, y_pred_test_grnn)

        st.write("Testing Mean Squared Error:", mse_test)
        st.write("Testing Root Mean Squared Error:", rmse_test)
        st.write("Testing Mean Absolute Error:", mae_test)
        st.write("Testing R^2:", r2_test)

        plt.plot(y_train, label='Actual Data', marker='o')
        plt.plot(y_pred_train_grnn, label='GRNN Prediction', marker='x')
        st.write('Actual Data vs GRNN Prediction')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.legend()
        plt.grid(True)
        st.pyplot()

        plt.plot(y_test, label='Actual Data', marker='o')
        plt.plot(y_pred_test_grnn, label='GRNN Prediction', marker='x')
        st.write('Actual Data vs GRNN Prediction')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.legend()
        plt.grid(True)
        st.pyplot()

    if __name__ == "__main__":
        main()

elif menu == "Predictions":
    st.header("Predictions 🔮")
    st.write("Make predictions using the trained model.")

    # Add code to load the model and make predictions here

elif menu == "About":
    st.header("About 🧾")
    st.write("Learn more about the application and its purpose.")
