import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from pyGRNN import GRNN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Title of the application
st.title(" TourVis Pro: Predictive Analytics for Tourism 📊 ")

# Main menu bar on the sidebar
st.sidebar.title("Main Menu")

# Add menu items
menu = st.sidebar.radio(
    "Select an option",
    ("Home", "Data Overview", "Model Training", "Predictions", "About")
)

# Display content based on the selected menu item
if menu == "Home":
    st.header(" Welcome to SVR & GRNN Web-based App! 👋 ")

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
    st.header(" Data Overview 📈 ")
    st.write("Here you can see the overview of the data used for forecasting from 2011 to 2023.")

    col1, col2 = st.columns([1, 2])  # Split the page into 1/4 and 3/4

    with col1:
        # Load and display your dataset here
        df = pd.read_csv('Malaysia-Tourism1.csv')
        st.dataframe(df)

    with col2:
        st.image("Data Animation.gif", width=400)

elif menu == "Model Training":
    # Add your model training code here
    def main():
        st.title(" 1) Inbound Tourism using SVR ")

        # Reading the csv file
        df = pd.read_csv('Malaysia-Tourism1.csv')
        df.isnull().sum()

        data = df.drop(['Date'], axis=1)
        data.head()

        # Time Series Generator
        # Choose input and output
        n_input = 1
        n_output = 1

        # Creating TimeseriesGenerator
        generator = TimeseriesGenerator(data.values, data.values, length=n_input, batch_size=1)

        # Creating DataFrame to store results
        data_ts = pd.DataFrame(columns=['x', 'y'])

        # Storing results from TimeseriesGenerator into DataFrame
        for i in range(len(generator)):
            x, y = generator[i]
            df_gen = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
            data_ts = pd.concat([data_ts, df_gen], ignore_index=True)

        # Displaying DataFrame with results
        st.write(data_ts)

        X = np.array(data_ts['x'])
        Y = np.array(data_ts['y'])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Plotting
        import matplotlib.pyplot as plt
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

        # Metrics for training set
        y_pred_train = svr_model.predict(X_train_scaled)
        mse_train = mean_squared_error(y_train_scaled, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        mae_train = mean_absolute_error(y_train_scaled, y_pred_train)
        r2_train = r2_score(y_train_scaled, y_pred_train)

        # Inverse transform for actual values
        y_pred_train_inv = scaler_y.inverse_transform(y_pred_train.reshape(-1,1))
        y_train_inv = scaler_y.inverse_transform(y_train_scaled.reshape(-1,1))

        mse_train_inv = mean_squared_error(y_train_inv, y_pred_train_inv)
        rmse_train_inv = np.sqrt(mse_train_inv)
        mae_train_inv = mean_absolute_error(y_train_inv, y_pred_train_inv)
        r2_train_inv = r2_score(y_train_inv, y_pred_train_inv)

        # Plotting actual vs SVR predictions for training set
        plt.plot(y_pred_train_inv, label='Actual Data', marker='o')
        plt.plot(y_train, label='SVR Prediction', marker='x')
        st.write('Actual Data vs SVR Prediction')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.legend()
        plt.grid(True)
        st.pyplot()

        # Metrics for test set
        y_pred_test = svr_model.predict(X_test_scaled)
        mse_test = mean_squared_error(y_test_scaled, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        mae_test = mean_absolute_error(y_test_scaled, y_pred_test)
        r2_test = r2_score(y_test_scaled, y_pred_test)

        # Inverse transform for actual values
        y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1,1))
        y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1))

        mse_test_inv = mean_squared_error(y_test_inv, y_pred_test_inv)
        rmse_test_inv = np.sqrt(mse_test_inv)
        mae_test_inv = mean_absolute_error(y_test_inv, y_pred_test_inv)
        r2_test_inv = r2_score(y_test_inv, y_pred_test_inv)

        # Plotting actual vs SVR predictions for test set
        plt.plot(y_pred_test_inv, label='Actual Data', marker='o')
        plt.plot(y_test, label='SVR Prediction', marker='x')
        st.write('Actual Data vs SVR Prediction')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
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
        st.title(" 2) Inbound Tourism using GRNN ")

        #Reading the csv file
        df = pd.read_csv('Malaysia-Tourism1.csv')
    

        st.subheader("Data:")
        st.write(df.head())

        # Convert 'Date' column to datetime format with dayfirst=True
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

        # Set 'Date' as the index
        df.set_index('Date', inplace=True)

        # Prepare data for GRNN
        df['NumericDate'] = df.index.map(pd.Timestamp.toordinal)

        # Features and target
        X = df['NumericDate'].values.reshape(-1, 1)
        y = df['Actual'].values

        # Normalize features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Predict on actual data
        y_pred_scaled = grnn_predict(X_scaled, y_scaled, X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Plot actual and predicted values
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Actual'], label='Actual')
        ax.plot(df.index, y_pred, label='GRNN Predictions', linestyle='--', color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Tourism Numbers')
        ax.set_title('Inbound Tourism using GRNN')
        ax.legend()
        st.pyplot(fig)

        # Calculate and display MSE, RMSE, and MAE
        st.subheader(f"Value of MSE, RMSE & MAE:")
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        st.write(f'Mean Squared Error (MSE): {mse}')
        st.write(f'Root Mean Squared Error (RMSE): {rmse}')
        st.write(f'Mean Absolute Error (MAE): {mae}')
        st.write(f'R-squared (R^2): {r2}')

    if __name__ == "__main__":
        main()

elif menu == "Predictions":

    def grnn_predict(X_train, y_train, X_test, sigma=0.1):
        # Calculate the Gaussian kernel
        diff = X_train - X_test[:, np.newaxis]
        distance = np.exp(-np.sum(diff ** 2, axis=2) / (2 * sigma ** 2))
    
        # Calculate the output using the kernel
        output = np.sum(distance * y_train, axis=1, keepdims=True) / np.sum(distance, axis=1, keepdims=True)
    
        return output

    def main():
        st.title("Inbound Tourism Forecasting")

        # Read data from CSV file
        df = pd.read_csv('Malaysia-Tourism1.csv')

        # Show data
        st.subheader("Data:")
        st.write(df.head())

        # Convert 'Date' column to datetime format with dayfirst=True
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

        # Set 'Date' as index
        df.set_index('Date', inplace=True)

        # Prepare data for model
        # Convert Date to numerical value because the model cannot work with datetime type directly
        df['NumericDate'] = df.index.map(pd.Timestamp.toordinal)

        # Features and target
        X = df['NumericDate'].values.reshape(-1, 1)
        y = df['Actual'].values

        # Feature scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Display "Select Model:" as a header
        st.subheader("Select Model:")
        # Model selection using radio button
        model_selection = st.radio("", ("Support Vector Regression (SVR)", "General Regression Neural Network (GRNN)"))

        if model_selection == "Support Vector Regression (SVR)":
            # Initialize SVR model
            model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

            # Fit the SVR model
            model.fit(X_scaled, y_scaled)

        elif model_selection == "General Regression Neural Network (GRNN)":
            # Predictions on actual data using GRNN
            y_pred_scaled = grnn_predict(X_scaled, y_scaled, X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            # Input for the number of months to forecast
            num_months = st.number_input("Enter the number of months to forecast:", min_value=1, max_value=50)

            # Make predictions for the next 'num_months' months
            last_date = df.index[-1]
            next_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_months + 1)]
            next_numeric_dates = np.array([date.toordinal() for date in next_dates]).reshape(-1, 1)

            # Normalizing the prediction dates
            next_numeric_dates_scaled = scaler_X.transform(next_numeric_dates)

            # Predicting values for the next 'num_months' months using GRNN
            next_predictions_scaled = grnn_predict(X_scaled, y_scaled, next_numeric_dates_scaled)
            next_predictions = scaler_y.inverse_transform(next_predictions_scaled.reshape(-1, 1)).ravel()

            # Plotting forecasted and actual values
            st.subheader("Predictions:")
            plot_predictions(df, y_pred, next_dates, next_predictions, num_months)

            # Print predicted values for the next 'num_months' months
            st.subheader(f"Predictions for the next {num_months} months:")
            for date, pred in zip(next_dates, next_predictions):
                st.write(f"Date: {date.strftime('%Y-%m')}, Predicted: {pred}")

            return

        # Predictions on actual data
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Input for the number of months to forecast
        num_months = st.number_input("Enter the number of months to forecast:", min_value=1, max_value=50)

        # Make predictions for the next 'num_months' months
        last_date = df.index[-1]
        next_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_months + 1)]
        next_numeric_dates = np.array([date.toordinal() for date in next_dates]).reshape(-1, 1)

        # Normalizing the prediction dates
        next_numeric_dates_scaled = scaler_X.transform(next_numeric_dates)

        # Predicting values for the next 'num_months' months
        next_predictions_scaled = model.predict(next_numeric_dates_scaled)
        next_predictions = scaler_y.inverse_transform(next_predictions_scaled.reshape(-1, 1)).ravel()

        # Plotting forecasted and actual values
        st.subheader("Predictions Graph:")
        plot_predictions(df, y_pred, next_dates, next_predictions, num_months)

        # Print predicted values for the next 'num_months' months
        st.subheader(f"Predictions for the next {num_months} months:")
        for date, pred in zip(next_dates, next_predictions):
            st.write(f"Date: {date.strftime('%Y-%m')}, Predicted: {pred}")

    def plot_predictions(df, y_pred, next_dates, next_predictions, num_months):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Actual'], label='Actual')
        ax.plot(df.index, y_pred, label='Predictions', linestyle='--', color='blue')
        ax.plot(next_dates, next_predictions, label='Future Predictions', linestyle='--', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Tourism Numbers')
        ax.set_title('Tourism Forecast')
        ax.legend()
        st.pyplot(fig)
    if __name__ == "__main__":
        main()

elif menu == "About":
    st.header("About")
    st.write("This section contains information about the application.")
    st.write("""
    This application was created to forecast tourism numbers using various machine learning models.
    """)
