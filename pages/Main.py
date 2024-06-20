import streamlit as st
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
        df = pd.read_excel('Malaysia-Tourism1.xlsx')
        st.dataframe(df)

    with col2:
        st.image("Data Animation.gif", width=400)

elif menu == "Model Training":
    class TimeSeriesDataset(Dataset):
        def __init__(self, data, input_length, output_length):
            self.data = data
            self.input_length = input_length
            self.output_length = output_length

        def __len__(self):
            return len(self.data) - self.input_length - self.output_length + 1

        def __getitem__(self, idx):
            x = self.data[idx:idx + self.input_length]
            y = self.data[idx + self.input_length:idx + self.input_length + self.output_length]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def main():
        st.title(" 1) Inbound Tourism using SVR ")

        #Reading the csv file
        df = pd.read_excel('Malaysia-Tourism1.xlsx')
        df

        df.isnull().sum()

        data = df.drop(['Date'], axis=1)
        data.head()

        #Time Series Generator
        #Choose input and output
        n_input = 1
        n_output = 1

        # Creating TimeSeriesDataset
        dataset = TimeSeriesDataset(data.values, n_input, n_output)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Creating DataFrame to store results
        data_ts = pd.DataFrame(columns=['x', 'y'])

        # Saving results from TimeSeriesDataset to DataFrame
        for x, y in dataloader:
            df = pd.DataFrame({'x': x.flatten().numpy(), 'y': y.flatten().numpy()})
            data_ts = pd.concat([data_ts, df], ignore_index=True)

        # Displaying the result DataFrame
        st.write(data_ts)

        # Split Data
        data_ts[['x', 'y']] = data_ts[['x', 'y']].astype(int)

        X = np.array(data_ts['x'])
        Y = np.array(data_ts['y'])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        import matplotlib.pyplot as plt
        # Creating plot
        plt.plot(Y, label='Prediction Value', marker='x')

        # Adding labels and title
        st.write('Actual Data')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
    
        # Adding legend
        plt.legend()

        # Displaying plot
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

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        # Predicting values for training data
        y_pred_train = svr_model.predict(X_train_scaled)

        # Calculating Mean Squared Error (MSE) for training data
        mse_train = mean_squared_error(y_train_scaled, y_pred_train)
        st.write("Mean Squared Error (Train):", mse_train)

        # Calculating Root Mean Squared Error (RMSE) for training data
        rmse_train = np.sqrt(mse_train)
        st.write("Root Mean Squared Error (Train):", rmse_train)

        # Calculating Mean Absolute Error (MAE) for training data
        mae_train = mean_absolute_error(y_train_scaled, y_pred_train)
        st.write("Mean Absolute Error (Train):", mae_train)

        # Calculating Coefficient of Determination (R^2) for training data
        r2_train = r2_score(y_train_scaled, y_pred_train)
        st.write("R^2 (Train):", r2_train)

        y_pred_train_inv = scaler_y.inverse_transform(y_pred_train.reshape(-1,1))
        y_train_inv = scaler_y.inverse_transform(y_train_scaled.reshape(-1,1))

        # Calculating Mean Squared Error (MSE) for training data
        mse_train = mean_squared_error(y_train_inv, y_pred_train_inv)
        st.write("Mean Squared Error (Train):", mse_train)

        # Calculating Root Mean Squared Error (RMSE) for training data
        rmse_train = np.sqrt(mse_train)
        st.write("Root Mean Squared Error (Train):", rmse_train)

        # Calculating Mean Absolute Error (MAE) for training data
        mae_train = mean_absolute_error(y_train_inv, y_pred_train_inv)
        st.write("Mean Absolute Error (Train):", mae_train)

        # Calculating Coefficient of Determination (R^2) for training data
        r2_train = r2_score(y_train_inv, y_pred_train_inv)
        st.write("R^2 (Train):", r2_train)
        
        import matplotlib.pyplot as plt
        # Creating plot
        plt.plot(y_pred_train_inv, label='Actual Data', marker='o')
        plt.plot(y_train, label='SVR Prediction', marker='x')

        # Adding labels and title
        st.write('Actual Data vs SVR Prediction')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')

        # Adding legend
        plt.legend()

        # Displaying plot
        plt.grid(True)
        st.pyplot()

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        # Predicting values for test data
        y_pred_test = svr_model.predict(X_test_scaled)

        # Calculating Mean Squared Error (MSE) for test data
        mse_test = mean_squared_error(y_test_scaled, y_pred_test)
        st.write("Mean Squared Error (Test):", mse_test)

        # Calculating Root Mean Squared Error (RMSE) for test data
        rmse_test = np.sqrt(mse_train)
        st.write("Root Mean Squared Error (Test):", rmse_test)

        # Calculating Mean Absolute Error (MAE) for test data
        mae_test = mean_absolute_error(y_test_scaled, y_pred_test)
        st.write("Mean Absolute Error (Test):", mae_test)

        # Calculating Coefficient of Determination (R^2) for test data
        r2_test = r2_score(y_test_scaled, y_pred_test)
        st.write("R^2 (Test):", r2_test)

        y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1,1))
        y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1))

        # Calculating Mean Squared Error (MSE) for test data
        mse_test = mean_squared_error(y_test_inv, y_pred_test_inv)
        st.write("Mean Squared Error (Test):", mse_test)

        # Calculating Root Mean Squared Error (RMSE) for test data
        rmse_test = np.sqrt(mse_test)
        st.write("Root Mean Squared Error (Test):", rmse_test)

        # Calculating Mean Absolute Error (MAE) for test data
        mae_test = mean_absolute_error(y_test_inv, y_pred_test_inv)
        st.write("Mean Absolute Error (Test):", mae_test)

        # Calculating Coefficient of Determination (R^2) for test data
        r2_test = r2_score(y_test_inv, y_pred_test_inv)
        st.write("R^2 (Test):", r2_test)

        import matplotlib.pyplot as plt
        # Creating plot
        plt.plot(y_pred_test_inv, label='Actual Data', marker='o')
        plt.plot(y_test_inv, label='SVR Prediction', marker='x')

        # Adding labels and title
        st.write('Actual Data vs SVR Prediction')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')

        # Adding legend
        plt.legend()

        # Displaying plot
        plt.grid(True)
        st.pyplot()
        
    if __name__ == '__main__':
        main()
