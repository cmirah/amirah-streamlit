import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load data function
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess data function
def preprocess_data(df):
    train_data = df.iloc[:250]
    valid_data = df.iloc[250:]
    
    x_train = train_data[['susceptible', 'infected', 'recovered', 'fatal']].values
    y_train = train_data['infected'].values.reshape(-1, 1)
    x_valid = valid_data[['susceptible', 'infected', 'recovered', 'fatal']].values
    y_valid = valid_data['infected'].values.reshape(-1, 1)
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    y_train = scaler_y.fit_transform(y_train)
    
    return x_train, y_train, x_valid, y_valid, scaler_y

# Build and train model function
def build_and_train_model(x_train, y_train, epochs=100):
    model = Sequential([
        Dense(70, input_shape=(x_train.shape[1],), activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
    
    return model, history

# Evaluate model function
def evaluate_model(model, x_valid, y_valid, scaler_y):
    y_pred_scale = model.predict(x_valid)
    y_pred = scaler_y.inverse_transform(y_pred_scale)
    
    MAE = mean_absolute_error(y_valid, y_pred)
    MSE = mean_squared_error(y_valid, y_pred)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs((y_valid - y_pred) / y_valid)) * 100
    
    return y_pred, MAE, MSE, RMSE, MAPE

# Main function
def main():
    st.title("SIR-F Model Prediction")
    
    file_path = st.text_input("Enter the path to the CSV file", "cases_malaysia.csv")
    
    if st.button("Load and Preprocess Data"):
        df = load_data(file_path)
        st.write(df.tail())
        
        x_train, y_train, x_valid, y_valid, scaler_y = preprocess_data(df)
        
        st.session_state['x_train'] = x_train
        st.session_state['y_train'] = y_train
        st.session_state['x_valid'] = x_valid
        st.session_state['y_valid'] = y_valid
        st.session_state['scaler_y'] = scaler_y
        
        st.write("Data Preprocessed Successfully")
    
    if st.button("Train Model"):
        if 'x_train' in st.session_state and 'y_train' in st.session_state:
            epochs = st.number_input("Enter the number of epochs", min_value=10, max_value=500, value=100)
            model, history = build_and_train_model(st.session_state['x_train'], st.session_state['y_train'], epochs)
            
            st.session_state['model'] = model
            st.session_state['history'] = history
            
            st.write("Model Trained Successfully")
            
            # Plot training loss
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Train Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_title('Model Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("Please preprocess the data before training the model.")
    
    if st.button("Evaluate Model"):
        if 'model' in st.session_state and 'x_valid' in st.session_state and 'y_valid' in st.session_state and 'scaler_y' in st.session_state:
            y_pred, MAE, MSE, RMSE, MAPE = evaluate_model(
                st.session_state['model'], 
                st.session_state['x_valid'], 
                st.session_state['y_valid'], 
                st.session_state['scaler_y']
            )
            
            st.write(f"MAE: {MAE:.4f}")
            st.write(f"MSE: {MSE:.4f}")
            st.write(f"RMSE: {RMSE:.4f}")
            st.write(f"MAPE: {MAPE:.4f}")
            
            # Plot actual vs predicted
            df_result = pd.DataFrame({'Actual': st.session_state['y_valid'].flatten(), 'Predicted': y_pred.flatten()})
            st.write(df_result)
            
            fig, ax = plt.subplots(figsize=(20, 10))
            df_result.plot(kind='line', ax=ax)
            ax.set_title('Actual vs Predicted')
            ax.set_xlabel('Day')
            ax.set_ylabel('Infected')
            st.pyplot(fig)
        else:
            st.error("Please train the model before evaluating.")

# Run the app
if __name__ == "__main__":
    main()
