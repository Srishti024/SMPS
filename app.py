import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model 
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Correct model file path
model = load_model("C:/Users/srish/Desktop/project/Stock Predictions Model.keras")

st.set_page_config(page_title="Stock Market Predictor", layout="wide")

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')
stock = st.sidebar.text_input('Enter Stock Symbol', 'GOOG')
start = st.sidebar.date_input('Start Date', pd.to_datetime('2012-01-01'))
end = st.sidebar.date_input('End Date', pd.to_datetime('2022-12-31'))

st.header('Stock Market Predictor')

# Download stock data
data = yf.download(stock, start=start, end=end)

st.subheader('Stock Data')
st.write(data)

# Splitting the data for training and testing
data_train = pd.DataFrame(data['Close'][0: int(len(data)*0.80)])  # Use 'Close' explicitly
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

# Normalizing the training data (Fit scaler on training data)
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

# Add the last 100 days from training data to test data
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

# Normalize the test data using the SAME SCALER
data_test_scaled = scaler.transform(data_test)

# Moving Averages (50, 100, 200)
ma_50_days = data['Close'].rolling(50).mean()
ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()

# Plotting the price vs MA50
st.subheader('Price vs MA50')
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label="MA 50")
plt.plot(data['Close'], 'g', label="Price")
plt.legend()
st.pyplot(fig1)

# Plotting the price vs MA50 and MA100
st.subheader('Price vs MA50 vs MA100')
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label="MA 50")
plt.plot(ma_100_days, 'b', label="MA 100")
plt.plot(data['Close'], 'g', label="Price")
plt.legend()
st.pyplot(fig2)

# Plotting the price vs MA100 and MA200
st.subheader('Price vs MA100 vs MA200')
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label="MA 100")
plt.plot(ma_200_days, 'b', label="MA 200")
plt.plot(data['Close'], 'g', label="Price")
plt.legend()
st.pyplot(fig3)

# Preparing data for prediction
x = []
y = []

for i in range(100, data_test_scaled.shape[0]):  # Ensure we skip the first 100 days for proper window
    x.append(data_test_scaled[i-100:i])  # Sliding window of 100 days
    y.append(data_test_scaled[i, 0])

# Converting to numpy arrays
x, y = np.array(x), np.array(y)

# Verify the shape of the input data
st.write(f"Shape of x: {x.shape}, Shape of y: {y.shape}")

# Predicting using the model
predicted_price = model.predict(x)

# Rescale the predictions and actual values back to original scale
predicted_price = scaler.inverse_transform(predicted_price)  # Use inverse transform for scaling back
y = scaler.inverse_transform(y.reshape(-1, 1))  # Reshape y before inverse scaling

# Plotting Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predicted_price, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
# streamlit run "C:\Users\srish\Desktop\project\app.py"

