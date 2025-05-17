# Stock-price-predication-Model
# Overview
This code snippet demonstrates how to download historical stock data, perform some basic data manipulation and visualization, prepare the data for a machine learning model, train a simple linear regression model to predict stock prices, evaluate the model's performance, and visualize the actual versus predicted stock prices.

# Code Breakdown
import yfinance as yf

# Download historical data for a stock, eg Tesla
data = yf.download('TSLA', start='2024-01-01', end='2025-01-01', progress=False)

print(data.head())

print(data.tail())

This part of the code uses the yfinance library to download historical stock data for Tesla (TSLA) from the start of 2024 to the start of 2025. The progress=False argument suppresses the download progress bar. After downloading, the first 5 rows (data.head()) and the last 5 rows (data.tail()) of the downloaded data are printed to show a sample of the data.

import matplotlib.pyplot as plt

data['Close'].plot(title= 'Stock Closing Price')

plt.xlabel("Date")

plt.ylabel("Price")

plt.show()

Here, we import the matplotlib.pyplot library for plotting. This code then plots the 'Close' price column from the data DataFrame. The title, xlabel, and ylabel are set for clarity. plt.show() displays the generated plot.

data['MA10'] = data['Close'].rolling(window=10).mean()

data['MA50'] = data['Close'].rolling(window=50).mean()

data[['Close', 'MA10', 'MA50']].plot(title='Stock Closing Price with Moving Averages')

plt.xlabel("Date")

plt.ylabel("Price")

plt.show()

This section calculates two simple moving averages (MA): a 10-day moving average (MA10) and a 50-day moving average (MA50) of the 'Close' price. A moving average is calculated by taking the average of the data points within a specified window. The results are stored in new columns named 'MA10' and 'MA50' in the data DataFrame. Then, the 'Close', 'MA10', and 'MA50' columns are plotted together to visualize how the moving averages track the closing price.

data['Prev_Close'] = data['Close'].shift(1)

data = data.dropna()

This code creates a new column named Prev_Close which contains the closing price from the previous day. This is achieved by shifting the 'Close' column down by one row using the .shift(1) method. Since the first row will now have a missing value (NaN) in Prev_Close, the .dropna() method is used to remove any rows with missing values, which includes the first row in this case.

from sklearn.model_selection import train_test_split

X = data[['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'Prev_Close']]

y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

This code prepares the data for the machine learning model. It separates the features (X) which are the input variables for the model, from the target variable (y) which is what we want to predict (the 'Close' price). The features include 'Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', and 'Prev_Close'. The train_test_split function is then used to split the data into training and testing sets. The test_size=0.2 argument means that 20% of the data will be used for testing, and the remaining 80% for training. shuffle=False is important for time-series data as it keeps the data in chronological order.

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

Here, a Linear Regression model is imported from the sklearn.linear_model module. A LinearRegression object is created and then the fit() method is used to train the model using the training data (X_train and y_train). During training, the model learns the relationship between the features and the target variable.

from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np

y_pred = model.predict(X_test)

print("MAE(Mean abssulate error):", mean_absolute_error(y_test, y_pred))

print("RMSE(Root mean square error):", np.sqrt(mean_squared_error(y_test, y_pred)))

This section evaluates the performance of the trained model. The predict() method is used to make predictions (y_pred) on the testing data (X_test). Then, two common evaluation metrics for regression models, Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), are calculated using the mean_absolute_error and mean_squared_error functions from sklearn.metrics. np.sqrt() is used to calculate the square root for RMSE. These metrics quantify the difference between the actual test prices (y_test) and the predicted prices (y_pred).

plt.figure(figsize=(10, 6))

plt.plot(y_test.index, y_test, label='Actual Price')

plt.plot(y_test.index, y_pred, label='Predicted Price')

plt.title("Stock Price Prediction")

plt.xlabel("Date")

plt.ylabel("Price")

plt.legend()

plt.show()

Finally, this code visualizes the actual closing prices (y_test) and the predicted closing prices (y_pred) for the test set. A plot is created with a specified figure size. The x-axis represents the date (using the index of y_test), and the y-axis represents the price. The label argument provides labels for the legend, which helps distinguish between the actual and predicted prices. The title, xlabel, and ylabel are set for clarity. plt.legend() displays the legend, and plt.show() displays the plot. This visualization helps to visually assess how well the model's predictions align with the actual stock prices.
