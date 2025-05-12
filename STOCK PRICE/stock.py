import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# 🔹 Specify the folder path and filename
folder_path = "E:\STOCK PRICE"  # <-- Change this to your dataset folder
file_name = "yahoo_stock.csv"      # <-- Your CSV file name
full_path = os.path.join(folder_path, file_name)

# 🔹 Load the dataset
df = pd.read_csv(full_path)

# 🔹 Parse dates and sort by date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 🔹 Use only the 'Date' and 'Close' columns
df = df[['Date', 'Close']].dropna()

# 🔹 Plot the closing prices
plt.figure(figsize=(10, 4))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.title('Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# 🔹 Convert dates to numeric values for ML
df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

# 🔹 Define features (X) and target (y)
X = df[['Date_ordinal']]
y = df['Close']

# 🔹 Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 🔹 Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔹 Make predictions
y_pred = model.predict(X_test)

# 🔹 Plot actual vs predicted prices
plt.figure(figsize=(10, 4))
plt.plot(df['Date'].iloc[-len(y_test):], y_test.values, label='Actual Price')
plt.plot(df['Date'].iloc[-len(y_test):], y_pred, label='Predicted Price', linestyle='--')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
