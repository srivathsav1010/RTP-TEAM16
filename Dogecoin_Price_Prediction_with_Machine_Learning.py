import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("DOGE-USD.csv")
data.head()

data.corr(numeric_only=True)

# Convert 'Date' column to datetime without deprecated argument
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data.isnull().any()
data.isnull().sum()
data = data.dropna()
data.describe()

plt.figure(figsize=(20, 7))
x = data.groupby('Date')['Close'].mean()
x.plot(linewidth=2.5, color='b')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title("Date vs Close of 2021")

data["gap"] = (data["High"] - data["Low"]) * data["Volume"]
data["y"] = data["High"] / data["Volume"]
data["z"] = data["Low"] / data["Volume"]
data["a"] = data["High"] / data["Low"]
data["b"] = (data["High"] / data["Low"]) * data["Volume"]
abs(data.corr()["Close"].sort_values(ascending=False))

data = data[["Close", "Volume", "gap", "a", "b"]]
data.head()

# Add more lag features
def create_lag_features(df, feature, lags=3):
    for lag in range(1, lags + 1):
        df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
    return df

# Add moving average features
def add_moving_average_features(df, feature, windows=[3, 7, 14]):
    for window in windows:
        df[f'{feature}_ma{window}'] = df[feature].rolling(window=window).mean()
    return df

# Add new features
data = create_lag_features(data, "Close", lags=3)
data = add_moving_average_features(data, "Close")
data.dropna(inplace=True)

# Split data into training and testing sets
train, test = train_test_split(data, test_size=0.2, shuffle=False)

# Prepare training data
X_train = train.drop(columns=["Close"])
y_train = train["Close"]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Hyperparameter tuning for RandomForestRegressor
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Train the best model
best_model = grid_search.best_estimator_

# Prepare test data
X_test = test.drop(columns=["Close"])
y_test = test["Close"]
X_test_scaled = scaler.transform(X_test)

# Function to predict future prices
def predict_future_prices(model, last_data, num_predictions, feature_columns):
    future_predictions = []
    current_data = last_data.copy()

    for _ in range(num_predictions):
        # Create lagged features for the current data
        current_data = create_lag_features(current_data, "Close", lags=1)  # Corrected argument name
        current_data.dropna(inplace=True)

        # Prepare the input features for prediction
        X_future = pd.DataFrame(current_data[feature_columns].iloc[-1:])  # Ensure feature names are preserved

        # Predict the next price
        next_prediction = model.predict(X_future)[0]
        future_predictions.append(next_prediction)

        # Add the predicted value to the current data for the next iteration
        new_row = {"Close": next_prediction}
        for col in feature_columns:
            new_row[col] = current_data[col].iloc[-1]
        new_row_df = pd.DataFrame([new_row])
        current_data = pd.concat([current_data, new_row_df], ignore_index=True)

    return future_predictions

# Predict on test data
predictions = best_model.predict(X_test_scaled)

# Predict future prices
num_future_days = 10  # Number of future days to predict
last_data = test.copy()  # Use the last available data for prediction
future_predictions = predict_future_prices(best_model, last_data, num_future_days, X_train.columns)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(test.index, y_test, label="Actual", color='blue')
plt.plot(test.index, predictions, label="Predicted (Test)", color='red')

# Plot future predictions
future_dates = pd.date_range(start=test.index[-1], periods=num_future_days + 1, freq='D')[1:]
plt.plot(future_dates, future_predictions, label="Predicted (Future)", color='green', linestyle='--')

plt.legend()
plt.title("Dogecoin Price Prediction (Improved)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()