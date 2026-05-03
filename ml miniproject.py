# Forecast Monthly Car Sales using Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("car_sales.csv")

# Convert Month column to datetime
df['Month'] = pd.to_datetime(df['Month'])

# Sort values
df = df.sort_values('Month')

# -----------------------------
# Step 2: Feature Engineering
# -----------------------------
df['Time_Index'] = np.arange(len(df))

X = df[['Time_Index']]
y = df['Sales']

# -----------------------------
# Step 3: Train-Test Split
# -----------------------------
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# -----------------------------
# Step 4: Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 6: Evaluation
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

print("R2 Score:", r2_score(y_test, y_pred))
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# -----------------------------
# Step 7: Future Forecast
# -----------------------------
future_months = 12
last_index = df['Time_Index'].iloc[-1]

future_index = np.arange(last_index + 1, last_index + future_months + 1)

# Fix: use DataFrame (avoids warning)
future_index_df = pd.DataFrame(future_index, columns=['Time_Index'])

future_predictions = model.predict(future_index_df)

# Fix: use 'ME' instead of 'M'
future_dates = pd.date_range(
    start=df['Month'].iloc[-1],
    periods=future_months + 1,
    freq='ME'
)[1:]

# -----------------------------
# Step 8: Forecast Table
# -----------------------------
forecast_df = pd.DataFrame({
    "Month": future_dates,
    "Predicted Sales": future_predictions
})

print("\nFuture Forecast:")
print(forecast_df)

# -----------------------------
# Step 9: Visualization
# -----------------------------
plt.figure(figsize=(10,5))

# Actual
plt.plot(df['Month'], df['Sales'], label="Actual Sales")

# Predicted
plt.plot(df['Month'][split:], y_pred, label="Predicted Sales")

# Future
plt.plot(future_dates, future_predictions, label="Future Forecast")

plt.xlabel("Month")
plt.ylabel("Car Sales")
plt.title("Monthly Car Sales Forecast using Linear Regression")
plt.legend()

plt.show()