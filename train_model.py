import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import yfinance as yf

# ========== Kronos Stock Predictor Training Script ==========

# 1️⃣ CONFIGURATION
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2025-11-05"

print(f"[INFO] Downloading {TICKER} data from Yahoo Finance...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)

if df.empty:
    raise ValueError("❌ No data found for ticker. Check symbol or internet connection.")

# 2️⃣ FEATURE ENGINEERING
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Features and target
X = df[['Open', 'High', 'Low', 'Volume', 'Day', 'Month', 'Year']]
y = df['Close']

# 3️⃣ SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ TRAIN MODEL
print("[INFO] Training RandomForestRegressor model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ EVALUATE
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n========== MODEL PERFORMANCE ==========")
print(f"Mean Absolute Error  : {mae:.3f}")
print(f"Root Mean Squared Err: {rmse:.3f}")
print(f"R² Score             : {r2:.3f}")

# 6️⃣ SAVE MODEL
joblib.dump(model, "model.joblib")
print("\n✅ Model saved as 'model.joblib' in project folder.")
