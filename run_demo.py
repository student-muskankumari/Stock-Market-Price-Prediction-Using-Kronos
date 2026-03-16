# run_demo.py
"""
Quick test: loads last 200 daily AMZN candles and runs Kronos-mini for 3-day prediction.
"""
from src.data_loader import fetch_yfinance
from src.forecast_utils import df_to_klines, klines_to_closes
from src.model_handler_kronos import KronosModelHandler

if __name__ == "__main__":
    df = fetch_yfinance("AMZN", start="2023-01-01", end="2025-10-31")
    klines = df_to_klines(df[-200:])
    handler = KronosModelHandler(model_id="NeoQuasar/Kronos-mini", device="cpu")
    res = handler.predict(history_klines=klines, horizon_klines=3)
    print("Predicted klines:", res["klines"])
