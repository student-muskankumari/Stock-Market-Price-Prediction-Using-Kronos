
import numpy as np
import pandas as pd
from typing import List, Dict

def df_to_klines(df: pd.DataFrame) -> List[Dict]:
    """Convert OHLCV DataFrame to list of kline dicts for Kronos tokenizer."""
    klines = []
    for _, row in df.iterrows():
        k = {
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row.get("Volume", 0.0))
        }
        klines.append(k)
    return klines

def klines_to_closes(klines: List[Dict]) -> np.ndarray:
    return np.array([k["close"] for k in klines], dtype=float)

def build_future_dates(last_date: pd.Timestamp, periods: int, freq: str = "D"):
    return pd.date_range(start=pd.to_datetime(last_date) + pd.Timedelta(days=1), periods=periods, freq=freq)

def evaluate(actual: np.ndarray, predicted: np.ndarray):
    import numpy as _np
    if actual.shape != predicted.shape:
        raise ValueError("Shapes mismatch")
    mae = float(_np.mean(_np.abs(actual - predicted)))
    rmse = float(_np.sqrt(_np.mean((actual - predicted) ** 2)))
    with _np.errstate(divide='ignore', invalid='ignore'):
        mape = _np.mean(_np.abs((actual - predicted) / _np.where(actual == 0, _np.nan, actual))) * 100
    if _np.isnan(mape):
        mape = float("inf")
    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}
