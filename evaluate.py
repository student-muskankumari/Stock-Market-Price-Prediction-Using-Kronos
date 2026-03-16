#!/usr/bin/env python3
"""
Multi-ticker evaluator with summary table + average metrics.

Usage:
    python evaluate.py --tickers AAPL MSFT TSLA NVDA --model model.joblib

Outputs:
 - eval_out/<TICKER>_predictions.csv
 - eval_out/<TICKER>_pred_vs_true.png
 - eval_out/multi_ticker_comparison.png
 - eval_out/multi_ticker_summary.csv
 Prints a nicely formatted metrics table and average accuracy.
"""
import argparse
import os
import math
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# Utilities
# -------------------------
def safe_yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(data, tuple):
        # sometimes yfinance returns (df, meta)
        for part in data:
            if isinstance(part, pd.DataFrame):
                data = part
                break
    if not isinstance(data, pd.DataFrame):
        raise RuntimeError(f"yfinance returned unexpected type: {type(data)}")
    return data

def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    # strip and unify names
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)
    # common rename
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("open","o"): rename_map[c] = "Open"
        elif lc in ("high","h"): rename_map[c] = "High"
        elif lc in ("low","l"): rename_map[c] = "Low"
        elif lc in ("close","c"): rename_map[c] = "Close"
        elif "volume" in lc: rename_map[c] = "Volume"
        elif lc in ("adj close","adjclose"): rename_map[c] = "Adj Close"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                df.index = pd.date_range(start="2000-01-01", periods=len(df), freq="D")
    df = df.sort_index()
    return df

def add_time_fields(df: pd.DataFrame):
    t = pd.to_datetime(df.index)
    df["Day"] = t.day
    df["Month"] = t.month
    df["Year"] = t.year
    return df

def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

# -------------------------
# Single-ticker evaluation (returns dict)
# -------------------------
def evaluate_ticker(model, ticker, start, end, out_dir="eval_out", feature_list=None):
    print(f"[INFO] Evaluating {ticker} from {start.date()} to {end.date()}...")
    df = safe_yf_download(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df.empty:
        print(f"[WARN] No data for {ticker}")
        return None

    df = standardize_df(df)
    # require basic OHLCV
    if not all(c in df.columns for c in ("Open","High","Low","Close","Volume")):
        print(f"[WARN] {ticker} missing OHLCV columns, skipping.")
        return None

    # create time features (Day, Month, Year) - the training script used these
    df = add_time_fields(df)

    # If features.joblib exists, use it; otherwise prefer the training features:
    if feature_list is None:
        # default to the features used in the training script you ran earlier:
        feature_list = ["Open","High","Low","Volume","Day","Month","Year"]

    # if some features not present, compute derivatives if possible
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"[WARN] Missing features for {ticker}: {missing}. Attempting to compute common derived features.")
        # compute simple MAs and returns to try fill missing ones (best-effort)
        if "ma5" not in df.columns and "Close" in df.columns:
            df["ma5"] = df["Close"].rolling(5).mean()
        if "Return" not in df.columns and "Close" in df.columns:
            df["Return"] = df["Close"].pct_change()
        # recompute missing list
        missing = [f for f in feature_list if f not in df.columns]
        if missing:
            print(f"[WARN] Still missing after attempted derivation: {missing}. Rows with NaNs will be dropped.")

    # select features and drop rows with NaNs
    Xdf = df[feature_list].copy()
    Xdf = Xdf.dropna()
    if Xdf.empty:
        print(f"[WARN] No usable rows for {ticker} after dropping NaNs for features.")
        return None

    y_series = df.loc[Xdf.index, "Close"]

    # Try predicting with DataFrame (preserve column names) then fallback to numpy
    try:
        y_pred = model.predict(Xdf)
    except Exception:
        y_pred = model.predict(Xdf.values)

    # ensure 1D numpy arrays and align length
    y_pred = np.array(y_pred).ravel()
    y_true = np.array(y_series.values).ravel()
    n = min(len(y_pred), len(y_true))
    if n == 0:
        print(f"[WARN] No aligned rows for predictions on {ticker}.")
        return None

    y_pred = y_pred[:n]
    y_true = y_true[:n]
    idx = Xdf.index[:n]

    # metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    accuracy_pct = max(0.0, 100.0 - mape)

    # directional accuracy (exclude first because prev missing)
    prev = np.concatenate(([np.nan], y_true[:-1]))[:n]
    mask = ~np.isnan(prev)
    dir_acc = float(np.mean(np.sign(y_true[mask] - prev[mask]) == np.sign(y_pred[mask] - prev[mask])))

    # simple long-only backtest (use predicted > prev -> long)
    if n >= 2:
        pred_next = y_pred[:-1]
        prev_for_strategy = y_true[:-1]
        true_next = y_true[1:]
        returns = []
        for p, pr, t in zip(pred_next, prev_for_strategy, true_next):
            if pr == 0:
                returns.append(0.0)
            elif p > pr:
                returns.append((t / pr) - 1.0)
            else:
                returns.append(0.0)
        cum = np.cumprod([1 + r for r in returns]) if len(returns) else np.array([])
        if len(returns) > 1:
            mean_r = float(np.mean(returns))
            std_r = float(np.std(returns, ddof=1))
            sharpe = (mean_r / std_r * math.sqrt(252.0)) if std_r > 0 else 0.0
            peak = np.maximum.accumulate(cum) if len(cum) else np.array([])
            max_dd = float(np.min((cum - peak) / peak)) if len(peak) else 0.0
        else:
            sharpe = 0.0
            max_dd = 0.0
    else:
        returns = []
        cum = np.array([])
        sharpe = 0.0
        max_dd = 0.0

    # outputs
    os.makedirs(out_dir, exist_ok=True)
    pred_df = pd.DataFrame({"Date": idx, "Close_true": y_true, "Close_pred": y_pred}).set_index("Date")
    csv_path = os.path.join(out_dir, f"{ticker}_predictions.csv")
    pred_df.to_csv(csv_path)

    # per-ticker plot
    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(idx, y_true, label="True Close", linewidth=1.6)
    plt.plot(idx, y_pred, label="Predicted Close", linestyle="--", linewidth=1.2)
    plt.title(f"{ticker} - Predicted vs True Close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.4)
    png_path = os.path.join(out_dir, f"{ticker}_pred_vs_true.png")
    plt.tight_layout()
    plt.savefig(png_path)
    try:
        plt.close()
    except Exception:
        pass

    result = {
        "Ticker": ticker,
        "Rows": int(n),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE_pct": float(mape),
        "Accuracy_pct": float(accuracy_pct),
        "DirAcc": float(dir_acc),
        "Sharpe": float(sharpe),
        "MaxDD": float(max_dd),
        "csv": csv_path,
        "png": png_path
    }
    print(f"[INFO] {ticker} done: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, Accuracy={accuracy_pct:.2f}%")
    return result

# -------------------------
# Main: multi-ticker evaluation
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Multi-ticker model evaluator")
    p.add_argument("--tickers", nargs="+", required=True, help="Tickers (space separated)")
    p.add_argument("--model", required=True, help="Path to joblib model file")
    p.add_argument("--days", type=int, default=365*2, help="Lookback days (default 2 years)")
    p.add_argument("--out", default="eval_out", help="Output folder")
    p.add_argument("--features", default="features.joblib", help="Optional saved feature list (joblib)")
    args = p.parse_args()

    # load model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    model = joblib.load(args.model)
    print(f"[INFO] Loaded model from {args.model}")

    # try load feature list if present
    feature_list = None
    if args.features and os.path.exists(args.features):
        try:
            feature_list = joblib.load(args.features)
            print(f"[INFO] Loaded features from {args.features}: {feature_list}")
        except Exception as e:
            print(f"[WARN] Could not load features.joblib: {e}; will use default features")

    end = datetime.today()
    start = end - timedelta(days=args.days)

    results = []
    # Combined plot (all tickers) - use normalized x-axis (index positions)
    plt.figure(figsize=(14, 7))
    for ticker in args.tickers:
        res = evaluate_ticker(model, ticker, start, end, out_dir=args.out, feature_list=feature_list)
        if res:
            results.append(res)
            # load per-ticker CSV and plot predicted vs true as normalized lines (for comparability)
            pred_df = pd.read_csv(res["csv"], index_col=0, parse_dates=True)
            # normalize series to start=1 for display (avoid divide by zero)
            true_vals = pred_df["Close_true"].values
            pred_vals = pred_df["Close_pred"].values
            if len(true_vals) == 0 or true_vals[0] == 0 or pred_vals[0] == 0:
                continue
            true_norm = true_vals / true_vals[0]
            pred_norm = pred_vals / pred_vals[0]
            days = np.arange(len(true_norm))
            plt.plot(days, true_norm, label=f"{ticker} True")
            plt.plot(days, pred_norm, linestyle="--", label=f"{ticker} Pred")

    plt.title("Multi-Ticker Predicted vs True (normalized)")
    plt.xlabel("Days (index)")
    plt.ylabel("Normalized Price")
    plt.legend(ncol=2, fontsize="small")
    plt.grid(alpha=0.4)
    combined_png = os.path.join(args.out, "multi_ticker_comparison.png")
    plt.tight_layout()
    plt.savefig(combined_png, dpi=200)
    try:
        plt.show()
    except Exception:
        pass

    # save summary results CSV and print table
    if results:
        df_res = pd.DataFrame(results)
        summary_csv = os.path.join(args.out, "multi_ticker_summary.csv")
        df_res.to_csv(summary_csv, index=False)

        # print a clean table of key metrics
        display_cols = ["Ticker","Rows","MAE","RMSE","R2","Accuracy_pct","MAPE_pct","DirAcc","Sharpe","MaxDD"]
        print("\n===== MODEL PERFORMANCE SUMMARY =====")
        # format numeric columns
        print(df_res[display_cols].to_string(index=False, float_format="%.4f"))

        avg_accuracy = df_res["Accuracy_pct"].mean()
        print(f"\n[INFO] Average Accuracy across tickers: {avg_accuracy:.2f}%")
        print(f"[INFO] Saved summary CSV: {summary_csv}")
        print(f"[INFO] Saved combined plot: {combined_png}")

if __name__ == "__main__":
    main()
