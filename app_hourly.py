import streamlit as st
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import fetch_yfinance
from src.forecast_utils import df_to_klines, klines_to_closes, build_future_dates
from src.model_handler_kronos import KronosModelHandler, KronosError

st.set_page_config(page_title="Kronos — Hourly/Daily", layout="wide")
st.title("📈 Kronos — Live Stock Forecasting App")


st.sidebar.header("Model Input Settings")
ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value="RELIANCE.NS")
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "1wk"], index=0)
period = st.sidebar.selectbox("History Period", ["60d", "6mo", "1y", "2y", "5y"], index=3 if interval == "1d" else 0)
horizon = st.sidebar.number_input("Horizon (steps ahead)", min_value=1, max_value=260, value=34, step=1)

live_mode = st.sidebar.checkbox("Enable Live Mode", value=False)
refresh_rate = st.sidebar.number_input(
    "Refresh every (seconds)", min_value=5, max_value=600, value=60, step=5
)

margin = st.sidebar.slider("Prediction Confidence Band (%)", 0, 80, 30, 1)

run = st.button("🚀 Run Forecast")


def compute_bands(values, margin_pct):
    values = pd.Series(values, dtype="float64")
    lower = values * (1 - margin_pct / 100.0)
    upper = values * (1 + margin_pct / 100.0)
    return lower.tolist(), upper.tolist()

def compute_returns(price_list):
    """Return % change between steps."""
    returns = []
    for i in range(1, len(price_list)):
        prev = price_list[i-1]
        cur = price_list[i]
        if prev == 0:
            returns.append(0.0)
        else:
            returns.append((cur - prev) / prev * 100)
    return returns

def investment_backtest(actual_prices, predicted_prices):
    """Simple BUY/SELL simulation."""
    actions = []
    profit_list = []

    n = min(len(predicted_prices), max(0, len(actual_prices) - 1))
    if n <= 0:
        return [], [], 0.0, 0.0

    for i in range(n):
        today_actual = actual_prices[i]
        next_actual = actual_prices[i+1]
        predicted_next = predicted_prices[i]

        if predicted_next > today_actual:  # BUY
            profit = next_actual - today_actual
            actions.append("BUY")
        else:  # SELL
            profit = today_actual - next_actual
            actions.append("SELL")

        profit_list.append(profit)

    win_rate = (sum(1 for p in profit_list if p > 0) / len(profit_list) * 100) if profit_list else 0
    total_profit = sum(profit_list)

    return actions, profit_list, win_rate, total_profit

def display_scrolling_ticker(ticker_data):
    ticker_html = ""
    for t in ticker_data:
        color = "lime" if t["change"] > 0 else "red" if t["change"] < 0 else "gray"
        arrow = "▲" if t["change"] > 0 else "▼" if t["change"] < 0 else "▬"
        ticker_html += (
            f"<span style='color:{color}; font-weight:bold; margin-right:80px;'>"
            f"{arrow} {t['symbol']} | O: ₹{t['open']:,.2f} | H: ₹{t['high']:,.2f} | "
            f"L: ₹{t['low']:,.2f} | C: ₹{t['close']:,.2f} ({t['change']:+.2f}%)</span>"
        )

    st.markdown(
        f"""
        <style>
        .scroll-left {{
            height: 40px;
            overflow: hidden;
            position: relative;
            background: #000;
            color: white;
            border: 2px solid #333;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .scroll-left div {{
            position: absolute;
            width: 100%;
            height: 100%;
            line-height: 40px;
            text-align: left;
            transform: translateX(100%);
            animation: scroll-left 25s linear infinite;
            white-space: nowrap;
        }}
        @keyframes scroll-left {{
            0%   {{ transform: translateX(100%); }}
            100% {{ transform: translateX(-100%); }}
        }}
        </style>
        <div class="scroll-left"><div>{ticker_html}</div></div>
        """,
        unsafe_allow_html=True
    )

if run:
    try:
        today = datetime.date.today()
        if period.endswith("d"):
            start = today - datetime.timedelta(days=int(period[:-1]))
        elif period.endswith("mo"):
            start = today - datetime.timedelta(days=int(period[:-2]) * 30)
        elif period.endswith("y"):
            start = today - datetime.timedelta(days=int(period[:-1]) * 365)
        else:
            start = today - datetime.timedelta(days=365)

        st.write(f"📊 Fetching {interval}-interval data for **{ticker}**...")
        df = fetch_yfinance(ticker, start=start, end=today, interval=interval)

        if df is None or df.empty:
            st.warning("No data returned. Check ticker / interval / connectivity.")
        else:
            st.subheader("Historical Data Preview")
            st.dataframe(df.tail(10))

            handler = KronosModelHandler(model_id="NeoQuasar/Kronos-mini", device="cpu")

            klines = df_to_klines(df.tail(240 if interval == "1h" else 200))
            st.write("✅ Data processed successfully:", len(klines), "klines ready.")
            result = handler.predict(history_klines=klines, horizon_klines=horizon)
            pred_klines = result.get("klines", [])
            pred_closes = klines_to_closes(pred_klines)
            pred_closes = [float(x) for x in pred_closes]  # ensure floats
            future_idx = build_future_dates(df["Date"].iloc[-1], periods=horizon)
            lower, upper = compute_bands(pred_closes, margin)

            actual_values = df["Close"].tail(horizon).tolist()
            min_len = min(len(actual_values), len(pred_closes))
            if min_len > 0:
                actual_eval = actual_values[:min_len]
                pred_eval = pred_closes[:min_len]
                mse = sum((a - p)**2 for a, p in zip(actual_eval, pred_eval)) / min_len
                rmse = mse ** 0.5
                avg_error_pct = sum(abs((a - p) / a) * 100 for a, p in zip(actual_eval, pred_eval)) / min_len
            else:
                mse = rmse = avg_error_pct = 0.0

            actual_for_returns = df["Close"].tail(horizon + 1).tolist()
            pred_for_returns = pred_closes
            actual_returns = compute_returns(actual_for_returns)
            last_known = float(df["Close"].iloc[-1]) if "Close" in df.columns else None
            pred_returns = compute_returns(([last_known] + pred_closes) if last_known is not None else pred_closes)
            min_len_ret = min(len(actual_returns), len(pred_returns))
            actual_returns_trim = actual_returns[:min_len_ret]
            pred_returns_trim = pred_returns[:min_len_ret]
            dir_matches = sum(1 for a, p in zip(actual_returns_trim, pred_returns_trim) if (a >= 0 and p >= 0) or (a < 0 and p < 0))
            dir_accuracy = (dir_matches / min_len_ret * 100.0) if min_len_ret > 0 else 0.0
            next_pred_return = pred_returns_trim[0] if len(pred_returns_trim) > 0 else 0.0

            actual_for_backtest = df["Close"].tail(min_len_ret + 1).tolist() if min_len_ret > 0 else []
            pred_for_backtest = pred_closes[:min_len_ret + 1] if len(pred_closes) >= (min_len_ret + 1) else pred_closes + ([pred_closes[-1]] * (min_len_ret + 1 - len(pred_closes)) if len(pred_closes) > 0 else [])
            actions, profits, win_rate, total_return = investment_backtest(actual_for_backtest, pred_for_backtest)

            st.subheader("🔮 Forecast Summary")
            predicted_close = float(pred_closes[0]) if len(pred_closes) > 0 else 0.0
            predicted_high = predicted_close * (1 + margin / 100.0)
            predicted_low = predicted_close * (1 - margin / 100.0)

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Close", f"₹{predicted_close:,.2f}")
            c2.metric("Predicted High", f"₹{predicted_high:,.2f}")
            c3.metric("Predicted Low", f"₹{predicted_low:,.2f}")

            st.subheader("📏 Model Evaluation Metrics (last horizon)")
            m1, m2, m3 = st.columns(3)
            m1.metric("MSE", f"{mse:,.4f}")
            m2.metric("RMSE", f"{rmse:,.4f}")
            m3.metric("Avg Error % (MAPE)", f"{avg_error_pct:.2f}%")

            st.subheader("📉 Return Forecasting (last horizon)")
            df_returns = pd.DataFrame({
                "Actual Return %": actual_returns_trim,
                "Predicted Return %": pred_returns_trim
            })
            if not df_returns.empty:
                st.dataframe(df_returns)
            st.metric("Directional Accuracy", f"{dir_accuracy:.2f}%")
            st.metric("Next Period Predicted Return %", f"{next_pred_return:.2f}%")

            st.subheader("💰 Investment Simulation (Backtest)")
            if actions and profits:
                df_bt = pd.DataFrame({"Action": actions, "Profit (₹)": profits})
                st.dataframe(df_bt)
                st.metric("Win Rate", f"{win_rate:.2f}%")
                st.metric("Total Profit/Loss (₹)", f"{total_return:.2f}")
            else:
                st.write("Not enough data to run backtest (need at least 2 actual points).")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["Date"], df["Close"], label="Historical Close", linewidth=2)
            if len(future_idx) > 0 and len(pred_closes) > 0:
                ax.plot(future_idx, pred_closes, label="Predicted Close", linestyle="--", linewidth=2, color="orange")
                ax.fill_between(future_idx, lower, upper, alpha=0.2, color="gray", label="Confidence Band")
            ax.set_title(f"{ticker} Forecast — Interval: {interval}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            if live_mode:
                st.info(f"🔄 Live mode active — updating every {refresh_rate} seconds")
                chart_placeholder = st.empty()
                metrics_placeholder = st.container()
                ticker_placeholder = st.empty()
                backtest_placeholder = st.empty()

                while True:
                    try:
                        df = fetch_yfinance(ticker, start=start, end=datetime.date.today(), interval=interval)
                        if df is None or df.empty:
                            st.warning("No data returned in live update.")
                            time.sleep(refresh_rate)
                            continue

                        klines = df_to_klines(df.tail(240 if interval == "1h" else 200))
                        result = handler.predict(history_klines=klines, horizon_klines=horizon)
                        pred_klines = result.get("klines", [])
                        pred_closes = klines_to_closes(pred_klines)
                        pred_closes = [float(x) for x in pred_closes]
                        future_idx = build_future_dates(df["Date"].iloc[-1], periods=horizon)
                        lower, upper = compute_bands(pred_closes, margin)

                        latest_open = float(df["Open"].iloc[-1]) if "Open" in df.columns else 0.0
                        latest_high = float(df["High"].iloc[-1]) if "High" in df.columns else 0.0
                        latest_low = float(df["Low"].iloc[-1]) if "Low" in df.columns else 0.0
                        latest_close = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0.0
                        predicted_close = float(pred_closes[0]) if len(pred_closes) > 0 else 0.0
                        predicted_high = predicted_close * (1 + margin / 100.0)
                        predicted_low = predicted_close * (1 - margin / 100.0)
                        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_close
                        change_pct = ((latest_close - prev_close) / prev_close * 100.0) if prev_close != 0 else 0.0

                        ticker_data = [{
                            "symbol": ticker,
                            "open": latest_open,
                            "high": latest_high,
                            "low": latest_low,
                            "close": latest_close,
                            "change": change_pct,
                        }]
                        ticker_placeholder.empty()
                        display_scrolling_ticker(ticker_data)

                        actual_live_vals = df["Close"].tail(horizon).tolist()
                        min_len_live = min(len(actual_live_vals), len(pred_closes))
                        if min_len_live > 0:
                            actual_live_trim = actual_live_vals[:min_len_live]
                            pred_live_trim = pred_closes[:min_len_live]
                            live_mse = sum((a - p)**2 for a, p in zip(actual_live_trim, pred_live_trim)) / min_len_live
                            live_rmse = live_mse ** 0.5
                            live_avg_error_pct = sum(abs((a - p) / a) * 100 for a, p in zip(actual_live_trim, pred_live_trim)) / min_len_live
                        else:
                            live_mse = live_rmse = live_avg_error_pct = 0.0

                        actual_for_returns_live = df["Close"].tail(horizon + 1).tolist()
                        pred_returns_live = compute_returns(([df["Close"].iloc[-1]] + pred_closes) if len(pred_closes) > 0 else pred_closes)[:horizon]
                        actual_returns_live = compute_returns(actual_for_returns_live)[:horizon]
                        min_len_ret_live = min(len(actual_returns_live), len(pred_returns_live))
                        if min_len_ret_live > 0:
                            dir_matches_live = sum(1 for a, p in zip(actual_returns_live[:min_len_ret_live], pred_returns_live[:min_len_ret_live]) if (a >= 0 and p >= 0) or (a < 0 and p < 0))
                            dir_acc_live = dir_matches_live / min_len_ret_live * 100.0
                            next_pred_ret_live = pred_returns_live[0]
                        else:
                            dir_acc_live = 0.0
                            next_pred_ret_live = 0.0

                        actual_for_backtest_live = df["Close"].tail(min_len_ret_live + 1).tolist() if min_len_ret_live > 0 else []
                        pred_for_backtest_live = pred_closes[:min_len_ret_live + 1] if len(pred_closes) >= (min_len_ret_live + 1) else pred_closes + ([pred_closes[-1]] * (min_len_ret_live + 1 - len(pred_closes)) if len(pred_closes) > 0 else [])
                        actions_live, profits_live, win_rate_live, total_return_live = investment_backtest(actual_for_backtest_live, pred_for_backtest_live)

                        metrics_placeholder.empty()
                        with metrics_placeholder:
                            cols = st.columns(4)
                            cols[0].metric("Latest Close", f"₹{latest_close:,.2f}")
                            cols[1].metric("Predicted Close", f"₹{predicted_close:,.2f}")
                            cols[2].metric("Live RMSE", f"{live_rmse:.4f}")
                            cols[3].metric("Live Avg Error %", f"{live_avg_error_pct:.2f}%")

                            cols2 = st.columns(3)
                            cols2[0].metric("Live Directional Acc.", f"{dir_acc_live:.2f}%")
                            cols2[1].metric("Live Next Pred Return %", f"{next_pred_ret_live:.2f}%")
                            if actions_live:
                                    cols2[2].metric("Live Win Rate", f"{win_rate_live:.2f}%")
                            else:
                                    cols2[2].write("")


                            if actions_live:
                                st.write("Live Backtest (recent):")
                                st.dataframe(pd.DataFrame({"Action": actions_live, "Profit (₹)": profits_live}))

                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(df["Date"], df["Close"], label="Historical Close", linewidth=2)
                        if len(future_idx) > 0 and len(pred_closes) > 0:
                            ax.plot(future_idx, pred_closes, label="Predicted Close", linestyle="--", linewidth=2, color="orange")
                            ax.plot(future_idx, [predicted_high] * len(future_idx), linestyle=":", color="green", label="Predicted High")
                            ax.plot(future_idx, [predicted_low] * len(future_idx), linestyle=":", color="red", label="Predicted Low")
                            ax.fill_between(future_idx, [predicted_low] * len(future_idx), [predicted_high] * len(future_idx), alpha=0.15, color="gray")
                        ax.set_title(f"📊 Live {ticker} Forecast — Interval: {interval}")
                        ax.legend()
                        ax.grid(True)
                        chart_placeholder.pyplot(fig)

                        time.sleep(refresh_rate)
                    except KronosError as ke:
                        st.error(f"Kronos Error (live): {ke}")
                        break
                    except Exception as le:
                        st.error(f"Live update error: {le}")
                        import traceback
                        st.text(traceback.format_exc())
                        break

    except KronosError as ke:
        st.error(f"Kronos Error: {ke}")
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        import traceback
        st.text(traceback.format_exc())
