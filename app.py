
import streamlit as st
import datetime
import pandas as pd
import io
from src.data_loader import fetch_yfinance, load_csv
from src.forecast_utils import df_to_klines, klines_to_closes, build_future_dates, evaluate
from src.model_handler_kronos import KronosModelHandler, KronosError
from src.visualization import plot_history_and_pred

st.set_page_config(page_title="Kronos Stock Forecast", layout="wide")
st.title("Kronos Stock Forecast (K-line foundation model)")


st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker (or leave blank to use CSV):", value="AMZN")
use_csv = st.sidebar.checkbox("Use local CSV (if checked, Ticker ignored)")
csv_path = st.sidebar.text_input("CSV path (if using CSV)", value="data/AMZN.csv")
start_date = st.sidebar.date_input("Start date", value=datetime.date.today() - datetime.timedelta(days=365*2))
end_date = st.sidebar.date_input("End date", value=datetime.date.today())
pred_horizon = st.sidebar.number_input("Prediction horizon (klines)", value=10, min_value=1, max_value=365)
model_id = st.sidebar.text_input("Kronos model id (HF)", value="NeoQuasar/Kronos-mini")
device = st.sidebar.selectbox("Device", options=["cpu", "cuda"], index=0)
run = st.sidebar.button("Run forecast")

st.markdown("**Notes:** For precise numeric decoding you should install the Kronos tokenizer from the Kronos GitHub or use the HF Kronos-Tokenizer model. The app will attempt to load the tokenizer automatically.")

if run:
    try:
        if use_csv:
            df = load_csv(csv_path)
        else:
            df = fetch_yfinance(ticker, start=start_date, end=end_date, interval="1d")
        st.success(f"Loaded {len(df)} rows.")
        st.subheader("History (last 200 rows)")
        st.dataframe(df.tail(200))

        klines = df_to_klines(df)
       
        st.write("Sample kline:", klines[-3:])

       
        with st.spinner("Loading Kronos model/tokenizer (may download weights)..."):
            handler = KronosModelHandler(model_id=model_id, device=device)

        with st.spinner("Running generation..."):
            res = handler.predict(history_klines=klines, horizon_klines=int(pred_horizon), temperature=0.0)
            preds = res["klines"]

        if len(preds) == 0:
            st.warning("Model returned no predicted klines. Possibly tokenizer not installed or decode failed.")
        else:
            pred_closes = klines_to_closes(preds)
            last_date = df["Date"].iloc[-1]
            future_dates = build_future_dates(last_date, periods=len(pred_closes))
            st.subheader("Predicted closes")
            st.write(pred_closes.tolist())

            fig = plot_history_and_pred(df, future_dates, pred_closes)
            st.pyplot(fig)

           
            st.download_button("Download predicted klines CSV",
                               data=pd.DataFrame(preds).to_csv(index=False).encode("utf-8"),
                               file_name="kronos_predicted_klines.csv",
                               mime="text/csv")
    except KronosError as ke:
        st.error("Kronos error: " + str(ke))
    except Exception as e:
        st.error("Unexpected error: " + str(e))
        import traceback
        st.text(traceback.format_exc())
