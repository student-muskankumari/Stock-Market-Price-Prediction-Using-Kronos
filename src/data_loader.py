# src/data_loader.py
import pandas as pd
import yfinance as yf
import streamlit as st

@st.cache_data(show_spinner=False)
def fetch_yfinance(ticker, start=None, end=None, interval="1d"):
    """
    Fetch and normalize stock data from Yahoo Finance.
    Always returns columns: Date, Open, High, Low, Close, Volume
    """

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        group_by=False
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for '{ticker}' at interval '{interval}'.")


    df = df.reset_index()

   
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns]

   
    df.columns = [str(c).strip().replace(" ", "_").capitalize() for c in df.columns]

  
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if "open" in low: col_map[col] = "Open"
        elif "high" in low: col_map[col] = "High"
        elif "low" in low: col_map[col] = "Low"
        elif "close" in low and "adj" not in low: col_map[col] = "Close"
        elif "volume" in low: col_map[col] = "Volume"
        elif "adj_close" in low: col_map[col] = "Close"

    df = df.rename(columns=col_map)

    
    if "Date" not in df.columns:
        possible_dates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if possible_dates:
            df.rename(columns={possible_dates[0]: "Date"}, inplace=True)
        else:
            raise ValueError(f"Could not find any 'Date' column in columns: {df.columns.tolist()}")


    keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

   
    missing = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"⚠️ Yahoo returned data missing expected columns: {missing}\n"
            f"Columns found: {df.columns.tolist()}\n"
            f"Sample data:\n{df.head()}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)

    return df
