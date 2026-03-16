import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_history_and_pred(df, future_idx, pred_closes):
    fig, ax = plt.subplots(figsize=(10, 5))

    
    ax.plot(df["Date"], df["Close"], label="history", color="tab:blue", linewidth=2)

   
    pred_array = np.array(pred_closes, dtype="object").flatten()
    if len(pred_array) == len(future_idx):
        ax.plot(future_idx, pred_array, label="predicted", color="tab:orange", marker="o", linewidth=2)

    
    try:
        last_date = pd.to_datetime(df["Date"].iloc[-1])
        last_close = float(df["Close"].iloc[-1])
        first_pred_date = pd.to_datetime(future_idx[0])
        first_pred_close = float(pred_array[0])
        ax.plot(
            [last_date, first_pred_date],
            [last_close, first_pred_close],
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
        )
    except Exception as e:
        print("Warning: skipping connecting line due to data shape mismatch:", e)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("History (Close) and Predicted Closes")
    ax.legend()
    plt.tight_layout()
    return fig
