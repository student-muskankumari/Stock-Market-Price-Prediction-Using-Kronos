# Stock Market Price Prediction using KRONOS

A hybrid **AI-powered stock forecasting system** that combines a **time-series foundation model (KRONOS)** with **classical machine learning baselines** to predict future stock prices using historical market data.

This project demonstrates how modern **Transformer-based time-series models** can be integrated into practical financial forecasting pipelines.



# 🚀 Features

✔ Real-time stock data fetching using **Yahoo Finance API**
✔ **KRONOS time-series foundation model** integration
✔ **Fallback forecasting model** for reliability
✔ **RandomForest baseline model** for comparison
✔ Interactive **Streamlit dashboard**
✔ Multi-ticker forecasting (AAPL, NVDA, TSLA, MSFT etc.)
✔ Visualization of **historical vs predicted prices**
✔ Evaluation metrics: MAE, RMSE, MAPE, Direction Accuracy



#  System Architecture

```
User Input (Ticker)
        │
        ▼
Data Loader (yfinance)
        │
        ▼
Preprocessing (OHLCV → K-lines)
        │
        ▼
Forecast Engine
 ├─ KRONOS Model
 └─ Fallback Linear Predictor
        │
        ▼
Prediction Results
        │
        ▼
Visualization Dashboard (Streamlit)
```



#  Example Prediction

The system generates forecasts showing:

* Historical stock prices
* Predicted future trend
* Confidence bounds

These predictions help analyze **trend behavior and potential market direction**.



#  Tech Stack

### Languages

* Python

### Libraries

* PyTorch
* HuggingFace Transformers
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Streamlit
* yfinance

### Models

* KRONOS Time-Series Foundation Model
* RandomForest Regressor
* Linear Trend Fallback Predictor



# Project Structure

```
Stock-Market-Price-Prediction-Using-Kronos
│
├── src/
│   ├── data_loader.py
│   ├── forecast_utils.py
│   ├── model_handler_kronos.py
│   └── visualization.py
│
├── config/
├── eval_out/
│
├── app.py
├── app_hourly.py
├── train_model.py
├── evaluate.py
├── setup_checker.py
│
├── requirements.txt
├── environment.yml
└── README.md
```


# Setup Instructions

## 1. Create Environment

Using **Conda (recommended)**

```
conda create -n kronos python=3.10 -y
conda activate kronos
```



## 2. Install PyTorch

### GPU

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### CPU

```
pip install torch torchvision
```



## 3. Install Dependencies

```
pip install -r requirements.txt
```


## 4. Install Kronos Tokenizer

```
pip install git+https://github.com/student-muskankumari/Stock-Market-Price-Prediction-Using-Kronos.git
```



## 5. Verify Setup

```
python setup_checker.py
```



## 6. Run the App

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```



# Data Source

The application fetches stock market data from:

**Yahoo Finance API (via yfinance)**

Supported tickers:

* AAPL
* NVDA
* TSLA
* MSFT
* AMZN
* Any valid Yahoo Finance ticker



# Evaluation Metrics

Model performance is evaluated using:

| Metric             | Description                             |
| ------------------ | --------------------------------------- |
| MAE                | Mean Absolute Error                     |
| RMSE               | Root Mean Squared Error                 |
| MAPE               | Mean Absolute Percentage Error          |
| Direction Accuracy | % of correct price movement predictions |
| Sharpe Ratio       | Risk-adjusted return metric             |


# Limitations

* Highly volatile stocks (e.g., TSLA) remain difficult to forecast.
* Predictions rely only on **historical OHLCV data**.
* News sentiment and macroeconomic signals are not included.

Future work could integrate:

* News sentiment analysis
* Reinforcement learning trading strategies
* Crypto & forex forecasting



# Author

**Muskan Kumari**

Computer Science Student
KIIT University

GitHub
https://github.com/student-muskankumari


# ⭐ If you find this project useful

Please consider **starring the repository** ⭐
