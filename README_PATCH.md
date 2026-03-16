# Patch: Hourly/Daily Streamlit App

Added `app_hourly.py` that supports:
- Interval selection: 1d or 1h
- Adjustable history period and horizon
- ±Margin bands around predicted closes
- CSV download of predictions

## Run
```bash
pip install -r requirements.txt
streamlit run app_hourly.py
```
