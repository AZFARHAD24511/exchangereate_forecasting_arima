import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from datetime import datetime, timedelta
from io import StringIO
from pytrends.request import TrendReq
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

# Streamlit config
st.set_page_config(page_title="پیش‌بینی نرخ دلار آزاد با ARIMA + XGBoost 📈", layout="wide")
st.markdown("""
---
📈 © 2025 Dr. Farhadi. All rights reserved.  
This application was developed by **Dr. Farhadi**, Ph.D. in *Economics (Econometrics)* and *Data Science*.  
All trademarks and intellectual property are protected. ™
""")
st.title("📈 پیش‌بینی نرخ دلار آزاد با ARIMA + XGBoost 📈")

# Constants
github_trends_url = (
    'https://raw.githubusercontent.com/AZFARHAD24511/exchange_rates_IRAN/main/'
    'predict/google_trends_daily_exrusd.csv'
)
KEYWORDS = ['خرید دلار', 'فروش دلار', 'دلار فردایی']

# Data loading functions
def load_usd_data():
    ts = int(datetime.now().timestamp() * 1000)
    url = (
        f"https://api.tgju.org/v1/market/indicator/"
        f"summary-table-data/price_dollar_rl?period=all&mode=full&ts={ts}"
    )
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    records = []
    for row in r.json().get('data', []):
        try:
            price = float(re.sub(r'<[^>]*>', '', row[0]).replace(',', ''))
            date = datetime.strptime(row[6], "%Y/%m/%d")
            records.append({'date': date, 'price': price})
        except:
            continue
    return pd.DataFrame(records).set_index('date').sort_index()

def load_trends_csv():
    r = requests.get(github_trends_url)
    df = pd.read_csv(StringIO(r.text), parse_dates=['date'])
    return df.set_index('date').sort_index()

@st.cache_data(ttl=3600)
def fetch_missing_trends(missing_dates, geo='IR'):
    pytrends = TrendReq(hl='fa', tz=330)
    df_list = []
    start, end = missing_dates.min(), missing_dates.max()
    timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
    for kw in KEYWORDS:
        pytrends.build_payload([kw], timeframe=timeframe, geo=geo)
        tmp = pytrends.interest_over_time()
        if not tmp.empty:
            df_list.append(tmp[kw].rename(kw))
    if df_list:
        df_new = pd.concat(df_list, axis=1).loc[missing_dates]
        return df_new.apply(lambda x: x / x.max() * 100)
    return pd.DataFrame(index=missing_dates)

# Load data
with st.spinner("در حال بارگذاری داده‌ها..."):
    usd_df = load_usd_data()
    trends_df = load_trends_csv()

# Prepare 2-year window
two_years = datetime.now() - timedelta(days=730)
udf = usd_df[usd_df.index >= two_years]
trf = trends_df[trends_df.index >= two_years]

# Fill missing trends
missing = udf.index.difference(trf.index)
if not missing.empty:
    trf_missing = fetch_missing_trends(missing)
    trf = pd.concat([trf, trf_missing]).sort_index().reindex(udf.index).ffill().bfill()

# Merge
df = pd.merge(udf, trf, left_index=True, right_index=True, how='inner').ffill().bfill()
series = df['price']

# Feature engineering for XGB
def make_exog(data, lags=7):
    exog = pd.DataFrame(index=data.index)
    # lagged price features
    for i in range(1, lags+1):
        exog[f'lag_{i}'] = data.shift(i)
    # rolling stats
    exog['roll_mean_7'] = data.shift(1).rolling(7).mean()
    exog['roll_std_7'] = data.shift(1).rolling(7).std()
    # date part
    exog['dow'] = data.index.dayofweek
    exog['day'] = data.index.day
    exog['month'] = data.index.month
    # trends
    for kw in KEYWORDS:
        exog[kw] = df[kw]
    exog = exog.dropna()
    return exog

exog = make_exog(series)
y_aligned = series.loc[exog.index]

# Train-test split by time index
split = int(len(exog)*0.8)
train_exog = exog.iloc[:split]
test_exog = exog.iloc[split:]
train_y = y_aligned.iloc[:split]
test_y = y_aligned.iloc[split:]

# 1. Fit SARIMAX on train
sarimax = sm.tsa.SARIMAX(
    train_y,
    order=(1,1,1),
    seasonal_order=(1,0,1,12),
    exog=train_exog
).fit(disp=False)
train_pred_arima = sarimax.fittedvalues
residuals = train_y - train_pred_arima

# 2. Train XGB on residuals
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(train_exog, residuals)

# 3. Predict test
arima_pred_test = sarimax.predict(start=test_exog.index[0], end=test_exog.index[-1], exog=test_exog)
xgb_pred_test = xgb_model.predict(test_exog)
final_pred = arima_pred_test + xgb_pred_test

# Metrics
mae = mean_absolute_error(test_y, final_pred)
mape = mean_absolute_percentage_error(test_y, final_pred) * 100

# Forecast next 2 days
last_exog = exog.tail(7).copy()
future_exogs = []
for i in range(1, 3):
    date = df.index[-1] + timedelta(days=i)
    row = {}
    # update lags
    for j in range(1, 8):
        if j < 7:
            row[f'lag_{j}'] = last_exog[f'lag_{j+1}'].iloc[-1]
        else:
            row['lag_7'] = final_pred.iloc[-1]
    # rolling stats
    row['roll_mean_7'] = final_pred.tail(7).mean()
    row['roll_std_7'] = final_pred.tail(7).std()
    # date features
    row['dow'] = date.dayofweek
    row['day'] = date.day
    row['month'] = date.month
    # trends
    for kw in KEYWORDS:
        row[kw] = df[kw].iloc[-1]
    future_exogs.append((date, row))
future_df = pd.DataFrame({d: pd.Series(r) for d, r in future_exogs}).T
future_df.index = [d for d, _ in future_exogs]

arima_future = sarimax.predict(start=future_df.index[0], end=future_df.index[-1], exog=future_df)
xgb_future = xgb_model.predict(future_df)
future_pred = arima_future + xgb_future

# Display
st.info(f"MAE: {mae:.2f}   MAPE: {mape:.2f}%")
for d, p in zip(future_pred.index, future_pred.values):
    st.success(f"🔮 نرخ دلار برای {d.date()}: {p:.0f} ریال")

# Plot
st.subheader("📊 Historical vs ARIMA+XGB Fit & Forecast")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(series, label='Historical')
ax.plot(train_pred_arima + xgb_model.predict(train_exog), label='Train Fit', alpha=0.7)
ax.plot(final_pred.index, final_pred, label='Test Prediction', alpha=0.9)
ax.scatter(future_pred.index, future_pred, color='red', label='Forecast')
ax.axvline(series.index[split], ls='--', color='gray')
ax.legend()
st.pyplot(fig)
