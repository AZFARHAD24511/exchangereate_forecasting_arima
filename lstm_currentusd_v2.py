import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from datetime import datetime, timedelta
from io import StringIO
from pytrends.request import TrendReq
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Streamlit page config
st.set_page_config(page_title="پیش‌بینی نرخ دلار آزاد تهران با Random Forest 📈", layout="wide")
st.markdown("""
---
📈 © 2025 Dr. Farhadi. All rights reserved.  
This application was developed by **Dr. Farhadi**, Ph.D. in *Economics (Econometrics)* and *Data Science*.  
All trademarks and intellectual property are protected. ™
""")
st.title("📈 پیش‌بینی نرخ دلار آزاد با Random Forest 📈")

# Constants
github_trends_url = (
    'https://raw.githubusercontent.com/AZFARHAD24511/exchange_rates_IRAN/main/'
    'predict/google_trends_daily_exrusd.csv'
)
KEYWORDS = ['خرید دلار', 'فروش دلار', 'دلار فردایی']

# Function to load historical USD data
@st.cache_data(ttl=3600)
def load_usd_data():
    ts = int(datetime.now().timestamp() * 1000)
    url = (
        f"https://api.tgju.org/v1/market/indicator/"
        f"summary-table-data/price_dollar_rl?period=all&mode=full&ts={ts}"
    )
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    data = r.json().get('data', [])
    records = []
    for row in data:
        try:
            price = float(
                re.sub(r'<[^>]*>', '', row[0]).replace(',', '')
            )
            date = datetime.strptime(row[6], "%Y/%m/%d")
            records.append({'date': date, 'price': price})
        except:
            continue
    df = pd.DataFrame(records).set_index('date').sort_index()
    return df

# Function to load Google Trends data
@st.cache_data(ttl=3600)
def load_trends_csv():
    r = requests.get(github_trends_url)
    df = pd.read_csv(StringIO(r.text), parse_dates=['date'])
    return df.set_index('date').sort_index()

# Fetch missing trends if any dates missing
@st.cache_data(ttl=3600, hash_funcs={pd.DatetimeIndex: lambda idx: idx.astype(str).tolist()})
def fetch_missing_trends(missing_dates, geo='IR'):
    if not isinstance(missing_dates, pd.DatetimeIndex):
        missing_dates = pd.to_datetime(list(missing_dates))
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

# Function to load today's data and compute avg of last 5 prices
@st.cache_data(ttl=300)
def load_today_avg():
    url = "https://api.tgju.org/v1/market/indicator/today-table-data/price_dollar_rl"
    params = {
        "lang": "fa",
        "draw": 1,
        "start": 0,
        "length": 30,
        "today_table_tolerance_open": 1,
        "today_table_tolerance_yesterday": 1,
        "today_table_tolerance_range": "week",
        "_": int(pd.Timestamp.now().timestamp() * 1000)
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json().get('data', [])
    prices = []
    for row in data:
        raw = row[0]
        clean = int(re.sub(r'<[^>]*>', '', raw).replace(',', ''))
        prices.append(clean)
    # prices sorted descending by time, take first 5
    last5 = prices[:5]
    return sum(last5) / len(last5) if last5 else np.nan

# Load data
with st.spinner("در حال بارگذاری داده‌ها..."):
    usd_df = load_usd_data()
    # Append today's average price
    today = datetime.now().date()
    avg_price = load_today_avg()
    if not np.isnan(avg_price):
        usd_df.loc[pd.to_datetime(today)] = avg_price
    trends_df = load_trends_csv()

# Use last 2 years
two_years_ago = datetime.now() - timedelta(days=730)
udf = usd_df[usd_df.index >= two_years_ago]
trf = trends_df[trends_df.index >= two_years_ago]

# Fill missing trend dates
missing = udf.index.difference(trf.index)
if not missing.empty:
    new_tr = fetch_missing_trends(tuple(date.strftime('%Y-%m-%d') for date in missing))
    trf = pd.concat([trf, new_tr]).sort_index()
    trf = trf.reindex(udf.index).ffill().bfill()

# Merge datasets
df = pd.merge(udf, trf, left_index=True, right_index=True, how='inner').ffill().bfill()

# Feature Engineering - Create lag features
SEQ_LEN = 30
for i in range(1, SEQ_LEN + 1):
    df[f'lag_{i}'] = df['price'].shift(i)

# Drop rows with NaN values (due to lag features)
df = df.dropna()

# Prepare features and target
X = df.drop(columns=['price'])
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# Build Random Forest model
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Train model
rf.fit(X_train, y_train)

# Predictions and metrics
preds = rf.predict(X_test)
mae = mean_absolute_error(y_test, preds)
mape = mean_absolute_percentage_error(y_test, preds) * 100

# Forecast next 2 days
def forecast_next(model, last_values, steps=2):
    forecasts = []
    current_features = last_values.copy()
    
    for _ in range(steps):
        # Predict next value
        next_value = model.predict(current_features.reshape(1, -1))[0]
        forecasts.append(next_value)
        
        # Update features for next prediction
        current_features = np.roll(current_features, 1)
        current_features[0] = next_value
    
    return forecasts

# Prepare last sequence for forecasting
last_values = df.iloc[-1][1:].values  # Exclude the current price
forecast = forecast_next(rf, last_values, steps=2)
forecast_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 3)]

# Display results
st.info(f"MAE: {mae:,.2f}    MAPE: {mape:.2f}%")
for d, v in zip(forecast_dates, forecast):
    st.success(f"🔮 نرخ دلار برای {d.date()}: {v:,.0f} ریال")

# Plot
st.subheader("📊 Historical & Random Forest Forecast")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['price'], label='Historical')
ax.plot(X_test.index, preds, label='Test Predictions', alpha=0.7)
ax.axvline(df.index[-1], linestyle='--')
for i, (d, v) in enumerate(zip(forecast_dates, forecast), 1):
    ax.scatter(d, v, color='red')
    ax.annotate(f'Day+{i}: {v:,.0f}', xy=(d, v), xytext=(0,10), 
                textcoords='offset points', ha='center', arrowprops=dict(arrowstyle='->'))
ax.set_title('USD Free Market Rate Random Forest Forecast')
ax.legend()
ax.grid(True)
st.pyplot(fig)
