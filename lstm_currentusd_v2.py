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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_today_avg():
    """
    Load today's average USD price from API, with error handling.
    Returns float or None.
    """
    url = "https://example.com/api/today_avg"  # ← آدرس واقعی API را اینجا بگذار
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("avg_price", 0))
    except requests.exceptions.HTTPError as he:
        st.error(f"سرور خطا داد (HTTP {he.response.status_code})")
    except requests.exceptions.Timeout:
        st.error("درخواست تایم‌اوت شد. لطفاً بعداً تلاش کنید.")
    except requests.exceptions.RequestException as e:
        st.error(f"خطای شبکه: {e}")
    return None

# -------------------------------------------------------------------
def load_historical_data():
    # نمونه بارگذاری داده‌های گذشته از فایل CSV
    df = pd.read_csv("data/historical_usd.csv", parse_dates=["date"])
    df = df.sort_values("date")
    return df

# -------------------------------------------------------------------
def build_and_train_model(series, n_lags=30, epochs=50, batch_size=16):
    """
    series: pandas.Series indexed by date
    returns trained keras model and scaler
    """
    # مقیاس‌بندی
    scaler = MinMaxScaler()
    values = scaler.fit_transform(series.values.reshape(-1, 1))
    # ساخت داده برای LSTM
    X, y = [], []
    for i in range(n_lags, len(values)):
        X.append(values[i - n_lags:i, 0])
        y.append(values[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # مدل
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(n_lags, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, 
              validation_split=0.1, callbacks=[es], verbose=0)
    return model, scaler

# -------------------------------------------------------------------
def forecast_next_days(model, scaler, series, n_lags=30, days_ahead=7):
    """
    سری را ادامه می‌دهد و پیش‌بینی می‌کند
    """
    last_vals = scaler.transform(series.values[-n_lags:].reshape(-1,1)).flatten().tolist()
    preds = []
    for _ in range(days_ahead):
        x_input = np.array(last_vals[-n_lags:]).reshape((1, n_lags, 1))
        yhat = model.predict(x_input, verbose=0)[0,0]
        preds.append(yhat)
        last_vals.append(yhat)
    # معکوس مقیاس
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    dates = [series.index[-1] + timedelta(days=i+1) for i in range(days_ahead)]
    return pd.Series(preds, index=dates)

# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="USD Price LSTM Forecast", layout="wide")
    st.title("📈 پیش‌بینی نرخ دلار با LSTM")

    # ۱) مقدار امروز
    avg_price = load_today_avg()
    if avg_price is not None:
        st.metric("قیمت میانگین امروز (USD)", f"{avg_price:.2f}")
    else:
        st.warning("قیمت امروز در دسترس نیست.")

    # ۲) داده‌های گذشته
    df = load_historical_data()
    df.set_index("date", inplace=True)

    st.subheader("نمودار تاریخی USD")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["price"], label="Historic Price")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # ۳) آموزش مدل
    with st.spinner("در حال آموزش مدل LSTM..."):
        model, scaler = build_and_train_model(df["price"])
    st.success("مدل آموزش داده شد!")

    # ۴) پیش‌بینی ۷ روز آینده
    future = forecast_next_days(model, scaler, df["price"], days_ahead=7)
    st.subheader("پیش‌بینی ۷ روز آینده")
    fig2, ax2 = plt.subplots()
    ax2.plot(df.index, df["price"], label="Historic")
    ax2.plot(future.index, future.values, "--", label="Forecast")
    ax2.legend()
    st.pyplot(fig2)

    # ۵) ارزیابی ساده (در صورت داشتن داده واقعی)
    # actual = ...  # اگر داده واقعی برای ۷ روز آینده داری
    # mae = mean_absolute_error(actual, future)
    # mape = mean_absolute_percentage_error(actual, future)
    # st.write(f"MAE: {mae:.3f}, MAPE: {mape:.3%}")

if __name__ == "__main__":
    main()
