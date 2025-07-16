import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

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
    # بارگذاری داده‌های گذشته از فایل CSV
    df = pd.read_csv("data/historical_usd.csv", parse_dates=["date"])
    df = df.sort_values("date")
    return df

# -------------------------------------------------------------------
def prepare_features(series, n_lags=30):
    """
    با استفاده از پنجره‌ی n_lags، ماتریس ویژگی X و بردار هدف y را بسازد
    """
    X, y = [], []
    vals = series.values
    for i in range(n_lags, len(vals)):
        X.append(vals[i-n_lags:i])
        y.append(vals[i])
    return np.array(X), np.array(y)

# -------------------------------------------------------------------
def train_random_forest(series, n_lags=30, n_estimators=100, max_depth=10):
    """
    سری را با Random Forest مدل می‌کند و مدل را برمی‌گرداند
    """
    # مقیاس‌بندی
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1)).flatten()
    # تهیه ویژگی‌ها
    X, y = prepare_features(pd.Series(scaled), n_lags=n_lags)
    # مدل جنگل تصادفی
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X, y)
    return rf, scaler

# -------------------------------------------------------------------
def forecast_next_days(model, scaler, series, n_lags=30, days_ahead=7):
    """
    سری را ادامه می‌دهد و پیش‌بینی می‌کند
    """
    scaled = scaler.transform(series.values.reshape(-1,1)).flatten()
    window = list(scaled[-n_lags:])
    preds = []
    for _ in range(days_ahead):
        x = np.array(window[-n_lags:]).reshape(1, -1)
        yhat = model.predict(x)[0]
        preds.append(yhat)
        window.append(yhat)
    # معکوس مقیاس
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    dates = [series.index[-1] + timedelta(days=i+1) for i in range(days_ahead)]
    return pd.Series(preds, index=dates)

# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="USD Price Random Forest Forecast", layout="wide")
    st.title("🌲 پیش‌بینی نرخ دلار با Random Forest")

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

    # ۳) آموزش مدل Random Forest
    with st.spinner("در حال آموزش مدل Random Forest..."):
        rf_model, scaler = train_random_forest(df["price"])
    st.success("مدل آموزش داده شد!")

    # ۴) پیش‌بینی ۷ روز آینده
    future = forecast_next_days(rf_model, scaler, df["price"], days_ahead=7)
    st.subheader("پیش‌بینی ۷ روز آینده")
    fig2, ax2 = plt.subplots()
    ax2.plot(df.index, df["price"], label="Historic")
    ax2.plot(future.index, future.values, "--", label="Forecast")
    ax2.legend()
    st.pyplot(fig2)

    # ۵) (اختیاری) ارزیابی اگر داده واقعی داشته باشی
    # actual = ...  
    # mae = mean_absolute_error(actual, future)
    # mape = mean_absolute_percentage_error(actual, future)
    # st.write(f"MAE: {mae:.3f}, MAPE: {mape:.3%}")

if __name__ == "__main__":
    main()
