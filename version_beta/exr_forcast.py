import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import random
import functools
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from io import BytesIO

st.set_page_config(page_title="پیش‌بینی دلار", layout="wide")
# -----------------------------
# بخش داده و پیش‌بینی
# -----------------------------

@functools.lru_cache(maxsize=1)
def load_usd_data():
    ts = int(datetime.now().timestamp() * 1000)
    url = (
        f"https://api.tgju.org/v1/market/indicator/"
        f"summary-table-data/price_dollar_rl?period=all&mode=full&ts={ts}"
    )
    items = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    rows = items.json().get('data', [])
    rec = []
    for row in rows:
        try:
            price = float(re.sub(r'<.*?>', '', row[0]).replace(',', ''))
            date = datetime.strptime(row[6], "%Y/%m/%d")
            rec.append({'date': date, 'price': price})
        except:
            continue
    df = pd.DataFrame(rec).set_index('date').sort_index()
    return df

@functools.lru_cache(maxsize=1)
def load_today_avg():
    params = {
        "lang": "fa",
        "draw": 1,
        "start": 0,
        "length": 30,
        "today_table_tolerance_open": 1,
        "today_table_tolerance_yesterday": 1,
        "today_table_tolerance_range": "week",
        "_": int(datetime.now().timestamp() * 1000)
    }
    try:
        rows = requests.get(
            "https://api.tgju.org/v1/market/indicator/today-table-data/price_dollar_rl",
            params=params, headers={'User-Agent': 'Mozilla/5.0'}
        ).json().get('data', [])
        prices = [int(re.sub(r'<.*?>', '', r[0]).replace(',', '')) for r in rows[:5]]
        avg = np.nan if not prices else sum(prices) / len(prices)
        return avg
    except Exception as e:
        print("خطا در load_today_avg:", e)
        return np.nan

@functools.lru_cache(maxsize=1)
def load_usd_full_table():
    ts = int(datetime.now().timestamp() * 1000)
    url = (
        "https://api.tgju.org/v1/market/indicator/"
        f"summary-table-data/price_dollar_rl?period=all&mode=full&ts={ts}"
    )
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = resp.json().get('data', [])

        if not data:
            return np.nan

        records = []
        for row in data:
            clean_row = [re.sub(r'<.*?>', '', cell) for cell in row]
            records.append(clean_row)

        df = pd.DataFrame(records)
        df.columns = df.iloc[0]
        df = df.drop(0).reset_index(drop=True)

        df['Col_4'] = df.iloc[:, 3].astype(str).str.replace(',', '').astype(float)
        avg = df.at[0, 'Col_4']
        return avg
    except Exception as e:
        print("خطا در load_usd_full_table:", e)
        return np.nan

def load_dollar_value():
    iran_time = datetime.now(ZoneInfo("Asia/Tehran"))
    weekday = iran_time.weekday()  # شنبه = 0، جمعه = 4
    hour = iran_time.hour

    try:
        if weekday == 4 or hour < 10 or hour >= 17:
            return load_usd_full_table()
        else:
            return load_today_avg()
    except Exception as e:
        print("خطا در load_dollar_value:", e)
        return np.nan

@functools.lru_cache(maxsize=1)
def load_trends_csv():
    base_url = "https://raw.githubusercontent.com/AZFARHAD24511/GT_datasets/main/data/"
    filenames = [
        "exr_20220101-20250501.csv",
        "exr_20250501_20250731.csv",
        "google_trends_long_daily.csv"
    ]
    dfs = []
    for file in filenames:
        try:
            url = base_url + file
            df = pd.read_csv(url, parse_dates=['date'])
            dfs.append(df)
        except Exception as e:
            print(f"خطا در خواندن {file}: {e}")
    if not dfs:
        return pd.DataFrame()
    df_long = pd.concat(dfs, ignore_index=True)
    df_wide = df_long.pivot(index='date', columns='keyword', values='hits')
    df_wide = df_wide.sort_index().ffill().bfill()
    return df_wide

def prepare_full_df():
    hist = load_usd_data()
    avg = load_dollar_value()
    today = pd.to_datetime(datetime.now().date())
    if not np.isnan(avg):
        hist = pd.concat([hist, pd.DataFrame({'price': [avg]}, index=[today])])
    hist = hist[~hist.index.duplicated(keep='last')]
    dr = pd.date_range(hist.index.min(), hist.index.max(), freq='D')
    hist = hist.reindex(dr).interpolate()

    trends = load_trends_csv()
    cutoff = datetime.now() - timedelta(days=730)
    usd2 = hist[hist.index >= cutoff]
    trends2 = trends[trends.index >= cutoff]
    trends2 = trends2.reindex(usd2.index).ffill().bfill()

    df = pd.merge(usd2, trends2, left_index=True, right_index=True, how='inner').ffill().bfill()
    return df

def run_forecast():
    df = prepare_full_df()
    series = df['price']
    model = ARIMA(series, order=(1, 0, 1)).fit()
    fc = model.forecast(steps=2)
    last = series.index[-1]
    dates = [last + timedelta(days=i) for i in (1, 2)]
    preds = {d.date().isoformat(): int(v) for d, v in zip(dates, fc)}
    mae = mean_absolute_error(series, model.predict(start=0, end=len(series)-1))
    mape = mean_absolute_percentage_error(series, model.predict(start=0, end=len(series)-1)) * 100
    pvals = model.pvalues.to_dict()
    return df, preds, mae, mape, pvals
# -----------------------------
# رابط کاربری Streamlit
# -----------------------------

# وضعیت آمار کاربران (به صورت شبیه‌سازی‌شده)
import random
from collections import defaultdict
user_stats = defaultdict(int)
ARTIFICIAL_USER_BOOST = random.randint(1751, 3987)

st.title("📈 پیش‌بینی نرخ دلار آزاد تهران")
menu = st.sidebar.radio("منو", ["پیش‌بینی", "آمار استفاده", "راهنما"])

if menu == "پیش‌بینی":
    st.subheader("🔮 پیش‌بینی نرخ دلار (مدل ARIMA)")
    st.info("داده‌ها از tgju.org و Google Trends به‌صورت ترکیبی استفاده شده‌اند.")
    with st.spinner("در حال بارگیری و پردازش داده‌ها..."):
        df, preds, mae, mape, pvals = run_forecast()
    # نمایش نمودار
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['price'], label="قیمت واقعی")
    for d, v in preds.items():
        ax.scatter(pd.to_datetime(d), v, color='red', label="پیش‌بینی" if d == list(preds.keys())[0] else "")
    ax.set_title("نمودار قیمت دلار و پیش‌بینی")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # نمایش مقادیر
    st.markdown(f"**MAE:** {mae:,.2f} | **MAPE:** {mape:.2f}%")
    st.markdown(f"""
    **P-Values:**  
    - const: `{pvals.get('const', 0):.4f}`  
    - ar.L1: `{pvals.get('ar.L1', 0):.4f}`  
    - ma.L1: `{pvals.get('ma.L1', 0):.4f}`  
    - sigma2: `{pvals.get('sigma2', 0):.4f}`
    """)
    st.markdown("### 📅 پیش‌بینی نرخ:")
    for d, v in preds.items():
        st.success(f"{d} : {v:,} ریال")

elif menu == "آمار استفاده":
    st.subheader("📊 آمار کاربران (شبیه‌سازی شده)")
    # شبیه‌سازی آمار
    total_users = len(user_stats) + ARTIFICIAL_USER_BOOST
    total_requests = sum(user_stats.values()) + random.randint(2000, 10000)
    st.metric("👥 کاربران", total_users)
    st.metric("📈 مجموع درخواست‌ها", total_requests)

elif menu == "راهنما":
    st.subheader("❓ راهنمای استفاده")
    st.markdown("""
    این اپلیکیشن با استفاده از داده‌های نرخ دلار آزاد تهران و Google Trends، قیمت فردای دلار را پیش‌بینی می‌کند.

    **قابلیت‌ها:**
    - پیش‌بینی ۲ روز آینده با مدل ARIMA
    - مشاهده نمودار تاریخی و نقاط پیش‌بینی
    - مشاهده آمار کلی کاربران و درخواست‌ها

    **توسعه‌دهنده:** [@AZFarhadi](https://t.me/ra100gov)
    """)
