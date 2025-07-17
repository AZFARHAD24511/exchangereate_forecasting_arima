import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from datetime import datetime, timedelta
from io import StringIO
from pytrends.request import TrendReq
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

st.set_page_config(page_title="پیش‌بینی نرخ دلار آزاد تهران 📈", layout="wide")
st.markdown("""
<hr>
<div style='font-size: 20px;'>
📈 <strong>© 2025 Dr. Farhadi. All rights reserved.</strong><br>
This application was developed by <strong>Dr. Farhadi</strong>, Ph.D. in <em>Economics (Econometrics)</em> and <em>Data Science</em>.<br>
All trademarks and intellectual property are protected(ARIMA). ™
</div>
""", unsafe_allow_html=True)

st.title("📈 پیش‌بینی نرخ دلار آزاد 📈")

# آدرس فایل ترندز در GitHub
GITHUB_TRENDS_CSV_URL = (
    'https://raw.githubusercontent.com/AZFARHAD24511/exchange_rates_IRAN/main/'
    'predict/google_trends_daily_exrusd_15.csv'
)
# KEYWORDS = ['خرید دلار', 'فروش دلار', 'دلار فردایی']
KEYWORDS = [
    'خرید دلار', 'فروش دلار', 'دلار فردا', 'نرخ ارز', 'سکه طلا',
    'صرافی آنلاین', 'تورم', 'انتخابات', 'اعتراضات', 'تحریم',
    'برجام', 'رئیس‌جمهور', 'انفجار', 'ترور', 'حمله', 'جنگ'
]

@st.cache_data(ttl=3600)
def load_usd_data():
    """بارگذاری داده‌های تاریخی دلار از API TGJU"""
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
            price = float(re.sub(r'<[^>]*>', '', row[0]).replace(',', ''))
            date = datetime.strptime(row[6], "%Y/%m/%d")
            records.append({'date': date, 'price': price})
        except:
            continue
    df = pd.DataFrame(records).set_index('date').sort_index()
    return df

@st.cache_data(ttl=3600)
def load_today_avg():
    """محاسبه‌ی میانگین ۵ نرخ اخیر امروز"""
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
    resp = requests.get(url, params=params, headers={'User-Agent': 'Mozilla/5.0'})
    resp.raise_for_status()
    data = resp.json().get('data', [])
    prices = []
    for row in data:
        raw = row[0]
        clean = int(re.sub(r'<[^>]*>', '', raw).replace(',', ''))
        prices.append(clean)
    last5 = prices[:5]
    return np.nan if not last5 else sum(last5) / len(last5)

@st.cache_data(ttl=3600)
def load_trends_csv():
    r = requests.get(GITHUB_TRENDS_CSV_URL)
    df = pd.read_csv(StringIO(r.text), parse_dates=['date'])
    return df.set_index('date').sort_index()

@st.cache_data(
    ttl=3600,
    hash_funcs={pd.DatetimeIndex: lambda idx: idx.astype(str).tolist()}
)
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

with st.spinner("در حال بارگذاری داده‌ها..."):
    # ۱. تاریخی + میانگین امروز
    hist_df = load_usd_data()
    avg_today = load_today_avg()
    today = pd.to_datetime(datetime.now().date())
    if not np.isnan(avg_today):
        today_df = pd.DataFrame({'price': [avg_today]}, index=[today])
        full_usd = pd.concat([hist_df, today_df])
    else:
        full_usd = hist_df.copy()
    full_usd = full_usd[~full_usd.index.duplicated(keep='last')]
    full_usd = full_usd.sort_index()

    # ایندکس روزانه و درون‌یابی
    full_usd = full_usd.reindex(
        pd.date_range(full_usd.index.min(), full_usd.index.max(), freq='D')
    )
    full_usd['price'] = full_usd['price'].interpolate(method='linear')

    # بارگذاری ترندز
    trends_df = load_trends_csv()
    two_years_ago = datetime.now() - timedelta(days=730)
    usd_df = full_usd[full_usd.index >= two_years_ago]
    trends_df = trends_df[trends_df.index >= two_years_ago]

    # پر کردن ترندز ناقص
    missing = usd_df.index.difference(trends_df.index)
    if not missing.empty:
        new_trends = fetch_missing_trends(missing)
        trends_df = pd.concat([trends_df, new_trends]).sort_index()
        trends_df = trends_df.reindex(usd_df.index).ffill().bfill()

    # ادغام نهایی
    df = pd.merge(usd_df, trends_df, left_index=True, right_index=True, how='inner').ffill().bfill()
    price_series = df['price']

# مدل ARIMA
model = ARIMA(price_series, order=(1, 0, 1))
model_fit = model.fit()

# پیش‌بینی ۲ روز آینده
forecast_vals = model_fit.forecast(steps=2)
forecast_dates = [price_series.index[-1] + timedelta(days=i) for i in range(1, 3)]

# ارزیابی مدل
preds_in = model_fit.predict(start=0, end=len(price_series)-1)
mae = mean_absolute_error(price_series, preds_in)
mape = mean_absolute_percentage_error(price_series, preds_in) * 100
try:
    pvals = model_fit.pvalues
    pval_str = ', '.join([f'{n}: {v:.4f}' for n, v in pvals.items()])
except:
    pval_str = "ناموجود"

st.info(f"MAE: {mae:,.2f}    MAPE: {mape:.2f}%    P-values: {pval_str}")
for d, v in zip(forecast_dates, forecast_vals):
    st.success(f"🔮 نرخ دلار برای {d.date()}: {v:,.0f} ریال")

st.subheader("📊 Historical Data & 2-Day Forecast")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(price_series.index, price_series.values, label='Historical')
ax.axvline(price_series.index[-1], linestyle='--')
for i, (d, v) in enumerate(zip(forecast_dates, forecast_vals), 1):
    ax.scatter(d, v)
    ax.annotate(f'Day+{i}: {v:,.0f}', xy=(d, v), xytext=(0, 10),
                textcoords='offset points', ha='center',
                arrowprops=dict(arrowstyle='->'))
ax.set_title('USD Free Market Rate Forecast')
ax.grid(True)
st.pyplot(fig)
