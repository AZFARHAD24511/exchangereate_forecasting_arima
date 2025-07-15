import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from io import StringIO
from pytrends.request import TrendReq
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# پیکربندی صفحه
st.set_page_config(page_title=" پیش‌بینی نرخ دلار آزاد تهران 📈", layout="wide")
st.markdown("""
---
📈 © 2025 Dr. Farhadi. All rights reserved.  
This application was developed by **Dr. Farhadi**, Ph.D. in *Economics (Econometrics)* and *Data Science*.  
All trademarks and intellectual property are protected. ™
""")
st.title("📈 پیش‌بینی نرخ دلار آزاد (با XGBoost) 📈")

# آدرس فایل ترندز در GitHub
GITHUB_TRENDS_CSV_URL = (
    'https://raw.githubusercontent.com/AZFARHAD24511/exchange_rates_IRAN/main/'
    'predict/google_trends_daily.csv'
)
KEYWORDS = ['خرید دلار', 'فروش دلار', 'دلار فردایی']

# بارگذاری داده‌های دلار آزاد از API
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
                row[0].replace(',', '')
                      .replace('<span class="high" dir="ltr">', '')
                      .replace('</span>', '')
            )
            date = datetime.strptime(row[6], "%Y/%m/%d")
            records.append({'date': date, 'price': price})
        except:
            continue
    df = pd.DataFrame(records).set_index('date').sort_index()
    return df

# بارگذاری داده‌های Google Trends از GitHub
@st.cache_data(ttl=3600)
def load_trends_csv():
    r = requests.get(GITHUB_TRENDS_CSV_URL)
    df = pd.read_csv(StringIO(r.text), parse_dates=['date'])
    return df.set_index('date').sort_index()

# گرفتن داده‌های ناقص Google Trends
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

# ایجاد ویژگی‌های زمانی برای مدل XGBoost
def create_features(df, target, lags=7):
    df = df.copy()
    
    # ایجاد تاخیرها
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = target.shift(lag)
    
    # ویژگی‌های زمانی
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # میانگین متحرک
    df['rolling_7d_mean'] = target.rolling(window=7).mean()
    df['rolling_30d_mean'] = target.rolling(window=30).mean()
    
    # نوسان
    df['rolling_7d_std'] = target.rolling(window=7).std()
    
    # حذف مقادیر NaN
    df = df.dropna()
    
    return df

# بارگذاری و ترکیب داده‌ها
with st.spinner("در حال بارگذاری داده‌ها و آموزش مدل XGBoost..."):
    usd_df = load_usd_data()
    trends_df = load_trends_csv()
    
    # استفاده از دو سال اخیر
    two_years_ago = datetime.now() - timedelta(days=730)
    usd_df = usd_df[usd_df.index >= two_years_ago]
    trends_df = trends_df[trends_df.index >= two_years_ago]
    
    # پر کردن تاریخ‌های ناقص
    missing = usd_df.index.difference(trends_df.index)
    if not missing.empty:
        missing_tuple = tuple(date.strftime('%Y-%m-%d') for date in missing)
        new_trends = fetch_missing_trends(missing_tuple)
        trends_df = pd.concat([trends_df, new_trends]).sort_index()
        trends_df = trends_df.reindex(usd_df.index).ffill().bfill()
    
    # ادغام داده‌ها
    df = pd.merge(usd_df, trends_df, left_index=True, right_index=True, how='inner').ffill().bfill()
    
    # ایجاد ویژگی‌ها
    full_df = create_features(df, df['price'], lags=7)
    
    # جداسازی ویژگی‌ها و هدف
    X = full_df.drop(columns=['price'])
    y = full_df['price']
    
    # تقسیم داده‌ها
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # نرمال‌سازی داده‌ها
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # آموزش مدل XGBoost
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=False
    )
    
    # پیش‌بینی روی داده‌های آزمون
    test_preds = model.predict(X_test_scaled)
    
    # محاسبه خطاها
    mae = mean_absolute_error(y_test, test_preds)
    mape = mean_absolute_percentage_error(y_test, test_preds) * 100
    
    # آماده‌سازی برای پیش‌بینی آینده
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, 3)]
    
    # ایجاد DataFrame برای پیش‌بینی
    forecast_df = pd.DataFrame(index=forecast_dates)
    
    # کپی آخرین داده‌های موجود
    current_data = full_df.iloc[[-1]].copy()
    
    # پیش‌بینی گام به گام
    forecast_vals = []
    for date in forecast_dates:
        # به‌روزرسانی ویژگی‌های زمانی
        current_data.index = [date]
        current_data['day_of_week'] = date.dayofweek
        current_data['day_of_month'] = date.day
        current_data['month'] = date.month
        current_data['year'] = date.year
        
        # پیش‌بینی قیمت
        current_scaled = scaler.transform(current_data)
        pred_price = model.predict(current_scaled)[0]
        forecast_vals.append(pred_price)
        
        # به‌روزرسانی تاخیرها برای گام بعدی
        for lag in range(7, 1, -1):
            current_data[f'lag_{lag}'] = current_data[f'lag_{lag-1}']
        current_data['lag_1'] = pred_price

# نمایش نتایج
st.info(f"دقت مدل XGBoost: MAE: {mae:,.2f}    MAPE: {mape:.2f}%")
st.success(f"🔮 نرخ دلار برای {forecast_dates[0].date()}: {forecast_vals[0]:,.0f} ریال")
st.success(f"🔮 نرخ دلار برای {forecast_dates[1].date()}: {forecast_vals[1]:,.0f} ریال")

# نمایش نمودار
st.subheader("📊 داده‌های تاریخی و پیش‌بینی ۲ روز آینده")

# پیش‌بینی روی کل داده‌ها برای نمایش
full_preds = model.predict(scaler.transform(X))
df['predicted'] = np.nan
df.loc[X.index, 'predicted'] = full_preds

# ایجاد نمودار
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['price'], label='داده‌های تاریخی', color='blue')
ax.plot(df.index, df['predicted'], label='پیش‌بینی مدل (داده آموزشی)', color='green', alpha=0.7)

# افزودن پیش‌بینی آینده
forecast_df = pd.DataFrame({
    'date': forecast_dates,
    'price': forecast_vals
}).set_index('date')

ax.plot(forecast_df.index, forecast_df['price'], 'ro-', label='پیش‌بینی آینده')

# تنظیمات نمودار
ax.axvline(last_date, linestyle='--', color='gray')
ax.set_title('پیش‌بینی نرخ دلار آزاد با XGBoost')
ax.set_xlabel('تاریخ')
ax.set_ylabel('نرخ دلار (ریال)')
ax.grid(True)
ax.legend()

# نمایش نمودار در Streamlit
st.pyplot(fig)

# نمایش اهمیت ویژگی‌ها
st.subheader("📊 اهمیت ویژگی‌ها در مدل XGBoost")

# ایجاد نمودار اهمیت ویژگی‌ها
fig2, ax2 = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model, ax=ax2, max_num_features=15)
ax2.set_title('اهمیت ویژگی‌ها')
st.pyplot(fig2)
