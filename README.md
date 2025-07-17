# 📈 Forecasting the USD Exchange Rate in Iran's Free Market Using Online Trends and Time Series Modeling

## Abstract

This application provides a real-time, data-driven forecast of the **free-market USD/IRR exchange rate** in Iran. We combine official market data from TGJU with public interest signals extracted from **Google Trends** to enhance short-term forecasting accuracy.

The proposed hybrid architecture leverages:

* A classical **ARIMA(p,d,q)** model,
* Augmented with **high-frequency exogenous variables** derived from trend data for 15 sensitive economic and political keywords in Persian.

The app is fully implemented in **Streamlit** and deploys all components online, including:

* Real-time data collection,
* Preprocessing (resampling, interpolation, and merging),
* Forecasting with ARIMA,
* Dynamic visualization.

---

## 📐 Model Design

Let $y_t$ denote the USD price in IRR on day $t$. The forecasting model follows a standard ARIMA(1,0,1) structure:

$$
y_t = \phi_1 y_{t-1} + \theta_1 \varepsilon_{t-1} + \varepsilon_t
$$

where:

* $\phi_1$ is the AR(1) coefficient,
* $\theta_1$ is the MA(1) coefficient,
* $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$ are white noise innovations.

We define an auxiliary matrix $X_t \in \mathbb{R}^{n \times T}$ representing normalized Google Trends scores for $n = 15$ Persian keywords across time $T$. These variables are used **not directly in the ARIMA model**, but to inform the data imputation, smoothing, and temporal resolution decisions.

---

## 📊 Data Sources

1. **USD Free Market Price** – Fetched via [TGJU API](https://www.tgju.org).
2. **Google Trends (PyTrends)** – Using the following Persian keywords:

```
['خرید دلار', 'فروش دلار', 'دلار فردا', 'نرخ ارز', 'سکه طلا',
 'صرافی آنلاین', 'تورم', 'انتخابات', 'اعتراضات', 'تحریم',
 'برجام', 'رئیس‌جمهور', 'انفجار', 'ترور', 'جنگ']
```

Daily scores are normalized:

$$
x_{i,t}^{\text{scaled}} = \frac{x_{i,t}}{\max_t x_{i,t}} \times 100
$$

Missing values are filled using backward and forward fill (bfill/ffill) strategies after interpolation.

---

## 🔮 Forecast Output

The model forecasts the next **two daily exchange rates** and displays:

* Historical chart with future points
* Forecast point estimates
* Evaluation metrics:

  * Mean Absolute Error (MAE)
  * Mean Absolute Percentage Error (MAPE)
  * ARIMA coefficient **p-values**

$$\text{MAE} = \frac{1}{T} \sum_{t=1}^T(y_t - \hat{y}_t)$$

$$\text{MAPE} = \frac{100}{T} \sum_{t=1}^T \left| \frac{y_t - \hat{y}_t}{y_t} \right|$$

---

## 🧠 Future Improvements

* Integrating exogenous variables (ARIMAX/XGBoost)
* Incorporating volatility modeling (GARCH)
* Topic modeling from Persian news headlines

---

## 📜 Citation and License

You are free to use or adapt the content of this repository **with proper attribution**.
Please cite this work as:

> Farhadi, A. (2025). *Forecasting USD/IRR Exchange Rate using Google Trends and ARIMA*. GitHub. [https://github.com/AZFARHAD24511/exchange\_rates\_IRAN](https://github.com/AZFARHAD24511/exchange_rates_IRAN)

---

If you'd like, I can also generate a PDF or add this to a `README.md` in your GitHub repo directly. Let me know!
