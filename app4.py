import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import numpy as np
import logging

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Streamlit UI ---
st.set_page_config(page_title="Time Series Forecasting App", layout="wide")
st.title("🕒 Time Series Forecasting App")

uploaded_file = st.file_uploader("📂 Upload a CSV file with 'Date' and 'Close' columns", type="csv")

if uploaded_file is not None:
    logger.info("File uploaded successfully")
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.replace('\xa0', '', regex=True).str.strip()

    if 'Date' not in data.columns or 'Close' not in data.columns:
        st.error("❌ Your file must contain 'Date' and 'Close' columns.")
    else:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        st.subheader("🔍 Raw Data Preview")
        st.dataframe(data.head())

        # --- EDA ---
        st.subheader("📊 Exploratory Data Analysis")
        st.write("### Descriptive Statistics")
        st.write(data.describe())

        st.write("### Missing Values")
        st.write(data.isnull().sum())

        st.write("### Time Series Plot")
        fig1, ax1 = plt.subplots()
        ax1.plot(data['Close'], color='blue')
        ax1.set_title('Close Price Over Time')
        st.pyplot(fig1)

        st.write("### Histogram of Close Prices")
        fig2, ax2 = plt.subplots()
        ax2.hist(data['Close'].dropna(), bins=30, color='orange', edgecolor='black')
        st.pyplot(fig2)

        st.write("### Outlier Detection (IQR Method)")
        Q1 = data['Close'].quantile(0.25)
        Q3 = data['Close'].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = data[(data['Close'] < lower) | (data['Close'] > upper)]
        st.write(f"Detected {outliers.shape[0]} outliers.")
        fig3, ax3 = plt.subplots()
        ax3.plot(data.index, data['Close'], label='Close')
        ax3.scatter(outliers.index, outliers['Close'], color='red', label='Outliers')
        ax3.legend()
        st.pyplot(fig3)

        st.write("### 📉 ADF Test (Stationarity Check)")
        result = adfuller(data['Close'].dropna())
        st.write(f"ADF Statistic: {result[0]:.4f}")
        st.write(f"p-value: {result[1]:.4f}")
        if result[1] <= 0.05:
            st.success("Series is likely stationary. Differencing may not be needed.")
        else:
            st.warning("Series is likely non-stationary. Consider differencing.")

        # --- Monthly Resampling ---
        monthly = data['Close'].resample('M').last().to_frame()

        # --- Decomposition ---
        st.subheader("📐 Time Series Decomposition")
        model_type = st.selectbox("Select decomposition type", ["Additive", "Multiplicative"])
        max_period = max(2, len(monthly) // 2)
        period = st.slider("Seasonal Period", min_value=2, max_value=max_period, value=min(12, max_period))

        try:
            decomposition = seasonal_decompose(monthly['Close'].dropna(), model=model_type.lower(), period=period)
            fig, axs = plt.subplots(4, 1, figsize=(12, 10))
            axs[0].plot(decomposition.observed); axs[0].set_title("Observed")
            axs[1].plot(decomposition.trend); axs[1].set_title("Trend")
            axs[2].plot(decomposition.seasonal); axs[2].set_title("Seasonal")
            axs[3].plot(decomposition.resid); axs[3].set_title("Residual")
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("**📌 Business Insight**: Use seasonal patterns to predict high/low demand months, trends for long-term growth, and residuals to assess random events or noise.")
        except Exception as e:
            logger.error(f"Decomposition error: {e}")
            st.error(f"Decomposition Error: {e}")

        # --- Forecasting ---
        st.subheader("📈 Forecasting Models")
        model_choice = st.selectbox("Choose a forecasting model", ["ARIMA", "ETS", "Prophet"])

        split = int(len(monthly) * 0.8)
        train, test = monthly[:split], monthly[split:]
        forecast_index = test.index

        def evaluate(true, pred):
            mse = mean_squared_error(true, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true, pred)
            mape = np.mean(np.abs((true - pred) / true)) * 100
            return rmse, mae, mape, mse

        forecast_all = {}
        model_metrics = {}

        # --- ARIMA Model ---
        try:
            arima_model = ARIMA(train, order=(5, 1, 5)).fit()
            arima_forecast = arima_model.predict(start=len(train), end=len(train) + len(test) - 1)
            arima_forecast.index = forecast_index
            forecast_all['ARIMA'] = arima_forecast
            model_metrics['ARIMA'] = evaluate(test['Close'], arima_forecast)
        except:
            model_metrics['ARIMA'] = [None]*4

        # --- ETS Model ---
        try:
            ets_model = SimpleExpSmoothing(train['Close']).fit(smoothing_level=0.2)
            ets_forecast = ets_model.forecast(len(test))
            ets_forecast.index = forecast_index
            forecast_all['ETS'] = ets_forecast
            model_metrics['ETS'] = evaluate(test['Close'], ets_forecast)
        except:
            model_metrics['ETS'] = [None]*4

        # --- Prophet Model ---
        try:
            prophet_df = train.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
            prophet_model = Prophet(yearly_seasonality=True)
            prophet_model.fit(prophet_df)
            future_df = prophet_model.make_future_dataframe(periods=len(test), freq='M')
            prophet_forecast_df = prophet_model.predict(future_df)
            prophet_forecast = prophet_forecast_df.set_index('ds').loc[forecast_index]['yhat']
            forecast_all['Prophet'] = prophet_forecast
            model_metrics['Prophet'] = evaluate(test['Close'], prophet_forecast)
        except:
            model_metrics['Prophet'] = [None]*4

        # --- Display Forecast Comparisons ---
        st.write("### 📈 Forecast Comparison Plot")
        comparison_df = pd.DataFrame({'Actual': test['Close']})
        for name, forecast in forecast_all.items():
            comparison_df[name] = forecast
        st.line_chart(comparison_df)

        # --- Metrics Table ---
        st.write("### 📋 Evaluation Metrics Table")
        metrics_df = pd.DataFrame(model_metrics, index=["RMSE", "MAE", "MAPE", "MSE"]).T
        st.dataframe(metrics_df.style.background_gradient(cmap='Blues'))

        # --- Best Model Selection ---
        best_model = min(model_metrics, key=lambda model: model_metrics[model][0] if model_metrics[model][0] is not None else float('inf'))
        st.subheader(f"🏆 Best Model: {best_model}")
        st.write(f"**Evaluation Metrics for Best Model ({best_model}):**")
        st.write(f"RMSE: {model_metrics[best_model][0]:.2f}")
        st.write(f"MAE: {model_metrics[best_model][1]:.2f}")
        st.write(f"MAPE: {model_metrics[best_model][2]:.2f}%")
        st.write(f"MSE: {model_metrics[best_model][3]:.2f}")

        # --- Insight Summary ---
        st.subheader("📘 Final Insight & Recommendations")
        st.markdown("""
        - **Short-term forecasting**: ETS is often suitable due to its simplicity.
        - **Long-term forecasting**: Prophet or ARIMA can capture trend + seasonality better.
        - **Finance industry**: ARIMA is traditionally favored due to strong statistical basis.
        - **Retail or seasonal industries**: Prophet is ideal due to seasonal trend handling.
        """)
