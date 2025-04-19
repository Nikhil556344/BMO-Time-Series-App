
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

st.title(" Time Series Forecasting App")
st.markdown("Upload your CSV, choose decomposition type and forecasting model to view results.")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.replace('\xa0', '', regex=True).str.strip()
    if 'Date' not in data.columns or 'Close' not in data.columns:
        st.error("Your file must contain 'Date' and 'Close' columns.")
    else:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        st.write("### Raw Data Preview")
        st.dataframe(data.head())

        # Resample monthly
        monthly = data['Close'].resample('M').last().to_frame()

        # Decomposition Selection
        model_type = st.selectbox("Select decomposition type", ["Additive", "Multiplicative"])
        period = st.slider("Seasonal Period", min_value=2, max_value=60, value=30)

        try:
            decomposition = seasonal_decompose(monthly['Close'].dropna(), model=model_type.lower(), period=period)
            st.write(f"### {model_type} Decomposition")
            fig, axs = plt.subplots(4, 1, figsize=(12, 10))
            axs[0].plot(decomposition.observed); axs[0].set_title("Original")
            axs[1].plot(decomposition.trend); axs[1].set_title("Trend")
            axs[2].plot(decomposition.seasonal); axs[2].set_title("Seasonal")
            axs[3].plot(decomposition.resid); axs[3].set_title("Residual")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Decomposition Error: {e}")

        # Forecasting Model
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

        if model_choice == "ARIMA":
            try:
                model_arima = ARIMA(train, order=(5, 1, 5)).fit()
                forecast = model_arima.predict(start=len(train), end=len(train) + len(test) - 1)
                forecast.index = forecast_index
                st.line_chart(pd.DataFrame({'Actual': test['Close'], 'Forecast': forecast}))
                rmse, mae, mape, mse = evaluate(test['Close'], forecast)
            except Exception as e:
                st.error(f"ARIMA Error: {e}")

        elif model_choice == "ETS":
            try:
                model_ets = SimpleExpSmoothing(monthly['Close']).fit(smoothing_level=0.2)
                forecast = model_ets.predict(start=test.index[0], end=test.index[-1])
                st.line_chart(pd.DataFrame({'Actual': test['Close'], 'Forecast': forecast}))
                rmse, mae, mape, mse = evaluate(test['Close'], forecast)
            except Exception as e:
                st.error(f"ETS Error: {e}")

        elif model_choice == "Prophet":
            try:
                prophet_data = train.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
                future = test.reset_index().rename(columns={'Date': 'ds'})
                m = Prophet(yearly_seasonality=True)
                m.fit(prophet_data)
                future_df = m.make_future_dataframe(periods=len(test), freq='M')
                forecast_df = m.predict(future_df)
                forecast = forecast_df.set_index('ds').loc[forecast_index]['yhat']
                st.line_chart(pd.DataFrame({'Actual': test['Close'], 'Forecast': forecast}))
                rmse, mae, mape, mse = evaluate(test['Close'], forecast)
            except Exception as e:
                st.error(f"Prophet Error: {e}")

        # Show evaluation
        st.write("###  Forecast Evaluation Metrics")
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("MAE", f"{mae:.2f}")
        st.metric("MAPE", f"{mape:.2f}%")
        st.metric("MSE", f"{mse:.2f}")
