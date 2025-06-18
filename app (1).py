import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock Time Series Forecast", layout="wide")
st.title("📊 Multi-Model Stock Forecasting")

# --- Sidebar Inputs --- #
ticker = st.text_input("Ticker Symbol", "AAPL")
years = st.slider("Forecast Horizon (years)", 1, 5)

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="10y", interval="1d")
    df = df[['Close']].dropna()
    df.reset_index(inplace=True)
    return df

df = load_data(ticker)
st.subheader("Recent Data")
st.write(df.tail())

fig = go.Figure([go.Scatter(x=df['Date'], y=df['Close'])])
fig.update_layout(title=f"{ticker} Closing Price", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

horizon = years * 365

# --- Prophet Model --- #
st.header("➡️ Prophet Forecast")
df_prophet = df.rename(columns={'Date':'ds','Close':'y'})
model_p = Prophet(daily_seasonality=True)
model_p.fit(df_prophet)
future_p = model_p.make_future_dataframe(periods=horizon)
fcst_p = model_p.predict(future_p)
fig_p = go.Figure([go.Scatter(x=fcst_p['ds'], y=fcst_p['yhat'], name="Prophet Forecast")])
fig_p.update_layout(xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_p, use_container_width=True)

# --- ARIMA Model --- #
st.header("➡️ ARIMA Forecast")
model_a = ARIMA(df['Close'], order=(5,1,0))
res_a = model_a.fit()
fcst_a = res_a.forecast(steps=horizon)
fcst_a = pd.DataFrame({"Date":pd.date_range(df['Date'].iloc[-1], periods=horizon+1, freq='D')[1:], "Forecast":fcst_a})
fig_a = go.Figure([go.Scatter(x=fcst_a['Date'], y=fcst_a['Forecast'], name="ARIMA Forecast")])
st.plotly_chart(fig_a, use_container_width=True)

# --- SARIMA Model --- #
st.header("➡️ SARIMA Forecast")
model_s = SARIMAX(df['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
res_s = model_s.fit(disp=False)
fcst_s = res_s.get_forecast(steps=horizon)
fcst_s_df = fcst_s.predicted_mean.reset_index().rename(columns={'index':'Date', 0:'Forecast'})
fig_s = go.Figure([go.Scatter(x=fcst_s_df['Date'], y=fcst_s_df['Forecast'], name="SARIMA Forecast")])
st.plotly_chart(fig_s, use_container_width=True)

# --- LSTM Model --- #
st.header("➡️ LSTM Forecast")
data_vals = df[['Close']].values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data_vals)

lookback = 60
X, y = [], []
for i in range(lookback, len(scaled)):
    X.append(scaled[i-lookback:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

model_l = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1],1)),
    Dropout(0.2),
    Dense(1)
])
model_l.compile(optimizer='adam', loss='mse')
model_l.fit(X, y, epochs=10, batch_size=32, verbose=0)

inp = scaled[-lookback:]
preds = []
for _ in range(horizon):
    x_in = inp.reshape((1, lookback, 1))
    pred = model_l.predict(x_in)[0][0]
    preds.append(pred)
    inp = np.append(inp[1:], [[pred]], axis=0)

fcst_l = pd.DataFrame({
    "Date": pd.date_range(df['Date'].iloc[-1], periods=horizon+1, freq='D')[1:],
    "Forecast": scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
})
fig_l = go.Figure([go.Scatter(x=fcst_l['Date'], y=fcst_l['Forecast'], name="LSTM Forecast")])
st.plotly_chart(fig_l, use_container_width=True)