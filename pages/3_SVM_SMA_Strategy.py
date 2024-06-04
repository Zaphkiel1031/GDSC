import streamlit as st
import backtrader as bt
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
import random

matplotlib.use('Agg')

window_size = 10

# 函數：獲取股票數據
def get_stock_data(code, start_date, end_date, short_period, long_period):
    try:
        # 將開始日期提前一年
        adjusted_start_date = pd.to_datetime(start_date) - pd.DateOffset(years=1)
        
        df = yf.download(code, start=adjusted_start_date, end=end_date)
        if df.empty:
            st.error(f"無法下載股票代碼 {code} 的數據，請檢查股票代碼和日期範圍。")
            return None
        df = df.sort_index(ascending=True)
        df['SMA_10'] = df['Close'].rolling(window=short_period).mean()
        df['SMA_20'] = df['Close'].rolling(window=long_period).mean()
        df = df.dropna()
        
        # 檢查數據長度是否足夠
        if len(df) < long_period + window_size:
            st.error(f"數據不足以計算長期均線，請選擇更長的時間範圍。")
            return None
        
        # 只返回實際開始日期之後的數據
        df = df[df.index >= pd.to_datetime(start_date)]
        return df
    except Exception as e:
        st.error(f"無法下載股票數據: {e}")
        return None

def create_dataset(stock_data, window_size):
    X = []
    y = []
    scaler = MinMaxScaler()
    stock_data_normalized = scaler.fit_transform(stock_data[['Close', 'SMA_10', 'SMA_20']].values)

    for i in range(len(stock_data) - window_size - 2):
        X.append(stock_data_normalized[i:i + window_size])
        if stock_data.iloc[i + window_size + 2]['Close'] > stock_data.iloc[i + window_size - 1]['Close']:
            y.append(1)
        else:
            y.append(0)

    X, y = np.array(X), np.array(y)
    return X, y, scaler

# 定義SVM策略
class SVMStrategy(bt.Strategy):
    params = (
        ("window_size", 10),
        ("scaler", None),
        ("model", None),
        ("short_period", 10),
        ("long_period", 20),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.sma10 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.short_period)
        self.sma20 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.long_period)
        self.counter = 1
        self.buyprice = None
        self.buycomm = None

    def log(self, txt, dt=None):
        pass

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            self.bar_executed = len(self)

        self.order = None

    def notify_trade(self, trade):
        pass

    def next(self):
        if self.counter < self.params.window_size:
            self.counter += 1
            return

        previous_features = [[self.data_close[-i], self.sma10[-i], self.sma20[-i]] for i in range(0, self.params.window_size)]
        X = np.array(previous_features).reshape(self.params.window_size, -1)

        X = self.params.scaler.transform(X)
        X = X.reshape(1, -1)  # 將 X 重新調整為 2D 數組

        prediction = self.params.model.predict(X)
        predicted_trend = prediction[0]

        if predicted_trend == 1 and not self.position:
            self.order = self.buy()
        elif predicted_trend == 0 and self.position:
            self.order = self.sell()
        elif self.position:
            if self.data_close[0] < self.buyprice * 0.9:
                self.order = self.sell()
            elif self.data_close[0] > self.buyprice * 1.5:
                self.order = self.sell()

# 函數：訓練SVM模型
def train_svm():
    global svm_model_ready, trained_svm_model, scaler
    with st.spinner("開始訓練SVM..."):
        stock_data = get_stock_data(symbol, start_date, end_date, short_period, long_period)
        if stock_data is None:
            return

        window_size = 10
        if len(stock_data) < long_period + window_size:
            st.error("股票數據不足以創建訓練數據集，請選擇更長的時間範圍。")
            return

        X, y, scaler = create_dataset(stock_data[['Close', 'SMA_10', 'SMA_20']], window_size)

        X = X.reshape(X.shape[0], -1)

        svm_model = SVC(kernel='rbf', C=1, gamma='scale')
        svm_model.fit(X, y)

        trained_svm_model = svm_model
        svm_model_ready = True

def display_results(cash, value, initial_cash, n_years):
    # 計算年回報率
    annual_return = ((value - initial_cash) / initial_cash + 1) ** (1 / n_years) - 1
    annual_return *= 100  # 轉換為百分比形式
    total_return = (value - initial_cash) / initial_cash * 100  # 轉換為百分比形式

    # 計算預算變化
    budget_delta = value - initial_cash

    # 設置獨立變數來顯示 delta 值
    budget_delta_display = f"${budget_delta:.2f}"
    annual_return_display = f"{annual_return:.2f}%"
    total_return_display = f"{total_return:.2f}%"

    # 創建多列佈局
    col1, col2, col3 = st.columns(3)

    # 在第一列中顯示預算
    custom_metric(col1, "預算", f"${initial_cash:.2f}", budget_delta_display)

    # 在第二列中顯示最終價值
    custom_metric(col2, "最終價值", f"${value:.2f}", "")

    # 在第三列中顯示年回報率
    custom_metric(col3, "年回報率", annual_return_display, total_return_display)

    return annual_return

# 自定義顏色顯示函數
def custom_metric(column, label, value, delta):
    # 去掉美元符號並轉換為浮點數
    delta_value = float(delta.replace('$', '').replace('%', '')) if delta else 0
    delta_color = "red" if delta_value > 0 else "green"
    delta_sign = "+" if delta_value > 0 else ""
    delta_display = f"{delta_sign}{delta}" if delta else ""
    column.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 10px;">
        <span style="font-size: 1rem;">{label}</span>
        <span style="font-size: 2rem; font-weight: bold;">{value}</span>
        <span style="font-size: 1rem; color: {delta_color};">{delta_display}</span>
    </div>
    """, unsafe_allow_html=True)

def run_backtrader():
    global scaler, svm_model_ready, trained_svm_model
    with st.spinner("運行Backtrader..."):
        while not svm_model_ready:
            time.sleep(1)

        st.write("SVM模型加載成功。")

        stock_data = get_stock_data(symbol, start_date, end_date, short_period, long_period)
        if stock_data is None:
            return

        window_size = 10
        if len(stock_data) < long_period + window_size:
            st.error("股票數據不足以創建訓練數據集，請選擇更長的時間範圍。")
            return

        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(initial_cash)
        cerebro.broker.setcommission(commission=commission/100)

        cerebro.addstrategy(SVMStrategy, scaler=scaler, model=trained_svm_model, short_period=short_period, long_period=long_period)

        data = bt.feeds.PandasData(dataname=stock_data)
        cerebro.adddata(data)

        results = cerebro.run()

        # 獲取當前現金餘額和總價值
        cash = cerebro.broker.get_cash()
        value = cerebro.broker.get_value()

        # 計算回測的年數
        n_years_backtest = (end_date - start_date).days / 365.25

        # 顯示結果
        annual_return = display_results(cash, value, initial_cash, n_years_backtest)

        # 繪製回測結果
        fig = cerebro.plot(style='candlestick')[0][0]
        st.pyplot(fig)

# 定義圖片URL列表
image_urls = [
    "https://raw.githubusercontent.com/j7808833/test_02/main/pic/Cyberpunk_bar_04.gif",
    "https://raw.githubusercontent.com/j7808833/test_02/main/pic/Cyberpunk_bar_06.gif",
    "https://raw.githubusercontent.com/j7808833/test_02/main/pic/Cyberpunk_bar_07.gif",
    "https://raw.githubusercontent.com/j7808833/test_02/main/pic/Cyberpunk_bar_08.gif",
    "https://raw.githubusercontent.com/j7808833/test_02/main/pic/Cyberpunk_bar_09.gif",
    "https://raw.githubusercontent.com/j7808833/test_02/main/pic/Cyberpunk_bar_10.gif"
]

# 隨機選擇一張圖片
selected_image_url = random.choice(image_urls)

# 顯示圖片
st.image(selected_image_url)

# Streamlit 應用
st.title("SVM股票交易策略")

st.markdown("""

### 策略概述
這個策略使用了一個簡單的長短期移動平均線（SMA）和一個基於支持向量機（SVM）模型來進行股票交易。主要步驟如下：

1. **獲取股票數據**：從Yahoo Finance下載指定股票的歷史數據。
2. **計算移動平均線**：計算短期和長期的移動平均線（SMA）。
3. **創建訓練數據集**：將股票數據轉換為SVM模型的訓練數據集。
4. **訓練SVM模型**：使用訓練數據集來訓練SVM模型。
5. **回測策略**：使用Backtrader回測引擎來運行交易策略，並評估其表現。

### 詳細步驟
1. **獲取股票數據**
   - 使用yfinance庫從Yahoo Finance下載股票數據。
   - 計算短期和長期的移動平均線，並將其添加到數據框中。

2. **創建訓練數據集**
   - 將股票數據轉換為SVM模型的訓練數據集。這裡使用了MinMaxScaler進行數據標準化。
   - 根據窗口大小（window_size）創建特徵和標籤。特徵是窗口內的股票價格和移動平均線，標籤是窗口結束後的價格變動方向（上漲或下跌）。

3. **訓練SVM模型**
   - 定義SVM模型的結構，包括核函數、懲罰參數和其他超參數。
   - 使用訓練數據集進行模型訓練，優化模型參數。

4. **回測策略**
   - 使用Backtrader回測引擎運行交易策略。
   - 策略根據SVM模型的預測結果進行交易決策。如果預測價格會上漲且目前沒有持倉，則買入股票；如果預測價格會下跌且目前有持倉，則賣出股票。
   - 策略還包括止損和止盈邏輯：如果價格下跌超過10%或上漲超過50%，則賣出股票。

5. **結果展示**
   - 回測完成後，顯示最終的投資組合價值、盈虧（P/L）和投資報酬率（ROI）。
   - 使用Matplotlib繪製回測結果的K線圖，並嵌入到Streamlit應用中展示。

### 使用方法
1. 在Streamlit應用中輸入股票符號、開始日期和結束日期。
2. 調整短期和長期移動平均線的參數、交易手續費、每次交易金額和初始現金。
3. 點擊“開始回測”按鈕，系統會自動訓練SVM模型並運行回測策略，最終展示回測結果。

這個策略結合了技術分析（移動平均線）和機器學習（SVM模型）的優勢，旨在提高交易決策的準確性和收益率。
""")

symbol = st.text_input("股票代碼，台股請記得在最後加上.TW", "AAPL")
start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"))
end_date = st.date_input("結束日期", pd.to_datetime("today"))

short_period = st.slider("短期均線", 1, 60, 5)
long_period = st.slider("長期均線", 30, 200, 60)
commission = st.slider('交易手續費 (%)', min_value=0.0, max_value=1.0, step=0.0005, format="%.4f", value=0.001)
trade_amount = st.slider("每次交易金額", min_value=0, max_value=50000, step=1000, value=1000)
initial_cash = st.slider("預算", min_value=0, max_value=10000000, step=10000, value=10000)

# 檢查短期均線是否大於長期均線
if short_period >= long_period:
    st.error("短期均線的窗口期不能大於或等於長期均線的窗口期，請調整參數。")
# 檢查時間跨度是否足夠長
elif (end_date - start_date).days < long_period + window_size:
    st.error(f"時間跨度過短，請選擇至少 {long_period + window_size} 天的時間跨度。")
else:
    if st.button("開始回測"):
        svm_model_ready = False
        trained_svm_model = None
        scaler = None

        # 獲取股票數據
        stock_data = get_stock_data(symbol, start_date, end_date, short_period, long_period)

        # 檢查股票數據是否成功下載
        if stock_data is None:
            st.error("股票代碼錯誤或無法下載數據，請檢查輸入的股票代碼。")
        else:
            try:
                # 檢查股票數據是否足夠
                if len(stock_data) < long_period + window_size:
                    st.error("股票數據不足以創建訓練數據集，請選擇更長的時間範圍。")
                else:
                    # 執行訓練和回測
                    train_svm()
                    run_backtrader()
            except ValueError as e:
                st.error(f"數據集創建失敗: {e}")