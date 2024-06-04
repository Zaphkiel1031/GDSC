import streamlit as st
import backtrader as bt
import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
import random

matplotlib.use('Agg')

# 定義全局變量
window_size = 10

def get_stock_data(code, start_date, end_date):
    try:
        # 將開始日期提前一年
        adjusted_start_date = pd.to_datetime(start_date) - pd.DateOffset(years=1)
        
        df = yf.download(code, start=adjusted_start_date, end=end_date)
        if df.empty:
            raise ValueError("下載的數據為空")
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

# 函數：將股票數據轉換為模型訓練數據集
def create_dataset(stock_data, window_size):
    if len(stock_data) < window_size + 2:
        raise ValueError("股票數據不足以創建訓練數據集")
    
    X = []
    y = []
    scaler = MinMaxScaler()
    stock_data_normalized = scaler.fit_transform(stock_data.values)

    for i in range(len(stock_data) - window_size - 2):
        X.append(stock_data_normalized[i:i + window_size])
        if stock_data.iloc[i + window_size + 2]['Close'] > stock_data.iloc[i + window_size - 1]['Close']:
            y.append(1)
        else:
            y.append(0)

    X, y = np.array(X), np.array(y)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    return X, y, scaler

# 函數：創建DataLoader
def create_dataloader(X, y, batch_size):
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# LSTM 模型定義
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.bn(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# 定義策略
class LSTMStrategy(bt.Strategy):
    params = (
        ("window_size", 10),
        ("scaler", None),
        ("model", None),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.sma10 = bt.indicators.SimpleMovingAverage(self.datas[0], period=short_period)
        self.sma20 = bt.indicators.SimpleMovingAverage(self.datas[0], period=long_period)
        self.counter = 1
        self.buyprice = None
        self.buycomm = None

    def log(self, txt, dt=None):
        pass  # 不再記錄詳細日志

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
        pass  # 不再記錄詳細日志

    def next(self):
        if self.counter < self.params.window_size:
            self.counter += 1
            return

        previous_features = [[self.data_close[-i], self.sma10[-i], self.sma20[-i]] for i in range(0, self.params.window_size)]
        X = torch.tensor(previous_features).view(1, self.params.window_size, -1).float()
        X = self.params.scaler.transform(X.numpy().reshape(-1, 3)).reshape(1, self.params.window_size, -1)

        # 將模型設置為評估模式
        self.params.model.eval()
        with torch.no_grad():
            prediction = self.params.model(torch.tensor(X).float())

        max_vals, max_idxs = torch.max(prediction, dim=1)
        predicted_trend = max_idxs.item()

        if predicted_trend == 1 and not self.position:
            self.order = self.buy()  # 買入股票
        elif predicted_trend == 0 and self.position:
            self.order = self.sell()
        elif self.position:
            # 這裡可以添加止損或止盈邏輯
            if self.data_close[0] < self.buyprice * 0.9:  # 假設止損點為買入價格的90%
                self.order = self.sell()
            elif self.data_close[0] > self.buyprice * 1.5:  # 假設止盈點為買入價格的150%
                self.order = self.sell()

# 定義訓練LSTM模型的函數
def train_lstm():
    global input_size, hidden_size, num_layers, num_classes, scaler, lstm_model_ready, trained_model
    with st.spinner("開始訓練LSTM..."):
        start_time = time.time()
        max_training_time = 300  # 最大訓練時間為300秒（5分鐘）

        # 獲取股票數據
        stock_data = get_stock_data(symbol, start_date, end_date)

        # 檢查股票數據是否足夠
        if len(stock_data) < window_size + 2:
            st.error("股票數據不足以創建訓練數據集，請選擇更長的時間範圍。")
            return

        # 將股票數據轉換為模型訓練數據集
        X, y, scaler = create_dataset(stock_data[['Close', 'SMA_10', 'SMA_20']], window_size)

        # 定義批量大小和DataLoader
        batch_size = 64
        train_loader = create_dataloader(X, y, batch_size)

        # 模型參數定義
        input_size = 3  # 更新為特徵數
        hidden_size = 128
        num_layers = 2
        num_classes = 2

        # LSTM 模型初始化
        model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        num_epochs = 200

        # 訓練模型
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch+1) % 10 == 0:
                st.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 保存訓練好的模型到內存
        trained_model = model
        lstm_model_ready = True

def display_results(cash, value, initial_value, n_years, initial_cash):
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
    global input_size, hidden_size, num_layers, num_classes, scaler, lstm_model_ready, trained_model
    with st.spinner("運行Backtrader..."):
        while not lstm_model_ready:
            time.sleep(10)

        st.write("LSTM模型加載成功。")

        # 獲取股票數據
        stock_data = get_stock_data(symbol, start_date, end_date)

        # 檢查股票數據是否足夠
        if len(stock_data) < window_size + 2:
            st.error("股票數據不足以創建訓練數據集，請選擇更長的時間範圍。")
            return

        # 創建Backtrader引擎
        cerebro = bt.Cerebro()

        # 設置初始資金
        cerebro.broker.set_cash(initial_cash)
        cerebro.broker.setcommission(commission=commission/100)

        # 添加策略並傳遞scaler和model
        cerebro.addstrategy(LSTMStrategy, scaler=scaler, model=trained_model)

        # 將數據添加到引擎中
        data = bt.feeds.PandasData(dataname=stock_data)
        cerebro.adddata(data)

        # 運行策略
        results = cerebro.run()

        # 獲取當前現金餘額和總價值
        cash = cerebro.broker.get_cash()
        value = cerebro.broker.get_value()

        # 計算回測的年數
        n_years_backtest = (end_date - start_date).days / 365.25

        # 顯示結果
        annual_return = display_results(cash, value, initial_cash, n_years_backtest, initial_cash)

        # 繪製回測結果
        fig = cerebro.plot(style='candlestick')[0][0]  # 獲取 Matplotlib 圖形對象
        st.pyplot(fig)  # 將圖形嵌入到 Streamlit 頁面中

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
st.title("LSTM 股票交易策略")

st.markdown("""
            
### 策略概述

這個策略使用了一個簡單的長短期移動平均線（SMA）和一個基於LSTM（長短期記憶）神經網絡的模型來進行股票交易。主要步驟如下：

1. **獲取股票數據**：從Yahoo Finance下載指定股票的歷史數據。
2. **計算移動平均線**：計算短期和長期的移動平均線（SMA）。
3. **創建訓練數據集**：將股票數據轉換為LSTM模型的訓練數據集。
4. **訓練LSTM模型**：使用訓練數據集來訓練LSTM模型。
5. **回測策略**：使用Backtrader回測引擎來運行交易策略，並評估其表現。

## 詳細步驟

### 1. 獲取股票數據
- 使用yfinance庫從Yahoo Finance下載股票數據。
- 計算短期和長期的移動平均線，並將其添加到數據框中。

### 2. 創建訓練數據集
- 將股票數據轉換為LSTM模型的訓練數據集。這裡使用了MinMaxScaler進行數據標準化。
- 根據窗口大小（window_size）創建特徵和標籤。特徵是窗口內的股票價格和移動平均線，標籤是窗口結束後的價格變動方向（上漲或下跌）。

### 3. 訓練LSTM模型
- 定義LSTM模型的結構，包括LSTM層、全連接層、激活函數、Dropout層和Batch Normalization層。
- 使用訓練數據集進行模型訓練，優化損失函數（交叉熵損失）並更新模型參數。

### 4. 回測策略
- 使用Backtrader回測引擎運行交易策略。
- 策略根據LSTM模型的預測結果進行交易決策。如果預測價格會上漲且目前沒有持倉，則買入股票；如果預測價格會下跌且目前有持倉，則賣出股票。
- 策略還包括止損和止盈邏輯：如果價格下跌超過10%或上漲超過50%，則賣出股票。

### 5. 結果展示
- 回測完成後，顯示最終的投資組合價值、盈虧（P/L）和投資報酬率（ROI）。
- 使用Matplotlib繪製回測結果的K線圖，並嵌入到Streamlit應用中展示。

## 使用方法

1. 在Streamlit應用中輸入股票符號、開始日期和結束日期。
2. 調整短期和長期移動平均線的參數、交易手續費、每次交易金額和初始現金。
3. 點擊“開始回測”按鈕，系統會自動訓練LSTM模型並運行回測策略，最終展示回測結果。

這個策略結合了技術分析（移動平均線）和機器學習（LSTM模型）的優勢，旨在提高交易決策的準確性和收益率。
""")

# 用戶輸入參數
symbol = st.text_input("股票代碼，台股請記得在最後加上.TW", "AAPL")
start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"))
end_date = st.date_input("結束日期", pd.to_datetime("today"))

short_period = st.slider("短期均線", 1, 60, 5)
long_period = st.slider("長期均線", 30, 200, 60)
commission = st.slider('交易手續費 (%)', min_value=0.0, max_value=1.0, step=0.0005, format="%.4f", value=0.001)
trade_amount = st.slider("每次交易金額", min_value=0, max_value=50000, step=1000, value=1000)
initial_cash = st.slider("預算", min_value=0, max_value=5000000, step=10000, value=10000)

# 檢查短期均線是否大於長期均線
if short_period >= long_period:
    st.error("短期均線的窗口期不能大於或等於長期均線的窗口期，請調整參數。")
# 檢查時間跨度是否足夠長
elif (end_date - start_date).days < long_period + window_size:
        st.error(f"時間跨度過短，請選擇至少 {long_period + window_size} 天的時間跨度。")
else:
    if st.button("開始回測"):
        lstm_model_ready = False
        trained_model = None
        
        # 獲取股票數據
        stock_data = get_stock_data(symbol, start_date, end_date)
        
        # 檢查股票數據是否成功下載
        if stock_data is None:
            st.error("股票代碼錯誤或無法下載數據，請檢查輸入的股票代碼。")
        else:
            try:
                # 檢查股票數據是否足夠
                if len(stock_data) < window_size + 2:
                    st.error("股票數據不足以創建訓練數據集，請選擇更長的時間範圍。")
                else:
                    # 執行訓練和回測
                    train_lstm()
                    run_backtrader()
            except ValueError as e:
                st.error(f"數據集創建失敗: {e}")