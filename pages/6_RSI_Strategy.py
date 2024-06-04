import streamlit as st
import yfinance as yf
import backtrader as bt
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import random

# 定義 RSI 策略
class RSIStrategy(bt.Strategy):
    params = (
        ('printlog', True),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('trade_amount', 1000),  # 每次交易的固定投入金額
    )

    def __init__(self):
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.data.close, period=self.params.rsi_period)

    def next(self):
        if not self.position:  # 沒有持倉
            if self.rsi < self.params.rsi_oversold:
                size = self.params.trade_amount // self.data.close[0]
                self.buy(size=size)  # RSI 低於超賣區域，買入
        else:
            if self.rsi > self.params.rsi_overbought:
                self.sell(size=self.position.size)  # RSI 高於超買區域，賣出

def display_results(initial_cash, final_value, start_date, end_date):
    # 計算年回報率
    duration_days = (end_date - start_date).days
    duration_years = duration_days / 365.25
    cagr = ((final_value / initial_cash) ** (1 / duration_years)) - 1
    annual_return = cagr * 100  # 轉換為百分比形式
    total_return = (final_value - initial_cash) / initial_cash * 100  # 轉換為百分比形式

    # 計算預算變化
    budget_delta = final_value - initial_cash

    # 設置獨立變數來顯示 delta 值
    budget_delta_display = f"${budget_delta:.2f}"
    annual_return_display = f"{annual_return:.2f}%"
    total_return_display = f"{total_return:.2f}%"

    # 創建多列佈局
    col1, col2, col3 = st.columns(3)

    # 在第一列中顯示預算
    custom_metric(col1, "預算", f"${initial_cash:.2f}", budget_delta_display)

    # 在第二列中顯示最終價值
    custom_metric(col2, "最終價值", f"${final_value:.2f}", "")

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

# Streamlit 應用程式
st.title("RSI 股票交易策略")

st.markdown("""
## 策略說明

RSI（相對強弱指數）是一種動量指標，用於衡量股票價格變動的速度和變動的幅度。RSI 的值在 0 到 100 之間波動，通常用來識別超買和超賣狀態。

### 策略邏輯
1. **RSI 計算**：
   - RSI 是根據一定的週期（例如 14 天）計算的。
   - 當 RSI 的值低於某個閾值（例如 30）時，表示股票可能被超賣，這是一個買入信號。
   - 當 RSI 的值高於某個閾值（例如 70）時，表示股票可能被超買，這是一個賣出信號。

2. **交易決策**：
   - 當 RSI 低於超賣區域（例如 30）時，策略會買入股票。
   - 當 RSI 高於超買區域（例如 70）時，策略會賣出股票。

### 使用方法
1. 輸入股票符號、開始和結束日期。
2. 設置 RSI 的週期、超買和超賣閾值。
3. 設置初始現金和每次交易的固定投入金額。
4. 設置交易手續費。
5. 點擊“開始回測”按鈕，運行回測策略並顯示結果。
""")

# 使用者輸入參數
symbol = st.text_input("股票符號", "AAPL")
start_date = st.date_input("開始日期", datetime(2020, 1, 1))
end_date = st.date_input("結束日期", datetime.today())
rsi_period = st.slider("RSI 週期", 1, 50, 14)
rsi_overbought = st.slider("RSI 超買區域", 50, 100, 70)
rsi_oversold = st.slider("RSI 超賣區域", 0, 50, 30)
initial_cash = st.slider("預算", min_value=0, max_value=5000000, step=10000, value=10000)
trade_amount = st.slider("每次交易金额", min_value=0, max_value=50000, step=1000, value=1000)
commission = st.slider('交易手續費 (%)', min_value=0.0, max_value=1.0, step=0.0005, format="%.4f", value=0.001)

if st.button("開始回測"):
    try:
        # 檢查 RSI 週期是否小於回測期間的天數
        if (end_date - start_date).days < rsi_period:
            st.error("RSI 週期必須小於回測期間的天數，請調整參數。")
        else:
            # 獲取股票數據
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                st.error("無法下載股票數據，請檢查股票代碼和日期範圍。")
            else:
                # 檢查數據是否足夠
                if len(data) < rsi_period:
                    st.error("股票數據不足以計算 RSI，請選擇更長的時間範圍。")
                else:
                    data = bt.feeds.PandasData(dataname=data)

                    # 創建回測引擎
                    cerebro = bt.Cerebro()
                    cerebro.adddata(data)
                    cerebro.addstrategy(RSIStrategy, rsi_period=rsi_period, rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold, trade_amount=trade_amount)
                    cerebro.broker.set_cash(initial_cash)
                    cerebro.broker.setcommission(commission=commission / 100)  # 設置手續費

                    # 運行回測
                    st.write("開始回測...")
                    cerebro.run()
                    final_portfolio_value = cerebro.broker.getvalue()

                    # 顯示結果
                    display_results(initial_cash, final_portfolio_value, start_date, end_date)

                    # 繪製回測結果
                    fig = cerebro.plot(style='candlestick')[0][0]
                    st.pyplot(fig)
    except Exception as e:
        st.error(f"出現錯誤: {e}")