import streamlit as st
import yfinance as yf
import pandas as pd
import backtrader as bt
import matplotlib
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import random

# 定義MACD策略
class MACDStrategy(bt.Strategy):
    params = (
        ('printlog', True),
        ('fast', 12),
        ('slow', 26),
        ('signal', 9),
    )

    def __init__(self):
        macd = bt.indicators.MACD(self.data.close, 
                                  period_me1=self.params.fast, 
                                  period_me2=self.params.slow, 
                                  period_signal=self.params.signal)
        self.macd = macd.macd
        self.signal = macd.signal
        self.crossover = bt.indicators.CrossOver(self.macd, self.signal)

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(
                    "SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm))

    def next(self):
        self.log("Close, %.2f" % self.data.close[0])
        if self.crossover > 0:  # 買入信號
            if not self.position:
                self.log("BUY CREATE, %.2f" % self.data.close[0])
                self.buy()
        elif self.crossover < 0:  # 賣出信號
            if self.position:
                self.log("SELL CREATE, %.2f" % self.data.close[0])
                self.sell()

    def stop(self):
        self.log("Ending Value %.2f" % (self.broker.getvalue()), doprint=True)

# 自定义Sizer，根據每次交易的金額計算股數
class FixedCashSizer(bt.Sizer):
    params = (('cash', None),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return self.params.cash // data.close[0]
        return self.broker.getposition(data).size

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

# Streamlit 用户界面
st.title("MACD 股票交易策略")

# 策略說明
st.header("策略說明")
st.markdown("""
MACD策略是一種基於移動平均線聚合散度（MACD）指標的交易策略。MACD 通常由三部分組成：MACD 線、信號線和柱狀圖。
- **MACD線**：由兩條指數移動平均線（EMA）的差值計算得來，通常使用12日和26日的EMA。
- **信號線**：MACD線的9日EMA，用於生成交易信號。
- **柱狀圖**：MACD線和信號線之間的差值，用於視覺化兩者之間的關係。

**買入信號**：當MACD線從下方穿過信號線時，產生買入信號。
**賣出信號**：當MACD線從上方穿過信號線時，產生賣出信號。
""")

# 用户输入参数
symbol = st.text_input("股票符號", "AAPL")
start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"))
end_date = st.date_input("結束日期", pd.to_datetime("today"))

fast_ema = st.slider('快速EMA周期', min_value=1, max_value=50, value=12)
slow_ema = st.slider('慢速EMA周期', min_value=1, max_value=50, value=26)
signal_ema = st.slider('信號EMA周期', min_value=1, max_value=50, value=9)
commission = st.slider('交易手續费 (%)', min_value=0.0, max_value=0.5, step=0.0005, format="%.4f", value=0.001)
trade_cash = st.slider("每次交易金额", min_value=0, max_value=50000, step=1000, value=1000)
initial_cash = st.slider("初始现金", min_value=0, max_value=10000000, step=10000, value=10000)

# 檢查均線設置是否符合要求
if fast_ema >= slow_ema or signal_ema >= slow_ema:
    st.error("快速EMA必須小於慢速EMA，信號EMA必須小於慢速EMA，請調整參數。")
# 檢查時間跨度是否足夠長
elif (end_date - start_date).days < slow_ema:
    st.error(f"時間跨度過短，請選擇至少 {slow_ema} 天的時間跨度。")
else:
    if st.button("開始回測"):
        try:
            # 獲取股票數據
            df = yf.download(symbol, start=start_date, end=end_date)
            if df.empty:
                st.error("無法下載股票數據，請檢查股票代碼和日期範圍。")
            else:
                df.dropna(inplace=True)

                # 檢查數據是否足夠
                if len(df) < slow_ema:
                    st.error("股票數據不足以計算慢速EMA，請選擇更長的時間範圍。")
                else:
                    data = bt.feeds.PandasData(dataname=df)

                    # 創建回測引擎
                    cerebro = bt.Cerebro()
                    cerebro.addstrategy(
                        MACDStrategy,
                        fast=fast_ema,
                        slow=slow_ema,
                        signal=signal_ema
                    )
                    cerebro.adddata(data)
                    cerebro.broker.setcash(initial_cash)
                    cerebro.broker.setcommission(commission=commission / 100)  # 转换为十进制形式
                    cerebro.addsizer(FixedCashSizer, cash=trade_cash)  # 使用自定义的FixedCashSizer

                    # 運行回測
                    st.write("開始回測...")
                    cerebro.run()
                    final_value = cerebro.broker.getvalue()

                    # 顯示結果
                    display_results(initial_cash, final_value, start_date, end_date)

                    # 绘制回测结果
                    fig = cerebro.plot(style='candlestick')[0][0]
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"出現錯誤: {e}")