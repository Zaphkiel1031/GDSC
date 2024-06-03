import streamlit as st
import yfinance as yf
import pandas as pd
import backtrader as bt
import matplotlib
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
matplotlib.use('Agg')
import random

# 自定义指标
class MySignal(bt.Indicator):
    lines = ("signal",)
    params = dict(short_period=None, median_period=None, long_period=None)

    def __init__(self):
        self.s_ma = bt.ind.SMA(period=self.p.short_period)
        self.m_ma = bt.ind.SMA(period=self.p.median_period)
        self.l_ma = bt.ind.SMA(period=self.p.long_period)
        self.signal1 = bt.And(self.m_ma > self.l_ma, self.s_ma > self.m_ma)
        self.buy_signal = bt.If((self.signal1 - self.signal1(-1)) > 0, 1, 0)
        self.sell_signal = bt.ind.CrossDown(self.s_ma, self.m_ma)
        self.lines.signal = bt.Sum(self.buy_signal, self.sell_signal * (-1))

# 自定义Sizer
class FixedAmountSizer(bt.Sizer):
    params = (("amount", None),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            size = self.p.amount // data.close[0]
            return size
        return self.broker.getposition(data).size

# 策略
class TestStrategy(bt.Strategy):
    params = dict(
        printlog=True,
        short_period=None,
        median_period=None,
        long_period=None,
        initial_cash=None,
    )

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.signal = MySignal(
            self.datas[0],
            short_period=self.params.short_period,
            median_period=self.params.median_period,
            long_period=self.params.long_period
        )
        self.s_ma = bt.ind.SMA(period=self.params.short_period)
        self.m_ma = bt.ind.SMA(period=self.params.median_period)
        self.l_ma = bt.ind.SMA(period=self.params.long_period)
        bt.indicators.MACDHisto(self.datas[0])

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
        self.log("Close, %.2f" % self.dataclose[0])
        if self.order:
            return
        if not self.position:
            if self.signal.lines.signal[0] == 1:
                self.log("BUY CREATE, %.2f" % self.dataclose[0])
                self.order = self.buy()
        else:
            if self.signal.lines.signal[0] == -1:
                self.log("SELL CREATE, %.2f" % self.dataclose[0])
                self.order = self.sell()

    def stop(self):
        self.log("Ending Value %.2f" % (self.broker.getvalue()), doprint=True)

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
st.title("三均線股票交易策略")

# 策略說明
st.header("策略說明")
st.markdown("""
三均線策略是一種基於短期、中期和長期簡單移動平均線（SMA）交叉來生成買賣信號的交易策略。
- **短期均線（SMA短期）**：通常設置為5天，用於捕捉近期價格趨勢。
- **中期均線（SMA中期）**：通常設置為20天，用於反映較長時間段的價格趨勢。
- **長期均線（SMA長期）**：通常設置為60天，用於反映更長時間段的價格趨勢。

**買入信號**：當短期均線向上穿越中期均線，且中期均線高於長期均線時，產生買入信號。

**賣出信號**：當短期均線向下穿越中期均線時，產生賣出信號。
""")

# 用户输入参数
symbol = st.text_input("股票符號", "AAPL")
start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"))
end_date = st.date_input("结束日期", pd.to_datetime("today"))

short_period = st.slider("短期均線", 1, 30, 5)
median_period = st.slider("中期均線", 15, 100, 20)
long_period = st.slider("長期均線", 30, 200, 60)
commission = st.slider('交易手續费 (%)', min_value=0.0, max_value=0.5, step=0.0005, format="%.4f", value=0.001)
trade_amount = st.slider("每次交易金额", min_value=0, max_value=50000 , step=1000, value=1000)
initial_cash = st.slider("初始现金", min_value=0, max_value=10000000, step=10000, value=10000)

# 檢查均線設置是否符合要求
if short_period >= median_period or median_period >= long_period:
    st.error("短期均線必須小於中期均線，中期均線必須小於長期均線，請調整參數。")
# 檢查時間跨度是否足夠長
elif (end_date - start_date).days < long_period:
    st.error(f"時間跨度過短，請選擇至少 {long_period} 天的時間跨度。")
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
                if len(df) < long_period:
                    st.error("股票數據不足以計算長期均線，請選擇更長的時間範圍。")
                else:
                    data = bt.feeds.PandasData(dataname=df)

                    cerebro = bt.Cerebro()
                    cerebro.addstrategy(
                        TestStrategy,
                        short_period=short_period,
                        median_period=median_period,
                        long_period=long_period,
                        initial_cash=initial_cash
                    )
                    cerebro.adddata(data)
                    cerebro.broker.setcash(initial_cash)
                    cerebro.broker.setcommission(commission=commission / 100)  # 转换为十进制形式
                    cerebro.addsizer(FixedAmountSizer, amount=trade_amount)  # 使用自定义的FixedAmountSizer
                    cerebro.run()
                    final_value = cerebro.broker.getvalue()

                    # 顯示結果
                    display_results(initial_cash, final_value, start_date, end_date)

                    # 繪製回測結果
                    fig = cerebro.plot(style='candlestick')[0][0]
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"出現錯誤: {e}")