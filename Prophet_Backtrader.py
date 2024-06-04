import streamlit as st
import pandas as pd
import yfinance as yf
import backtrader as bt
import datetime
import plotly
import matplotlib.pyplot as plt
from prophet import Prophet
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import requests
from io import BytesIO
import random
import base64
import numpy as np
from matplotlib.animation import FuncAnimation
from streamlit_tags import st_tags

# 設置 Matplotlib 背景顏色
plt.rcParams['axes.facecolor'] = 'black'  # 設置圖表區域背景顏色為黑色
plt.rcParams['figure.facecolor'] = 'black'  # 設置整個圖表背景顏色為黑色
plt.rcParams['text.color'] = 'gray'  # 設置圖表文字顏色為白色
fig, ax = plt.subplots()

# 調整標註的底色為黑色
# 這行代碼應該在圖例存在的情況下執行
if ax.get_legend() is not None:
    legend = ax.legend()
    legend.get_frame().set_facecolor('black')

# Prophet 預測函數
def predict_stock(selected_stock, n_years):
    try:
        data = yf.download(selected_stock, start="2010-01-01", end=datetime.date.today().strftime("%Y-%m-%d"))
        if data.empty:
            raise ValueError("無法抓取股票數據")
        data.reset_index(inplace=True)

        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=n_years * 365)
        forecast = m.predict(future)

        return data, forecast, m
    except Exception as e:
        raise ValueError(f"抓取股票數據時發生錯誤: {e}")

class PeriodicInvestmentStrategy(bt.Strategy):
    params = (
        ('monthly_investment', None),  # 每期投資金額
        ('commission', None),  # 手續費
        ('investment_day', None),  # 投資日
        ('printlog', True),  # 是否打印交易日誌
    )

    def __init__(self, **kwargs):
        self.order = None
        self.add_timer(
            when=bt.Timer.SESSION_START,
            monthdays=[self.params.investment_day],  # 每月的特定日期投資
            monthcarry=True,  # 如果特定日期不是交易日，則延至下一個交易日
        )

        # 從kwargs中獲取初始資金
        self.initial_cash = kwargs.get('initial_cash', 10000)  # 初始資金設置為10000

    def notify_timer(self, timer, when, *args, **kwargs):
        self.log('進行定期投資')
        # 獲取當前價格
        price = self.data.close[0]
        # 計算購買數量
        investment_amount = self.params.monthly_investment / price
        # 檢查資金是否足夠
        if self.broker.get_cash() >= self.params.monthly_investment:
            # 執行購買
            self.order = self.buy(size=investment_amount)

    def log(self, txt, dt=None):
        ''' 日誌函數 '''
        dt = dt or self.datas[0].datetime.date(0)
        if self.params.printlog:
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                cost = order.executed.price * order.executed.size
                commission = cost * self.params.commission / 100  # 將百分比轉換為小數
                self.log('買入執行, 價格: %.2f, 成本: %.2f, 手續費: %.2f' %
                        (order.executed.price, cost, commission))

            elif order.issell():
                self.log('賣出執行, 價格: %.2f, 成本: %.2f, 手續費: %.2f' %
                        (order.executed.price,
                        order.executed.value,
                        order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('訂單 取消/保證金不足/拒絕')

        self.order = None

# 以50%的機率選擇圖片連結
if random.random() < 0.5:
    image_url = 'https://raw.githubusercontent.com/j7808833/test_02/main/pic/Cyberpunk_bar_03.gif'
else:
    image_url = 'https://raw.githubusercontent.com/j7808833/test_02/main/pic/Cyberpunk_bar_02.gif'

# 顯示GIF圖片
st.markdown(f'<img src="{image_url}" style="width: 100%;">', unsafe_allow_html=True)

# Streamlit 頁面佈局
st.title('Prophet & Backtrader  Bar')

# 提示用戶輸入股票代碼，並使用逗號分隔
user_input = st.text_area("請輸入股票代碼，用逗號分隔，台股請記得在最後加上.TW", "AAPL, MSFT, GOOG, AMZN, 0050.TW")
# 將用戶輸入的股票代碼轉換為列表
stocks = [stock.strip() for stock in user_input.split(",")]
st.write("您輸入的股票代碼：", stocks)

# 股票選擇器和預測年限滑塊
selected_stock = st.selectbox('選擇股票進行預測和回測', stocks)
n_years = st.slider('預測年限:', 1, 3)

# 預測和顯示結果
if st.button('運行預測'):
    try:
        # 做預測並獲取數據、預測結果和 Prophet 模型
        data, forecast, m = predict_stock(selected_stock, n_years)
        st.write('預測數據:')
        st.write(forecast)
        st.write(f'{n_years} 年的預測圖')
        fig1 = m.plot(forecast)
        
        # 調整底色
        fig1.set_facecolor('black')

        # 調整網格繪圖區顏色
        for ax in fig1.axes:
            ax.set_facecolor('black')
            ax.tick_params(axis='x', colors='white')  # 調整x軸刻度顏色為白色
            ax.tick_params(axis='y', colors='white')  # 調整y軸刻度顏色為白色
            ax.yaxis.label.set_color('white')  # 調整y軸標籤顏色為白色
            ax.xaxis.label.set_color('white')  # 調整x軸標籤顏色為白色

        # 調整數值和框線顏色
        for text in fig1.findobj(match=matplotlib.text.Text):
            text.set_color('white')

        # 修改折線和點的顏色
        for dot in fig1.findobj(match=matplotlib.patches.Circle):
            dot.set_edgecolor('white')  # 點的邊緣顏色
            dot.set_facecolor('white')  # 點的填充顏色

        st.pyplot(fig1)
        st.success('您的股票預測已生成！')
    except ValueError as e:
        st.error('無法抓取股票數據，請檢查股票編號是否正確。')
    except Exception as e:
        st.error(f'發生錯誤: {e}')

st.markdown("***")

# 添加滑塊來控制參數
initial_cash = st.slider('預算', min_value=0, max_value=5000000, step=10000, value=10000)
monthly_investment = st.slider('每月投資金額', min_value=0, max_value=50000, step=1000, value=1000)
commission = st.slider('手續費 (%)', min_value=0.0, max_value=1.0, step=0.0005, format="%.4f", value=0.001)
investment_day = st.slider('每月投資日', min_value=1, max_value=28, step=1, value=1)
n_years_backtest = st.slider('回測持續時間 (年)', min_value=1, max_value=10, step=1, value=5)

if initial_cash == 0:
    st.error("預算不可以為0")

# 定義顯示結果的函數
def display_results(cash, value, initial_value, n_years):
    # 計算年回報率
    annual_return = ((value - cash) / (initial_cash - cash)) ** (1 / n_years) - 1
    annual_return *= 100  # 轉換為百分比形式
    total_return = ((value - cash) / (initial_cash - cash)) - 1
    total_return *= 100  # 轉換為百分比形式

    # 計算預算變化
    budget_delta = initial_value - initial_cash

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

def get_drink_name(investment_ratio, commission, annual_return):
    if investment_ratio > 0.1:
        if commission < 0.15:
            if annual_return <= 2:
                return "Vodka_Soda"
            elif annual_return <= 5:
                return "Vodka_Martini"
            elif annual_return <= 10:
                return "Whiskey_Sour"
            else:
                return "Whiskey_Neat"
        else:
            if annual_return <= 2:
                return "Moscow_Mule"
            elif annual_return <= 5:
                return "Bloody_Mary"
            elif annual_return <= 10:
                return "Old_Fashioned"
            else:
                return "Manhattan"
    else:
        if commission < 0.15:
            if annual_return <= 2:
                return "Screwdriver"
            elif annual_return <= 5:
                return "Vodka_Collins"
            elif annual_return <= 10:
                return "Rob_Roy"
            else:
                return "Sazerac"
        else:
            if annual_return <= 2:
                return "Aperol_Spritz"
            elif annual_return <= 5:
                return "Cosmopolitan"
            elif annual_return <= 10:
                return "Boulevardier"
            else:
                return "Vieux_Carré"

# 調酒信息
drinks_info = {
    "Vodka_Soda": {
        "報酬率": 1,
        "金額大小": 1,
        "特性": "清新的氣味和輕盈的感覺象徵著保守和穩健的投資風格。適合謹慎型投資者，短期內尋求低風險回報。",
        "成分": ["伏特加", "蘇打水"],
        "酒精濃度": 10,
        "口感": "甘口",
        "建議杯型": "高球杯",
        "調製法": "直調法",
        "風味": "柑橘香",
        "飲用時間": 5,
        "飲用溫度": 3,
        "投資建議": "適合低風險、穩健型的短期投資者，建議選擇穩定性較高的基金或定存。"
    },
    "Vodka_Martini": {
        "報酬率": 2,
        "金額大小": 1,
        "特性": "辛辣且微苦的味道代表了適中的風險，投資者具有一定的冒險精神，追求平衡的短期回報。",
        "成分": ["伏特加", "乾苦艾酒"],
        "酒精濃度": 30,
        "口感": "中口",
        "建議杯型": "馬丁尼杯",
        "調製法": "攪拌法",
        "風味": "草本香",
        "飲用時間": 5,
        "飲用溫度": 3,
        "投資建議": "適合中等風險投資者，建議選擇平衡型基金或股票，追求穩健與回報的平衡。"
    },
    "Whiskey_Sour": {
        "報酬率": 3,
        "金額大小": 1,
        "特性": "濃烈的味道和多層次的口感象徵著積極進取的投資策略，投資者願意承擔高風險以換取高回報。",
        "成分": ["威士忌", "檸檬汁", "糖漿"],
        "酒精濃度": 40,
        "口感": "中口",
        "建議杯型": "古典杯",
        "調製法": "搖盪法",
        "風味": "柑橘香",
        "飲用時間": 7,
        "飲用溫度": 5,
        "投資建議": "適合高風險、高回報的投資者，建議選擇成長型股票或高收益債券。"
    },
    "Whiskey_Neat": {
        "報酬率": 4,
        "金額大小": 1,
        "特性": "強烈且直截了當的風味比喻極端冒險的投資風格，適合非常自信且追求極高回報的投資者。",
        "成分": ["純飲威士忌"],
        "酒精濃度": 50,
        "口感": "辛口",
        "建議杯型": "威士忌杯",
        "調製法": "直調法",
        "風味": "木質香",
        "飲用時間": 5,
        "飲用溫度": 20,
        "投資建議": "適合極高風險承受能力的投資者，建議選擇高波動性的股票或新興市場投資。"
    },
    "Moscow_Mule": {
        "報酬率": 1,
        "金額大小": 2,
        "特性": "溫和且帶有薑味的口感象徵著謹慎且穩定的投資策略，適合大額低風險的投資。",
        "成分": ["伏特加", "薑汁啤酒", "青檸汁"],
        "酒精濃度": 10,
        "口感": "甘口",
        "建議杯型": "銅杯",
        "調製法": "直調法",
        "風味": "薑香",
        "飲用時間": 7,
        "飲用溫度": 5,
        "投資建議": "適合大額低風險的投資者，建議選擇國債或高評級的企業債券。"
    },
    "Bloody_Mary": {
        "報酬率": 2,
        "金額大小": 2,
        "特性": "豐富且多層次的味道代表著多元化的投資策略，適合大額中等風險的投資者。",
        "成分": ["伏特加", "番茄汁", "各種調味料"],
                "酒精濃度": 20,
        "口感": "中口",
        "建議杯型": "高球杯",
        "調製法": "攪拌法",
        "風味": "番茄香",
        "飲用時間": 5,
        "飲用溫度": 3,
        "投資建議": "適合大額中等風險的投資者，建議選擇多元資產配置的基金或ETF。"
    },
    "Old_Fashioned": {
        "報酬率": 3,
        "金額大小": 2,
        "特性": "經典且濃烈的口感象徵著強勢且積極的投資策略，適合大額高風險的投資。",
        "成分": ["威士忌", "苦味酒", "糖"],
        "酒精濃度": 40,
        "口感": "辛口",
        "建議杯型": "古典杯",
        "調製法": "攪拌法",
        "風味": "柑橘香",
        "飲用時間": 7,
        "飲用溫度": 5,
        "投資建議": "適合大額高風險的投資者，建議選擇藍籌股或高收益股票。"
    },
    "Manhattan": {
        "報酬率": 4,
        "金額大小": 2,
        "特性": "非常濃烈且複雜的味道象徵著精密且策略性強的投資風格，適合追求極高回報的大額投資者。",
        "成分": ["威士忌", "甜苦艾酒", "苦味酒"],
        "酒精濃度": 45,
        "口感": "辛口",
        "建議杯型": "馬丁尼杯",
        "調製法": "攪拌法",
        "風味": "木質香",
        "飲用時間": 5,
        "飲用溫度": 20,
        "投資建議": "適合極高風險承受能力的大額投資者，建議選擇私募股權或高風險的對沖基金。"
    },
    "Screwdriver": {
        "報酬率": 1,
        "金額大小": 1,
        "特性": "清新的橙汁味道代表著穩健和簡單的投資策略，適合保守型投資者，追求長期穩定的回報。",
        "成分": ["伏特加", "橙汁"],
        "酒精濃度": 10,
        "口感": "甘口",
        "建議杯型": "高球杯",
        "調製法": "直調法",
        "風味": "橙香",
        "飲用時間": 5,
        "飲用溫度": 3,
        "投資建議": "適合小額低風險的長期投資者，建議選擇定期存款或保本型理財產品。"
    },
    "Vodka_Collins": {
        "報酬率": 2,
        "金額大小": 1,
        "特性": "清爽的口感和適中的甜味象徵著平衡且多元的投資策略，適合希望在長期內獲得穩定回報的投資者。",
        "成分": ["伏特加", "檸檬汁", "糖漿", "蘇打水"],
        "酒精濃度": 20,
        "口感": "中口",
        "建議杯型": "高球杯",
        "調製法": "搖盪法",
        "風味": "柑橘香",
        "飲用時間": 5,
        "飲用溫度": 3,
        "投資建議": "適合小額中等風險的長期投資者，建議選擇混合型基金或債券基金。"
    },
    "Rob_Roy": {
        "報酬率": 3,
        "金額大小": 1,
        "特性": "經典而濃烈的口感象徵著經驗豐富的投資者，具有高風險承受能力，追求長期的高回報。",
        "成分": ["威士忌", "甜苦艾酒", "苦味酒"],
        "酒精濃度": 40,
        "口感": "辛口",
        "建議杯型": "馬丁尼杯",
        "調製法": "攪拌法",
        "風味": "木質香",
        "飲用時間": 5,
        "飲用溫度": 20,
        "投資建議": "適合小額高風險的長期投資者，建議選擇高成長股票或國際股票基金。"
    },
    "Sazerac": {
        "報酬率": 4,
        "金額大小": 1,
        "特性": "複雜而濃烈的風味象徵著非常精細和策略性的投資風格，適合追求極高回報並願意承擔高風險的投資者。",
        "成分": ["威士忌", "苦艾酒", "苦味酒"],
        "酒精濃度": 45,
        "口感": "辛口",
        "建議杯型": "古典杯",
        "調製法": "攪拌法",
        "風味": "香草香",
        "飲用時間": 5,
        "飲用溫度": 20,
        "投資建議": "適合小額極高風險的長期投資者，建議選擇創投基金或高風險的衍生品。"
    },
    "Aperol_Spritz": {
        "報酬率": 1,
        "金額大小": 2,
        "特性": "溫和且帶有薑味的口感象徵著謹慎且穩定的投資策略，適合大額低風險的投資。",
        "成分": ["Aperol", "蘇打水", "香檳"],
        "酒精濃度": 8,
        "口感": "甘口",
        "建議杯型": "笛型杯",
        "調製法": "直調法",
        "風味": "柑橘香",
        "飲用時間": 5,
        "飲用溫度": 3,
        "投資建議": "適合大額低風險的投資者，建議選擇高評級債券或優質藍籌股。"
    },
    "Cosmopolitan": {
        "報酬率": 2,
        "金額大小": 2,
        "特性": "帶有水果味的口感代表著平衡且多元的投資策略，適合希望在長期內獲得穩定回報的投資者。",
        "成分": ["伏特加", "柑橘利口酒", "蔓越莓汁", "青檸汁"],
        "酒精濃度": 20,
        "口感": "中口",
        "建議杯型": "馬丁尼杯",
        "調製法": "搖盪法",
        "風味": "果香",
        "飲用時間": 5,
        "飲用溫度": 3,
        "投資建議": "適合大額中等風險的長期投資者，建議選擇多元化的國際股票基金或混合型基金。"
    },
    "Boulevardier": {
        "報酬率": 3,
        "金額大小": 2,
        "特性": "濃烈且複雜的口感象徵著強勢且積極的投資策略，適合大額高風險的投資。",
        "成分": ["威士忌", "甜苦艾酒", "苦味酒"],
        "酒精濃度": 40,
        "口感": "辛口",
        "建議杯型": "古典杯",
        "調製法": "攪拌法",
        "風味": "香料香",
        "飲用時間": 5,
        "飲用溫度": 20,
                "投資建議": "適合大額高風險的投資者，建議選擇全球大宗商品或能源股票。"
    },
    "Vieux_Carré": {
        "報酬率": 4,
        "金額大小": 2,
        "特性": "非常濃烈且複雜的味道象徵著精密且策略性強的投資風格，適合追求極高回報的大額投資者。",
        "成分": ["威士忌", "干邑", "甜苦艾酒", "苦味酒"],
        "酒精濃度": 50,
        "口感": "辛口",
        "建議杯型": "古典杯",
        "調製法": "攪拌法",
        "風味": "木質香",
        "飲用時間": 5,
        "飲用溫度": 20,
        "投資建議": "適合大額極高風險的投資者，建議選擇私募股權基金或對沖基金。"
    }
}

# 雷達圖數據
radar_data = {
    "Vodka_Soda": [1, 1, 1, 1, 1],
    "Vodka_Martini": [2, 2, 2, 2, 1],
    "Whiskey_Sour": [3, 3, 3, 3, 1],
    "Whiskey_Neat": [4, 4, 4, 4, 1],
    "Moscow_Mule": [1, 1, 1, 1, 2],
    "Bloody_Mary": [2, 2, 2, 2, 2],
    "Old_Fashioned": [3, 3, 3, 3, 2],
    "Manhattan": [4, 4, 4, 4, 2],
    "Screwdriver": [1, 1, 1, 1, 3],
    "Vodka_Collins": [2, 2, 2, 2, 3],
    "Rob_Roy": [3, 3, 3, 3, 3],
    "Sazerac": [4, 4, 4, 4, 3],
    "Aperol_Spritz": [1, 1, 1, 1, 4],
    "Cosmopolitan": [2, 2, 2, 2, 4],
    "Boulevardier": [3, 3, 3, 3, 4],
    "Vieux_Carré": [4, 4, 4, 4, 4],
}

# 定義指標標籤
attribute_labels = ['Risk', 'Returns', 'Complexity', 'Alcohol Content', 'Investment Duration']

# 新的指標標籤
attribute_labels_extended = [
    'Volatility', 'Maximum Drawdown', 'Historical Returns', 
    'Expense Ratio', 'Fund Size', 'Sharpe Ratio'
]

def make_radar_chart(name, stats, attribute_labels):
    labels = np.array(attribute_labels[:len(stats)])
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    stats = stats + stats[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='magenta', alpha=0.25)
    ax.plot(angles, stats, color='magenta', linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='white', fontsize=10)  # 修改指標標籤顏色和字體大小

    plt.title(name, size=10, color='white', y=1.1)  # 修改標題顏色和字體大小
    st.pyplot(fig)

# 執行回測並顯示結果
if st.button('Run Backtest'):
    try:
        # 初始化 Cerebro 引擎
        cerebro = bt.Cerebro()
        cerebro.addstrategy(PeriodicInvestmentStrategy, initial_cash=initial_cash, monthly_investment=monthly_investment, commission=commission, investment_day=investment_day)

        # 添加數據
        start_date = datetime.datetime.now() - relativedelta(years=n_years_backtest)  # 根據回測年限動態計算開始時間
        data = yf.download(selected_stock,
                        start=start_date,
                        end=datetime.datetime.now())
        if data.empty:
            raise ValueError("無法抓取股票數據")
        cerebro.adddata(bt.feeds.PandasData(dataname=data))

        # 設置初始資本
        cerebro.broker.setcash(initial_cash)

        # 設置每筆交易的手續費
        cerebro.broker.setcommission(commission=commission)

        # 執行策略
        cerebro.run()

        # 獲取初始總價值
        initial_value = cerebro.broker.get_value()

        # 獲取當前現金餘額和總價值
        cash = cerebro.broker.get_cash()
        value = cerebro.broker.get_value()

        # 顯示結果
        annual_return = display_results(cash, value, initial_value, n_years_backtest)

        # 繪製結果
        fig = cerebro.plot(style='plotly')[0][0]  # 獲取 Matplotlib 圖形對象
        st.pyplot(fig)  # 將圖形嵌入到 Streamlit 頁面中

        # 顯示成功提示
        st.toast('Your stock has been generated!', icon='🥂')

        # 計算投資比例
        investment_ratio = monthly_investment / initial_cash if initial_cash != 0 else float('inf')

        # 計算年化回報率
        annual_return = ((value - initial_cash) / initial_cash + 1) ** (1 / n_years_backtest) - 1
        annual_return *= 100  # 轉換為百分比形式

        # 根據投資參數查找對應的調酒名稱
        drink_name = get_drink_name(investment_ratio, commission, annual_return)
        
        # 調酒圖片 URL 字典
        drink_images = {
            "Vodka_Soda": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_01_Vodka%20Soda.jpg",
            "Vodka_Martini": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_02_Vodka%20Martini.jpg",
            "Whiskey_Sour": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_03_Whiskey%20Sour.jpg",
            "Whiskey_Neat": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_04_Whiskey%20Neat.jpg",
            "Moscow_Mule": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_05_Moscow%20Mule.jpg",
            "Bloody_Mary": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_06_Bloody%20Mary.jpg",
            "Old_Fashioned": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_07_Old%20Fashioned.jpg",
            "Manhattan": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_08_Manhattan.jpg",
            "Screwdriver": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_09_Screwdriver.jpg",
            "Vodka_Collins": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_10_Vodka%20Collins.jpg",
            "Rob_Roy": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_11_Rob%20Roy.jpg",
            "Sazerac": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_12_Sazerac.jpg",
            "Aperol_Spritz": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_13_Aperol%20Spritz.jpg",
            "Cosmopolitan": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_14_Cosmopolitan.jpg",
            "Boulevardier": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_15_Boulevardier.jpg",
            "Vieux_Carré": "https://raw.githubusercontent.com/j7808833/test_02/main/pic/cocktail_16_Vieux%20Carr%C3%A9.jpg"
        }

        # 顯示調酒圖片
        image_url = drink_images[drink_name]
        response = requests.get(image_url)
        drink_image = Image.open(BytesIO(response.content))
        st.markdown(f'<p align="center"><img src="{image_url}" alt="{drink_name}" width="240"></p>', unsafe_allow_html=True)

        # 顯示特性和成分
        if drink_name in drinks_info:
            st.markdown(f"""
            <div style="border:2px solid #00BFFF; padding: 10px;">
                <h2 style="color: #00BFFF;">調酒名稱：<strong>{drink_name}</strong></h2>
                <p style="font-size: 16px; font-weight: bold;">成分：<span style="font-size: 10pt; color: #FF00FF;">{', '.join(drinks_info[drink_name]['成分'])}</span></p>
                <p style="font-size: 16px; font-weight: bold;">口感：<span style="font-size: 10pt; color: #FF00FF;">{drinks_info[drink_name]['口感']}</span></p>
                <p style="font-size: 16px; font-weight: bold;">建議杯型：<span style="font-size: 10pt; color: #FF00FF;">{drinks_info[drink_name]['建議杯型']}</span></p>
                <p style="font-size: 16px; font-weight: bold;">調製法：<span style="font-size: 10pt; color: #FF00FF;">{drinks_info[drink_name]['調製法']}</span></p>
                <p style="font-size: 16px; font-weight: bold;">風味：<span style="font-size: 10pt; color: #FF00FF;">{drinks_info[drink_name]['風味']}</span></p>
                <p style="font-size: 16px; font-weight: bold;">特性：<span style="font-size: 10pt; color: #FF00FF;">{drinks_info[drink_name]['特性']}</span></p>
                <p style="font-size: 16px; font-weight: bold;">投資建議：<span style="font-size: 10pt; color: #FF00FF;">{drinks_info[drink_name]['投資建議']}</span></p>
            </div>
            """, unsafe_allow_html=True)

            # 獲取酒精濃度、飲用時間和飲用溫度的整數值
            alcohol_content_value = drinks_info[drink_name]["酒精濃度"]
            drinking_time_value = drinks_info[drink_name]["飲用時間"]
            drinking_temp_value = drinks_info[drink_name]["飲用溫度"]

            # 創建包含酒精濃度的 DataFrame
            alcohol_df = pd.DataFrame(
                {
                    "酒精濃度": [alcohol_content_value]
                }
            )

            # 創建包含飲用時間的 DataFrame
            time_df = pd.DataFrame(
                {
                    "飲用時間": [drinking_time_value]
                }
            )

            # 創建包含飲用溫度的 DataFrame
            temp_df = pd.DataFrame(
                {
                    "飲用溫度": [drinking_temp_value]
                }
            )

            # 顯示帶有進度條的 DataFrame
            st.data_editor(
                alcohol_df,
                column_config={
                    "酒精濃度": st.column_config.ProgressColumn(
                        "酒精濃度",
                        help="酒精濃度",
                        format="%d",
                        min_value=0,
                        max_value=50,
                    ),
                },
                hide_index=True,
                width=800  # 設置寬度為800像素
            )

            st.data_editor(
                time_df,
                column_config={
                    "飲用時間": st.column_config.ProgressColumn(
                        "飲用時間",
                        help="飲用時間",
                        format="%d",
                        min_value=0,
                        max_value=10,
                    ),
                },
                hide_index=True,
                width=800  # 設置寬度為800像素
            )

            st.data_editor(
                temp_df,
                column_config={
                    "飲用溫度": st.column_config.ProgressColumn(
                        "飲用溫度",
                        help="飲用溫度",
                        format="%d",
                        min_value=0,
                        max_value=20,
                    ),
                },
                hide_index=True,
                width=800  # 設置寬度為800像素
            )
        else:
            st.write("找不到對應的調酒信息。")

        # 顯示對應的雷達圖
        stats = radar_data[drink_name]
        make_radar_chart(drink_name, stats, attribute_labels_extended)

    except ValueError as e:
        st.error('無法抓取股票數據，請檢查股票編號是否正確。')
    except Exception as e:
        st.error(f'發生錯誤: {e}')