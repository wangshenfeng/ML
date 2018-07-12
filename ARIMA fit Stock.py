# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 17:37:01 2018

@author: wangshaoxin
"""
import os

import pandas as pd
import datetime
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

style.use('ggplot')    
plt.rcParams['font.sans-serif'] = ['SimHei'] 

plt.rcParams['axes.unicode_minus'] = False
os.chdir("D://机器学习课程//代码//Python时间序列")
stockFile = 'data/T10yr.csv'
stock = pd.read_csv(stockFile, index_col=0, parse_dates=[0])
stock.head(10)


stock_week = stock['Close'].resample('W-MON').mean()
stock_train = stock_week['2000':'2015']

model = ARIMA(stock_train, order=(1, 1, 1),freq='W-MON')
result = model.fit()

pred = result.predict('20000724', '20160701',dynamic=True, typ='levels')

plt.figure(figsize=(6, 6))
plt.xticks(rotation=45)
plt.plot(pred)
plt.plot(stock_train)