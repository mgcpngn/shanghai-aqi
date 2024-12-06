import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息

# 导入字体管理工具
from matplotlib.font_manager import FontProperties

# 设置中文字体
# 请确保系统中已安装支持中文的字体，例如SimHei
font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows系统路径，其他系统请修改为相应路径
font_prop = FontProperties(fname=font_path)

# 1. 读取数据
df = pd.read_excel('shanghai.xlsx')

# 2. 数据预处理
print("原始数据预览：")
print(df.head())

print("\n数据基本信息：")
print(df.info())

print("\n缺失值检查：")
print(df.isnull().sum())

# 处理缺失值（此处选择删除包含缺失值的行）
df.dropna(inplace=True)

# 将'月份'列转换为日期类型（修正unit参数为'D'）
df['月份'] = pd.to_datetime(df['月份'], unit='D', origin='1899-12-30')

# 提取年份和月份
df['年份'] = df['月份'].dt.year
df['月份_num'] = df['月份'].dt.month

print("\n预处理后数据预览：")
print(df.head())

# 3. 描述性统计分析
print("\n描述性统计：")
print(df.describe())

# 4. 可视化分析
# AQI时间趋势
plt.figure(figsize=(14,6))
plt.plot(df['月份'], df['AQI'], label='AQI', color='blue')
plt.title('上海市月度AQI变化趋势', fontproperties=font_prop, fontsize=16)
plt.xlabel('时间', fontproperties=font_prop, fontsize=12)
plt.ylabel('AQI', fontproperties=font_prop, fontsize=12)
plt.legend(prop=font_prop)
plt.grid(True)
plt.show()

# 各污染物时间趋势
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
plt.figure(figsize=(14,8))
for pollutant in pollutants:
    plt.plot(df['月份'], df[pollutant], label=pollutant)
plt.title('上海市月度各污染物变化趋势', fontproperties=font_prop, fontsize=16)
plt.xlabel('时间', fontproperties=font_prop, fontsize=12)
plt.ylabel('浓度', fontproperties=font_prop, fontsize=12)
plt.legend(prop=font_prop)
plt.grid(True)
plt.show()

# 质量等级分布
plt.figure(figsize=(8,6))
sns.countplot(x='质量等级', data=df, order=df['质量等级'].value_counts().index)
plt.title('上海市空气质量等级分布', fontproperties=font_prop, fontsize=16)
plt.xlabel('质量等级', fontproperties=font_prop, fontsize=12)
plt.ylabel('频次', fontproperties=font_prop, fontsize=12)
plt.xticks(rotation=45)
plt.show()

# 季节性分析
df['季度'] = df['月份'].dt.quarter
plt.figure(figsize=(10,6))
sns.boxplot(x='季度', y='AQI', data=df)
plt.title('上海市季度AQI分布', fontproperties=font_prop, fontsize=16)
plt.xlabel('季度', fontproperties=font_prop, fontsize=12)
plt.ylabel('AQI', fontproperties=font_prop, fontsize=12)
plt.show()

# 5. 相关性分析
corr_matrix = df[['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].corr()
print("\n相关系数矩阵：")
print(corr_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('空气质量各指标相关性热力图', fontproperties=font_prop, fontsize=16)
plt.show()

# 6. 多元线性回归模型
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]
y = df['AQI']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
lr = LinearRegression()

# 训练模型
lr.fit(X_train, y_train)

# 进行预测
y_pred = lr.predict(X_test)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'\n线性回归模型均方误差: {mse:.2f}')
print(f'线性回归模型决定系数 R²: {r2:.2f}')

# 绘制实际值与预测值对比图
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.7, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
plt.title('实际AQI值与预测AQI值对比', fontproperties=font_prop, fontsize=16)
plt.xlabel('实际AQI值', fontproperties=font_prop, fontsize=12)
plt.ylabel('预测AQI值', fontproperties=font_prop, fontsize=12)
plt.grid(True)
plt.show()

# 7. 时间序列预测模型（SARIMA）
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 按月份排序并设置为索引
df_sorted = df.sort_values('月份')
df_sorted.set_index('月份', inplace=True)

# 构建并训练SARIMA模型
model = SARIMAX(df_sorted['AQI'], order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit()
print("\nSARIMA模型摘要：")
print(results.summary())

# 预测未来12个月AQI
pred = results.get_forecast(steps=12)
pred_ci = pred.conf_int()

plt.figure(figsize=(14,6))
plt.plot(df_sorted['AQI'], label='历史AQI', color='blue')
plt.plot(pred.predicted_mean, label='预测AQI', color='red')
plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('上海市AQI时间序列预测', fontproperties=font_prop, fontsize=16)
plt.xlabel('时间', fontproperties=font_prop, fontsize=12)
plt.ylabel('AQI', fontproperties=font_prop, fontsize=12)
plt.legend(prop=font_prop)
plt.grid(True)
plt.show()