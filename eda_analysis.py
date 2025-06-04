import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 사용할 한글 폰트 설정 (Apple 기본 'AppleGothic' 사용)
plt.rcParams['font.family'] = 'AppleGothic'

# 2. 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('sales_data.csv')

## 수치형 상관관계 분석
numerical_features = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 
                      'Discount', 'Competitor Pricing', 'Demand']

corr = df[numerical_features].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("📊 수치형 변수 간 상관관계 히트맵")
plt.tight_layout()
plt.show()

## 범주형 변수 vs 평균 수요
categorical_groups = ['Category', 'Region', 'Weather Condition', 
                      'Seasonality', 'Promotion', 'Epidemic']

fig, axes = plt.subplots(3, 2, figsize=(15, 14))
axes = axes.flatten()

for i, var in enumerate(categorical_groups):
    sns.barplot(data=df, x=var, y='Demand', estimator=np.mean, ci=None, ax=axes[i])
    axes[i].set_title(f"{var}별 평균 수요")
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

## 시계열 분석
### 월별
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Month', y='Demand', estimator=np.mean, ci=None)
plt.title("월별 평균 수요")
plt.xlabel("월")
plt.ylabel("평균 수요")
plt.show()

### 요일별
df['Date'] = pd.to_datetime(df['Date'])

df['Weekday'] = df['Date'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Weekday', y='Demand', estimator=np.mean, ci=None, order=weekday_order)
plt.title("요일별 평균 수요")
plt.xlabel("요일")
plt.ylabel("평균 수요")
plt.xticks(rotation=45)
plt.show()

### daily
daily_demand = df.groupby('Date')['Demand'].sum().reset_index()

plt.figure(figsize=(14, 6))
sns.lineplot(data=daily_demand, x='Date', y='Demand')
plt.title("일별 전체 수요 추이")
plt.xlabel("날짜")
plt.ylabel("수요량")
plt.show()


# 가격/할인 vs 수요
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='Price', y='Demand', alpha=0.3)
plt.title("💰 가격 vs 수요")

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='Discount', y='Demand', alpha=0.3)
plt.title("🎯 할인율 vs 수요")

plt.tight_layout()
plt.show()

# 재고 수준 vs 수요
inventory_demand = df.pivot_table(index='Inventory Level', 
                                   values='Demand', aggfunc='mean').reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=inventory_demand, x='Inventory Level', y='Demand')
plt.title("📦 재고 수준 vs 평균 수요")
plt.xlabel("Inventory Level")
plt.ylabel("Demand")
plt.show()


# 이상치 탐지
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.boxplot(data=df, y='Demand', ax=axes[0])
axes[0].set_title('📦 Demand (수요) 이상치 탐지')

sns.boxplot(data=df, y='Units Sold', ax=axes[1])
axes[1].set_title('🛒 Units Sold (판매량) 이상치 탐지')

sns.boxplot(data=df, y='Inventory Level', ax=axes[2])
axes[2].set_title('🏬 Inventory Level (재고) 이상치 탐지')

plt.tight_layout()
plt.show()

# 수요와 판매량 차이 분석
## 수요와 판매량 차이 계산
df['Demand_Gap'] = df['Demand'] - df['Units Sold']

## 간단 통계 요약
print(df['Demand_Gap'].describe())
## 히스토그램으로 분포 확인
plt.figure(figsize=(10, 6))
sns.histplot(df['Demand_Gap'], bins=50, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.title('💡 수요와 판매량 차이 분포 (Demand - Units Sold)')
plt.xlabel('Demand Gap')
plt.ylabel('Frequency')
plt.show()
## 과잉 공급 (재고 낭비 위험)
print("▶️ 판매량 << 수요 (품절/기회 손실 사례)")
print(df[df['Demand_Gap'] > 50][['Date', 'Store ID', 'Product ID', 'Demand', 'Units Sold', 'Demand_Gap']].head())

## 과잉 판매 (예상보다 많이 팔림)
print("▶️ 판매량 > 수요 (예측 실패/초과 판매 사례)")
print(df[df['Demand_Gap'] < -50][['Date', 'Store ID', 'Product ID', 'Demand', 'Units Sold', 'Demand_Gap']].head())
