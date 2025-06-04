import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 데이터 로딩
df = pd.read_csv("sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# ✅ 날짜 기반 파생변수
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.day_name()
df['Is_Weekend'] = df['Weekday'].isin(['Saturday', 'Sunday']).astype(int)

# ✅ 수요 vs 판매량 차이 변수
df['Demand_Gap'] = df['Demand'] - df['Units Sold']

# ✅ 시차 변수 (제품별 전일 수요)
df = df.sort_values(by=['Product ID', 'Date'])
df['Prev_Day_Demand'] = df.groupby('Product ID')['Demand'].shift(1)
df['Rolling_7D_Demand'] = df.groupby('Product ID')['Demand'].transform(lambda x: x.rolling(7, min_periods=1).mean())

# ✅ 범주형 변수 인코딩 (Label Encoding 예시)
cat_cols = ['Region', 'Category', 'Weather Condition', 'Seasonality', 'Weekday']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ✅ 스케일링 (가격 관련 변수)
scaler = StandardScaler()
scaled_cols = ['Price', 'Competitor Pricing', 'Inventory Level']
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# ✅ 결측치 제거 (시차 변수 때문에 생긴 NaN)
df.dropna(inplace=True)

# ✅ 이상치 제거 함수 (IQR 기준)
def remove_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[col] >= lower) & (data[col] <= upper)]

# ✅ 주요 수치형 변수 이상치 제거
for col in ['Demand', 'Units Sold', 'Inventory Level']:
    df = remove_outliers_iqr(df, col)

# ✅ 이상치 제거 후 남은 관측치 수 확인
print(f"📦 이상치 제거 후 데이터 크기: {df.shape[0]} rows, {df.shape[1]} columns")

