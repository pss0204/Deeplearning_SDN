import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# 로그 파일 읽기
with open('log.txt', 'r') as file:
    lines = file.readlines()

# 데이터 추출 함수 수정
def extract_data(line):
    timestamp_match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3})', line)
    flows_match = re.findall(r'\d+(?=\s+switches: flows\/sec:)(.*?)total = ([\d\.]+)', line)
    
    if timestamp_match and flows_match:
        timestamp = timestamp_match.group(1)
        flows_str = flows_match[0][0]
        total = float(flows_match[0][1])
        
        # flows/sec 값들 추출
        flows = [int(x) for x in re.findall(r'\d+', flows_str)]
        
        return {
            'Timestamp': timestamp,
            'Total_flows_per_ms': total,
            'Min_flow': min(flows),
            'Max_flow': max(flows),
            'Avg_flow': np.mean(flows),
            'Std_flow': np.std(flows),
            'Flow_values': flows
        }
    return None

# 데이터 추출
data = []
for line in lines:
    result = extract_data(line)
    if result:
        data.append(result)

df = pd.DataFrame(data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')

# 다양한 시각화
plt.figure(figsize=(15, 5))

# 1. 전체 성능 추이
sns.lineplot(data=df, x='Timestamp', y='Total_flows_per_ms', marker='o')
plt.title('Total Flows/ms Over Time')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('flows_analysis.png')

# 통계 분석 결과 출력
print("\n=== 성능 통계 분석 ===")
print(f"평균 처리량: {df['Total_flows_per_ms'].mean():.2f} flows/ms")
print(f"최대 처리량: {df['Total_flows_per_ms'].max():.2f} flows/ms")
print(f"최소 처리량: {df['Total_flows_per_ms'].min():.2f} flows/ms")
print(f"처리량 표준편차: {df['Total_flows_per_ms'].std():.2f}")

# 안정성 분석 (변동계수)
cv = df['Total_flows_per_ms'].std() / df['Total_flows_per_ms'].mean()
print(f"\n변동계수 (CV): {cv:.4f}")

# 시계열 특성 분석
autocorr = df['Total_flows_per_ms'].autocorr()
print(f"자기상관계수: {autocorr:.4f}")