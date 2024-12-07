import re
import statistics

flows = []

with open('log.txt', 'r') as file:
    for line in file:
        if 'flows/sec:' in line:
            # 흐름 수치 추출
            numbers = re.findall(r'\d+', line.split('flows/sec:')[1])
            flows.extend(map(int, numbers))

# 평균 및 표준편차 계산
average = statistics.mean(flows)
stdev = statistics.stdev(flows)

print(f"평균 처리량: {average:.2f} responses/s")
print(f"표준편차: {stdev:.2f} responses/s")