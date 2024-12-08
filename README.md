
# 딥러닝을 활용한 SDN 패킷 처리 성능 최적화

---


---

## SDN이란? (What is SDN?)
- **정의**: 소프트웨어를 통해 네트워크를 제어하고 관리하는 기술.
- **특징**: 네트워크의 **제어 평면(Control Plane)**과 **데이터 평면(Data Plane)**을 분리.

### SDN과 기존 네트워크의 비교
- **기존 네트워크**: 제어 평면이 각 네트워크 장비에 분산.
- **SDN**: 통합된 제어 평면 제공.

---

## 개발 환경 (Development Environment)
- **네트워크 시뮬레이터**: Mininet
- **SDN 컨트롤러**: RYU
- **딥러닝 라이브러리**: PyTorch


- <img width="869" alt="image" src="https://github.com/user-attachments/assets/a040f7a1-310d-49f5-b409-6e2bb6bffae8">


---

## 결과 (Results)

### Base Line
- **조건**: `simple_switch`를 기준으로 벤치마크 실행.
- **동작 원리**:
  - 소스 MAC 주소를 테이블에 저장.
  - 목적지 MAC 주소가 학습 테이블에 있으면 해당 포트로 전달.
  - 학습되지 않은 경우 패킷 플러딩.
- **성과**:
  - 평균 Throughput: **3312.12 responses/s**
  - 표준 편차: **198.10 (작을수록 안정적)**

    <img width="639" alt="image" src="https://github.com/user-attachments/assets/c819fa3d-eae5-4b72-9b8d-64688f8e49e5">


### 딥러닝 적용 후
- **조건**: 동일한 테스트 환경 (128개 스위치, 3시간 실행).
  - GPU 포팅, 타일링, 캐싱 등 최적화 기법 사용.
- **성과**:
  - 평균 Throughput: **8949.96 responses/s**
  - 표준 편차: **249.12 (작을수록 안정적)**

  - <img width="563" alt="image" src="https://github.com/user-attachments/assets/a47ab268-9dba-48a7-8553-540dd0b44579">


    

---

## 시각화
- 결과 데이터는 그래프로 시각화하여 성능 개선 확인 가능 (e.g., Cbench 사용).

---

## 결론
- SDN의 딥러닝 최적화는 기존 방식 대비 **Throughput**이 크게 향상됨.
- **네트워크 관리의 지능화 및 성능 최적화**에 있어 딥러닝 기술의 가능성을 입증.
```
