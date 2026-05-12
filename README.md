# DL competition

## DL 기반 신용 등급 예측
---
### 기간
- 2026년 5월 12일
---
### 기술스택 
- 언어(language) : Python
- 데이터 전처리 : Pandas, NumPy
- 데이터 스케일링 : Scikit-learn (StandardScaler)
- 딥러닝 모델링 : PyTorch, PyTorch-TabNet
- 데이터 시각화 : Matplotlib, Seaborn
---
### 데이터 출처
- 원본데이터 : https://www.kaggle.com/datasets/parisrohan/credit-score-classification
- 데이터 클리닝 방법 : https://www.kaggle.com/code/clkmuhammed/credit-score-classification-part-1-data-cleaning#Download-Link
---
### 데이터 전처리
- 비식별 변수 제거: 신용 등급과 직접적인 연관성이 낮은 ID, Customer_ID, Name, SSN 컬럼을 제거하여 모델의 노이즈를 최소화했습니다.

- 결측치 처리: 수치형 변수는 평균값(Mean)으로, 범주형 변수는 최빈값(Mode)으로 대체하여 데이터 손실 없이 학습 샘플을 유지했습니다.

- 인코딩 및 스케일링: 범주형 변수는 LabelEncoder를 통해 수치화하였으며, 수치형 변수는 StandardScaler를 적용하여 모든 피처가 동일한 스케일 범위 내에서 학습되도록 표준화했습니다.


<img src="https://github.com/user-attachments/assets/681c1351-bc54-46ee-8421-d7931ee94da1" width="500">


- 다중공선성 해결: 상관관계 분석을 통해 Annual_Income과 Monthly_Inhead_Salary 간의 극도로 높은 상관관계(0.99)를 확인하고, 중복 정보를 제거하여 모델의 과적합 위험을 방지했습니다


---
### EDA 및 해석

#### 1. 신용 등급 클래스 분포 분석

<img width="558" height="393" alt="image" src="https://github.com/user-attachments/assets/245d6596-d62d-41c9-bb79-98431e440094" />

해석: 

전체 데이터에서 Standard(2) 등급의 비중이 가장 높으며, Poor(1), Good(0) 순으로 분포되어 있습니다.

클래스 간 불균형이 존재하므로, 학습 시 데이터 분할 과정에서 stratify 옵션을 적용하여 클래스 비율을 유지했습니다.

#### 2. 변수 간 상관관계 히트맵 분석

<img width="1087" height="1007" alt="image" src="https://github.com/user-attachments/assets/cfae3c48-f911-42e2-9531-9d3f48e921e2" />

해석: 

소득 수준, 부채 규모, 연체 여부가 신용 등급 결정에 가장 핵심적인 영향을 미치는 변수임을 시사합니다.

특히 부채 관련 변수들이 신용 점수와 밀접하게 연동되어 있음을 확인했습니다.

---
### TabNet 모델 구축 및 하이퍼파라미터 최적화

##### 1. 모델 선정: TabNetClassifier

- 선정 이유: 정형데티어 학습에 최적화된 딥러닝 구조로, 신경망의 유연함과 의사결정나무의 변수 선택 능력을 동시에 갖추고 있어 선정하였습니다.

##### 2. Hyperparameter Tuning

- Learning Rate (0.02): TabNet 특성에 맞춰 학습률을 상향 조정하여 변수 선택의 과감성을 높였습니다.

- Batch Size (1024): Ghost Batch Normalization 효과를 극대화하기 위해 대규모 배치를 적용했습니다.

- Mask Type ('entmax'): 중요도가 낮은 피처를 더 확실하게 억제하여 예측의 선명도를 높였습니다.

- Lambda Sparse (1e-4): 희소성 규제를 통해 핵심 변수에만 집중하도록 유도했습니다.



##### 3. 학습 결과 및 성능 평가

- 결과: Validation Accuracy 79.35%

<img width="445" height="209" alt="image" src="https://github.com/user-attachments/assets/f67be8f9-3f35-4c99-a9f3-1985236c87ea" />

- 평가: 하이퍼파라미터 최적화와 과적합 방지 전략(Early Stopping, Weight Decay)을 통해 단일 딥러닝 모델로 매우 견고한 성능을 확보했습니다.
  
---
### 개선사항

- Feature Selection을 적용하여 중요한 변수만 선택하였습니다.
- TabNet 모델을 적용하여 tabular 데이터 학습 성능을 향상시켰습니다.
- Attention 기반 구조를 통해 중요한 feature를 선택적으로 학습하였습니다.
- Early Stopping을 적용하여 과적합을 방지하였습니다.
- Heatmap과 Countplot을 통해 데이터 분포와 변수 관계를 시각화하였습니다.
- Stratify를 적용하여 클래스 비율 불균형 문제를 완화하였습니다.
- 상관관계 분석을 통해 Annual_Income과 Monthly_Inhand_Salary 간의 매우 높은 상관성(0.99)을 확인하고, 과적합 방지를 위해 중복 변수를 제거하여 모델의 안정성을 높였습니다.
