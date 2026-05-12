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

- 다중공선성 해결: 상관관계 분석을 통해 Annual_Income과 Monthly_Inhead_Salary 간의 극도로 높은 상관관계(0.99)를 확인하고, 중복 정보를 제거하여 모델의 과적합 위험을 방지했습니다. 

---
### EDA 및 해석
<img width="558" height="393" alt="image" src="https://github.com/user-attachments/assets/245d6596-d62d-41c9-bb79-98431e440094" />

#### 1. 신용 등급 클래스 분포 분석

전체 데이터에서 Standard(2) 등급의 비중이 가장 높으며, Poor(1), Good(0) 순으로 분포되어 있습니다.

클래스 간 불균형이 존재하므로, 학습 시 데이터 분할 과정에서 stratify 옵션을 적용하여 클래스 비율을 유지했습니다.

비즈니스 인사이트:

가장 적은 비중을 차지하는 Good 등급 고객을 확충하기 위한 우대 금리 상품 기획이 필요합니다.

<img width="1087" height="1007" alt="image" src="https://github.com/user-attachments/assets/cfae3c48-f911-42e2-9531-9d3f48e921e2" />

#### 2. 변수 간 상관관계 히트맵 분석



#### 고객 연령대 분포 분석
- 해석
1. 데이터 내 고객 연령층은 30대 초반에서 40대 초반에 가장 밀집되어 있는 '항아리형' 분포를 보입니다.
2. 20대 초반부터 급격히 유입이 증가하다가, 40대 중반을 기점으로 완만하게 감소하는 추세를 보입니다.

- 비즈니스 인사이트
1. 현재 우리 은행의 주력 고객층은 경제 활동이 가장 활발한 3040 세대임을 알 수 있습니다.
2. 20대 초반의 가파른 상승 곡선은 잠재 고객 확보 가능성을 시사하므로, 이들을 주거래 고객으로 안착시키기 위한 '생애 첫 금융 상품' 등의 타겟 마케팅이 유효할 것으로 판단됩니다.
---
### AutoML – Hyperparameter Tuning – Stacking Pipe – Shap value

##### 1. AutoML & Model Selection

<img width="980" height="559" alt="image" src="https://github.com/user-attachments/assets/8964e7be-adce-4f09-aa22-eb195f67087d" />

- 다양한 머신러닝 알고리즘의 베이스라인 성능을 비교하여 최적의 상위 모델을 선정하였습니다.

- 결과 : CatBoost, LightGBM과 GBC가 F1-Score 약 0.59 대를 기록하며 금융 데이터의 불균형 속에서도 안정적인 성능을 보임을 확인했습니다.



##### 2. Hyperparameter Tuning (Optuna)

- 선정된 상위 3개 모델(CatBoost, LGBM, GBC)에 대해 Optuna 프레임워크를 적용, 베이지안 최적화 기반의 하이퍼파라미터 튜닝을 수행했습니다.

- 목적 : 각 모델의 오버피팅을 방지하고 F1-Score를 극대화.

- 전략 : 10~50회 이상의 Trial을 통해 learning_rate, depth, iterations 등의 최적 조합을 도출했습니다.

<img width="286" height="81" alt="image" src="https://github.com/user-attachments/assets/b6da5b8d-70cf-4596-a8e1-cc85a404cb21" />

> 하이퍼파라미터 튜닝을 통해 개별 모델들이 안정적인 예측력을 확보하였습니다.


##### 3. Stacking Ensemble Pipeline

- 단일 모델의 한계를 극복하기 위해 StackingClassifier를 구축하여 예측력을 한 단계 높였습니다.

- Layer 1 (Base Estimators) : Optuna로 최적화된 CatBoost, LGBM, GBC

- Layer 2 (Final Estimator) : Logistic Regression을 메타 모델로 사용하여 각 모델의 예측 결과를 최종 통합.

- 결과: 단일 모델 대비 더욱 견고한 예측 성능 확보.

<img width="319" height="24" alt="image" src="https://github.com/user-attachments/assets/e5c75744-4191-4df3-881b-ecb4af91fb63" />

> 앙상블 기법을 통해 단일 모델 대비 더욱 견고하고 신뢰도 높은 최종 모델을 완성했습니다.


##### 4. Model Interpretation (SHAP Value)

- 모델의 판단 근거를 시각화하기 위해 SHAP 분석을 수행했습니다.

<img width="757" height="550" alt="image" src="https://github.com/user-attachments/assets/88b460c4-1181-47a1-9a13-b4c761f68c64" />


#### 핵심 변수 기여도: Age와 Products_Number가 이탈 예측에 가장 결정적인 역할을 수행.
---
### 인사이트 및 제언

##### 1. 연령대별 차별화된 리텐션 전략의 필요성

- 현상 : SHAP 분석 결과 이탈에 가장 지대한 영향을 미치는 요인은 나이로 나타났습니다. 특히 은퇴 전후 시점의 고객층에서 이탈 징후가 뚜렷합니다.

- 해석 : 이는 기존 서비스가 고령층 고객의 변화된 금융 니즈(자산 인출, 상속 등)를 충실히 반영하지 못하고 있음을 시사합니다.

- 제언 : 중장년층을 위한 전용 자산 관리 솔루션과 시니어 친화적 디지털 뱅킹 UI/UX 개선을 통해 고객 이탈을 선제적으로 방어해야 합니다.

##### 2. 양적 팽창에서 질적 관리로의 패러다임 전환

- 현상 : 상품 보유 수와 고객 충성도는 단순 비례 관계가 아님이 확인되었습니다. 오히려 다수의 상품을 보유한 고객이 관리 미흡 시 더 큰 기회비용을 느끼고 이탈할 위험이 존재합니다.

- 해석 : 무분별한 상품 가입 권유는 고객에게 관리 피로도를 유발하며, 실제 혜택 체감도를 떨어뜨릴 수 있습니다.

- 제언 : 상품 가입 '개수' 중심의 마케팅에서 벗어나, 주거래 고객이 확실한 우대 혜택을 체감할 수 있는 패키지 리워드 체계 및 주거래 고객 우대 제도의 실효성을 재점검해야 합니다.
