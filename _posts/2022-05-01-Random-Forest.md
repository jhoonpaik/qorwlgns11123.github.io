---
title: Random Forest
description: 랜덤 포레스트는 결정트리 기반으로 알고리즘으로 앙상블 기법 중 빠른 속도와 높은 예측성능을 보이는 기법중 하나이다.
tags:
- Machine learning
- Bagging
- classification
---

# 앙상블 학습

랜덤포레스트를 소개하기 앞서 앙상블의 개념에 대해 살펴보자  
앙상블 학습(Ensemble Learning)은 여러 개의 분류기(Classifier)를 생성하여 그 예측의 결합을 통해 정확한 최종예측을 도출하는 기법이다.

앙상블의 유형에는 보팅(voting), 배깅(Bagging), 부스팅(Boosting) 세 가지가 대표적이며, 이외에도 스태깅 등의 다양한 앙상블 방법이 있다.
보팅과 배깅은 여러 개의 분류기를 투표를 통해 최종 예측 결과를 결정하는 방식이지만
보팅의 경우 서로 다른 알고리즘을 가진 분류기를 결합, 배깅의 경우 모두 같은 알고리즘 분류기이지만 데이터 샘플링이 서로 다르게 가진 상태로 학습을 수행하는 방식이다. 그리고 이 배깅의 대표적인 방식이 랜덤포레스트이다.

![](https://velog.velcdn.com/images/adastra/post/ee4c6d7d-e046-476e-9253-c588dff94f0d/image.png)

부스팅은 여러 개의 분류기로 순차 학습을 진행하고 앞서 학습한 분류기가 예측이 틀린 데이터에 대해선 다음에 올바르게 예측하도록 분류기에 가중치를 부스팅을 부여하면서 학습과 예측을 진행하는 방식으로 이뤄진다.  
계속해서 가중치를 부여하면서 학습을 진행하는 방식이기 때문에 부스팅이라고 불린다.(boosting 단어의미: 증가, 북돋우다.)

![](https://velog.velcdn.com/images/adastra/post/8b5ea5ee-aa1a-4255-b46e-cdd9c7d9a77f/image.png)
(출처: https://hyunlee103.tistory.com/25)


## Voting: Hard Voting vs Soft Voting
하드 보팅은 예측 결과값중 다수의 분류기가 결정한 예측값을 선정하므로 다수결 방식의 개념이다.
소프트 보팅의 경우 분류기들의 레이블 값 결정 확률을 모두 더하고 평균내서 확률이 가장 높은 레이블 값으로 선정하는 방식이다.
일반적으로 보팅은 소프트 보팅이 적용된다.

랜덤 포레스트는 결정트리 기반으로 알고리즘으로 앙상블 기법 중 빠른 속도와 높은 예측성능을 보이는 기법중 하나이다.

랜덤 포레스트는 여러 개의 결정트리 분류기를 배깅 방식으로 각자 데이터를 샘플링해 개별 학습한 후 최종적으로 모든 분류기를 보팅을 통해 예측 결정하게 되는 방식이다.

랜덤포레스트의 개별 트리가 학습하는 데이터셋은 전체 데이터셋에서 일부가 중첩되게 샘플링된 데이터셋이다. 이렇게 중첩되게 샘플링된 분할 방식을 Bootstrapping 분할 방식이다.(Bagging이 bootstrap aggregating의 줄임말이다.)

원본데이터셋 건수가 10개인 학습 데이터셋을 3개의 결정트리 기반의 랜덤포레스트를 학습하기 위해 n_estimator=3으로 하이퍼 파라미터를 부여하면 아래와 깉이 데이트 서브셋이 생성된다.

![](https://velog.velcdn.com/images/adastra/post/636da942-7df8-4c8c-b0ee-1daab51f0bf7/image.png)

사이킷런은 RandomForestClassifier 클래스를 통해 모델을 지원한다.  
사용자 행동 인식 데이터셋을 기반으로 랜덤포레스트 실습을 진행해보자

![](https://velog.velcdn.com/images/adastra/post/e0d1e8a6-c727-4c18-9d26-3c9181c58b5e/image.png)
https://archive.ics.uci.edu/ml/datasets/Human+Activity+recognition+using+smartphones

실습데이터는 UCI 머신러닝 Repository에서 제공하는 사용자 행동 인식 데이터셋을 활용한다.  
해당 데이터는 30명에게 스마트폰 센서를 장착시키고 사람 동작과 관련된 여러 가지 피처를 수집한 데이터다.  
해당 데이터셋을 전처리하여 학습/테스트 데이터셋으로 반환하는 함수를 생성하였다.

```python
## 원본 데이터에 중복된 Feature 명으로 인하여 신규 버전의 Pandas에서 Duplicate name 에러를 발생.
## 중복 feature명에 대해서 원본 feature 명에 '_1(또는2)'를 추가로 부여하는 함수인 get_new_feature_name_df() 생성

def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(),
                                  columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1]) 
                                                                                         if x[1] >0 else x[0] ,  axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df
```

```python
## 데이터 수정 및 train/test 데이터셋 생성 함수

import pandas as pd

def get_human_dataset( ):
    
    # 각 데이터 파일들은 공백으로 분리되어 있으므로 read_csv에서 공백 문자를 sep으로 할당.
    feature_name_df = pd.read_csv('C:/Users/JIHOON PARK/kfq_study/machine_learning/dataset/human_activity/features.txt',sep='\s+',
                        header=None,names=['column_index','column_name'])
    
    # 중복된 피처명을 수정하는 get_new_feature_name_df()를 이용, 신규 피처명 DataFrame생성. 
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    
    # DataFrame에 피처명을 컬럼으로 부여하기 위해 리스트 객체로 다시 변환
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()
    
    # 학습 피처 데이터 셋과 테스트 피처 데이터을 DataFrame으로 로딩. 컬럼명은 feature_name 적용
    X_train = pd.read_csv('C:/Users/JIHOON PARK/kfq_study/machine_learning/dataset/human_activity/train/X_train.txt',sep='\s+', names=feature_name )
    X_test = pd.read_csv('C:/Users/JIHOON PARK/kfq_study/machine_learning/dataset/human_activity/test/X_test.txt',sep='\s+', names=feature_name)
    
    # 학습 레이블과 테스트 레이블 데이터을 DataFrame으로 로딩하고 컬럼명은 action으로 부여
    y_train = pd.read_csv('C:/Users/JIHOON PARK/kfq_study/machine_learning/dataset/human_activity/train/y_train.txt',sep='\s+',header=None,names=['action'])
    y_test = pd.read_csv('C:/Users/JIHOON PARK/kfq_study/machine_learning/dataset/human_activity/test/y_test.txt',sep='\s+',header=None,names=['action'])
    
    # 로드된 학습/테스트용 DataFrame을 모두 반환 
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_human_dataset()
```

먼저 학습 데이터셋을 살펴보자
```python
print('## 학습 피처 데이터셋 info()')
print(X_train.info())
```
```
## 학습 피처 데이터셋 info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7352 entries, 0 to 7351
Columns: 561 entries, tBodyAcc-mean()-X to angle(Z,gravityMean)
dtypes: float64(561)
memory usage: 31.5 MB
None
```

학습 데이터셋은 7352개의 레코드, 561개의 피처를 가지며 피처가 전부 float형이므로 인코딩을 수행할 필요는 없다.

```python
print(y_train['action'].value_counts())
```

```
6    1407
5    1374
4    1286
1    1226
2    1073
3     986
Name: action, dtype: int64
```

위와 같이 레이블 값은 총 6개의 값으로 비교적 고르게 분포된 것을 확인할 수 있다.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# get_human_dataset()함수를 통해 train/test용 DF 반환
X_train, X_test, y_train, y_test = get_human_dataset()

# RandomForest 학습 및 테스트 셋을 통해 예측 성능 평가

rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('randomforest 정확도: {0:4f}'.format(accuracy))
```

```
randomforest 정확도: 0.925348
```

## 하이퍼 파라미터

트리기반의 앙상블 알고리즘의 경우 파라미터가 워낙 많고 오래 걸리며 튜닝 후 예측성능이 크게 상승하지 않는 단점을 갖고 있다.
랜덤포레스트의 경우 아래와 같이 결정트리와 같은 파라미터가 대부분이다.

- n_estimators: 랜덤포레스트에서 결정트리 갯수 지정(디폴트: 10). 늘릴수록 좋은 것은 아니며 늘릴수록 학습 수행시간이 오래걸림
- max_feature: 최적의 분할을 위해 고려할 최대 피처 개수(전체 피처 중 √(피처개수) 만큼 선정. ex)전체 피처 16개면 4개 참조)
- max_depth
- min_samples_leaf

GridSearchCV를 통해 랜덤 포레스트 하이퍼 파라미터를 튜닝해보자

```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators':[100],
    'max_depth' : [6, 8, 10, 12], 
    'min_samples_leaf' : [8, 12, 18 ],
    'min_samples_split' : [8, 16, 20]
}
# RandomForestClassifier 객체 생성 후 GridSearchCV 수행
rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf_clf , param_grid=params , cv=2, n_jobs=-1 )
grid_cv.fit(X_train , y_train)

print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))
```

```
최적 하이퍼 파라미터:
 {'max_depth': 10, 'min_samples_leaf': 8, 'min_samples_split': 8, 'n_estimators': 100}
최고 예측 정확도: 0.9180
```

{'max_depth': 10, 'min_samples_leaf': 8, 'min_samples_split': 8, 'n_estimators': 100}일 때 약 91.80%의 평균정확도가 측정되었다.
이제 n_estimators를 300으로 증가시키고 최적화 하이퍼 파라미터로 다시 학습시킨 뒤 테스트 데이터셋으로 예측성능을 평가해보자

```python
rf_clf1 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=8, \
                                 min_samples_split=8, random_state=0)
rf_clf1.fit(X_train , y_train)
pred = rf_clf1.predict(X_test)
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test , pred)))
```

```
예측 정확도: 0.9165
```

별도의 테스트데이터셋으로 수행한 예측 정확도는 91.65%이다.
이제 feature_importance_ 속성을 통해 알고리즘이 선택한 피처 중요도를 시각화하여 확인해보자

```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ftr_importances_values = rf_clf1.feature_importances_
ftr_importances = pd.Series(ftr_importances_values,index=X_train.columns  )
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20 , y = ftr_top20.index)
plt.show()
```

![](https://velog.velcdn.com/images/adastra/post/25959396-41e3-4313-8520-35ccde686d90/image.png)

> 참고자료: 파이썬 머신러닝 완벽 가이드 - 권철민