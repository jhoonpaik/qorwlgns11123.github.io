---
title: 2022-05-13-Boosting의-향연-GBM, XGBoost, LightGBM
description: ㅇㅇ
tags:
- Machine learning
- Boosting
- classification
---

# Boosting

부스팅 알고리즘은 순차적 학습, 예측을 진행하며 error 데이터 가중치 부여를 통해 개선하는 학습방식이다.
부스팅의 대표적 방식은 AdaBoost(Adaptive boosting)와 그래디언트 부스트가 있다.

![](https://velog.velcdn.com/images/adastra/post/016606f5-6bde-4445-8c06-f8cae0555d01/image.png)

step1. 약한 분류기 가 분류 기준 +와 -로 분류  
step2. 오류 데이터에 가중치 부여  
step3. step1 반복  
step4. step2 반복  
step5. 약한 학습기가 순차적으로 오류 값에 대해 가중치를 부여한 예측 결정 기준을 모두 결합해 예측 수행

마지막 결과는 약한 학습기를 모두 결합한 예측 결과로 개별의 약한 학습기보다 훨씬 정확도가 높음을 알 수 있다.
에이다 부스트는 위와 같은 방식으로 진행된다.

# GBM

GBM(Gradient Boost Machine)도 에이다 부스트와 같은 방식과 이뤄지지만 추가적으로 **GBM은 가중치 업데이트를 경사하강법으로 이용**한다는 차이가 있다.

분류의 실제 결과를 y, 피처가 x1, x2, ... 피처에 기반한 예측함수를 F(x)함수이면
오류식은 h(x)=y-F(x)가 된다. 이 오류식을 최소화하는 방향성을 갖고 가중치를 반복적으로 업데이터 하는 것이 경사하강법(Gradient Descent)이다.
![](https://velog.velcdn.com/images/adastra/post/3dac4823-3f0d-49fa-affa-e6116653de48/image.png)


GBM은 사이킷런에서 GradientBoostingClassifier 클래스를 통해 활용할 수 있다.  
사용자 행동 데이터셋을 활용하여 GBM을 통한 예측 분류를 진행해보자

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

import pandas as pd


import time
import warnings
warnings.filterwarnings('ignore')

def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(),
                                  columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1]) 
                                                                                         if x[1] >0 else x[0] ,  axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df

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

gb_clf = GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)

print('GBM 정확도: {0:.4f}'.format(gb_accuracy))
```
```
GBM 정확도: 0.9389
```

GBM 수행결과, 정확도는 약 93.89%로 도출되었다. 
GBM의 경우 랜덤포레스트보다 일반적으로 예측 성능이 뛰어나지만 매우 느린 수행 시간 문제를 단점으로 갖고 있다.

## GBM 하이퍼 파라미터 및 튜닝

기존의 트리기반 파라미터는 제외하고 살펴보자

- loss: 경사 하강법에서 상요할 비용함수 지정(기본값: deviance)
- learning_rate: 학습 진행시 적용하는 학습률. 약한 분류기가 순차적으로 오류값을 보정해 나가는데 적용하는 계수이며 0~1사이의 값을 지정할 수 있음(기본값: 0.1)
- n_estimator: week learner의 개수(기본값: 100)
- subsample: weak learner가 학습에 사용하는 데이터샘플링 비율(기본값: 1. 1은 전체 학습 데이터를 기반으로 학습을 함)

이제 GridSearchCV를 통해 살펴본 GBM 하이퍼 파라미터를 최적화해보자

```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators':[100, 500],
    'learning_rate':[0.05, 0.1]
}

grid_cv = GrideSearchCV(gb_clf, param_grid=params, cv=2, verbose=1)
grid_cv.fit(X_train, y_train)

print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도:{0:.4f}'.format(grid_cv.best_score_))
```
```
최적하이퍼 파라미터:
{'learning_rate': 0.05, 'n_estimators': 500}
최고 예측 정확도: 0.9010
```

최종적으로 도출된 최적 하이퍼 파라미터를 테스트 데이터셋에 적용하여 예측 정확도를 확인해보자

```python
gb_pred = grid_cv.best_estimator_.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print('GBM 정확도: {0:.4f}'.format(gb_accuracy))
```
```
GBM 정확도: 0.9410
```

# XGBOOST

# LightBGM