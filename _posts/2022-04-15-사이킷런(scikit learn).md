---
title: scikit learn
description: 사이킷런(scikit learn)은 파이썬 머신러닝 라이브러리중 가장 많이 사용되는 라이브러리다.
tags:
- Machine learning
- KFold
- GridSearchCV
---

# scikit learn 설치

사이킷런(scikit learn)은 파이썬 머신러닝 라이브러리중 가장 많이 사용되는 라이브러리다.

사이킷런 설치는 아래와 같이 설치를 진행하며 numpy, scipy와 같은 다양한 라이브러리를 동시에 설치한다.
> conda install scikit-learn  
> pip install scikit-learn

우선 사이킷런을 import 해서 버전을 확인해보자

```python
import sklearn
print(sklearn.__version__)
```

```
1.0.2
```


# 사이킷런에 내장된 붓꽃데이터셋 확인

붓꽃 데이터의 피처는 Sepal length, Sepal width, Petal length, Petal width로 구성되어 있다.  
붓꽃데이터의 레이블은 Setosa, Vesicolor, Virginica로 구성되어있다.

먼저 sklearn 데이터셋에서 iris 데이터를 불러와보자

```python
from sklearn.datasets import load_iris
iris = load_iris()
type(iris)
```

```
sklearn.utils.Bunch
```

## Bunch

데이터셋의 type을 확인한 결과 sklearn.utils.Bunch라는 결과가 출력되었다.  
Bunch는 sklearn에서 자체적으로 만든 Bunch라는 객체 type이며 파이썬 기반의 자료구조이다.  
Bunch는 {data : ndarray(2차원 데이터셋())}와 같이 딕셔너리 형태를 반환한다.

딕셔너리 형태이므로 iris 데이터셋의 key값을 아래와 같이 확인할 수 있다.

```python
keys = iris.keys()
print('붓꽃 데이터셋 keys:', keys)
```

```
붓꽃 데이터셋 keys: dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
```

## 데이터셋의 피처, 피처명과 레이블 확인

```python
iris_features = iris.data # 2차원 150, 4
iris_features
```
![](https://velog.velcdn.com/images/adastra/post/80fdc17a-93d8-469f-9c7f-6b1b9f7d2ad5/image.png)

```python
iris.feature_names
```
![](https://velog.velcdn.com/images/adastra/post/7fc34aa5-9ee0-4db6-85e1-403f24b6e344/image.png)

```python
iris_label = iris.target # ndarray 1차원 150, 4
iris_label
```
![](https://velog.velcdn.com/images/adastra/post/2ad033f3-acff-4465-ade1-63d02acd2cf2/image.png)

## DataFrame 생성

```python
import pandas as pd

iris_df = pd.DataFrame(data=iris_features, columns = iris.feature_names)
# iris_df
iris_df['label'] = iris.target
iris_df
```
![](https://velog.velcdn.com/images/adastra/post/735d7cda-3744-4c56-b78e-66d9f17d388f/image.png)

위와 같이 iris 데이터프레임이 완성되었다.

## 사이킷런을 통한 붓꽃 데이터 실습

사이킷런을 제대로 이해하기 위한 실습으로 대중적인 데이터셋인 붓꽃 데이터셋을 통해 붓꽃 품종 분류(classification)을 진행해보자 
분류는 DecisionTree로 진행한다.  
*출력하고자 하는 값이 수치형이면 회귀, 범주형이면 분류를 진행

```python
# iris 데이터셋 로딩
from sklearn.datasets import load_iris

# 의사결정트리 클래스
from sklearn.tree import DecisionTreeClassifier

# 정확도
from sklearn.metrics import accuracy_score

# iris 데이터셋 생성
iris = load_iris()

# 모델 생성
dt_clf = DecisionTreeClassifier()

train_data = iris.data
train_label = iris.target

dt_clf.fit(train_data, train_label)

# 학습 데이터 셋으로 예측 수행
pred = dt_clf.predict(train_data)
print('예측 정확도:',accuracy_score(train_label,pred))
```
```
예측 정확도: 1.0
```
위와 같이 예측 정확도가 100%이 나온 이유는 이미 학습한 학습 데이터셋을 기반으로 예측했기 때문이다.
즉, 제대로 예측을 수행하기 위해서는 테스트셋을 통해 예측을 진행해야 한다.

# 사이킷런 model selection 소개
## train_test_split
sklearn의 train_test_split을 통해 원본 데이터 셋을 학습, 테스트 데이터셋으로 분리할 수 있다.

데이터를 분리하는 데 있어서 아래와 같은 파라미터를 설정할 수 있다.

- test_size: 테스트셋 비율 설정 (default: 25%). 일반적으로 0.3으로 분리한다.

- shuffle: 데이터를 분리하기 전 데이터를 미리 섞을지 결정 (default: True)

- random_state: 호출시 동일한 학습/테스트 셋을 생성하기 위해 주어지는 난수값. 호출시 무작위로 분리하므로 동일한 데이터셋을 위해 설정한다.

그럼 이제 붓꽃 데이터셋을 train_test_split()을 통해 분리해보자

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# train_test_split import
from sklearn.model_selection import train_test_split

# 모델 클래스 생성
dt_clf = DecisionTreeClassifier()
iris_data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size = 0.3, random_state = 121)
```

X_train(train 속성), X_test(test 속성), y_train(X_train에 대응하는 y_label값), y_test(X_test에 대응하는 y_label 값)

iris_data.data: 데이터  
iris_data.target: 타깃  
test_size = 0.3: train 70%, test 30%

분리한 데이터를 기반으로 모델을 학습하고 예측 정확도를 측정해보자

```python
dt_clf.fit(X_train, y_train) # 훈련데이터로 학습
pred = dt_clf.predict(X_test) # test 데이터로 예측
print('예측정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))
```

```
예측정확도: 0.9556
```

붓꽃 데이터 150개를 통해 학습하고 테스트 데이터 45개로 예측한 결과 정확도는 약 95%가 나온 것을 확인할 수 있다.

하지만 단순히 정확도만으로 모델의 예측성능을 평가하기엔 적절하지 않다.  
학습된 모델에 대해서도 다양한 데이터를 기반으로 예측 성능을 평가하는 것도 매우 중요하다.


## 교차 검증

별도의 테스트셋을 통해 예측 성능을 평가하더라도 과적합이 발생할 수 있다.
과적합(Overfitting)은 모델이 학습 데이터에 과도하게 최적화됨으로 실제 예측을 다른 데이터로 수행할 경우 예측성능이 과도하게 떨어지는 것을 의미한다.
고정된 학습, 테스트 데이터로만 평가하게 되면 테스트 데이터에만 편향되게 모델이 성능을 높이는 경향이 생겨 과적합 문제가 발생할 수 있다.
이러한 문제점을 개선하기 위해 교차검증을 이용해 다양한 학습과 평가를 수행한다.

교차검증은 별도의 여러 세트로 구성된 학습 데이터셋과 검증 데이터셋에서 학습과 평가를 수행한다.  
각 데이터셋에서 수행한 평가결과를 통해 모델 최적화를 손쉽게 할 수 있다.

### K 폴드 교차 검증

K 폴드 교차 검증은 가장 보편적으로 사용되는 교차 검증으로 k개의 데이터 폴드 세트를 만들어서 k번 만큼 각 폴드 세트에 학습과 검증 평가를 반복적으로 수행하는 방법이다.

![](https://velog.velcdn.com/images/adastra/post/5ac0ddc3-2a13-4a7c-908b-826a3c268a6f/image.png)

위의 사진은 5 폴드 교차 검증을 수행하는 프로세스이다.

5개의 데이터셋을 학습과 검증을 위한 데이터셋으로 변경한 뒤 4개의 학습데이터셋과 한 개의 검증데이터셋으로 나눈 뒤 학습데이터셋으로 학습 수행, 검증데이터셋으로 평가를 수행한 뒤 이를 다섯 번 번갈아가며 반복한다.  
최종적으로 나온 검증 평가 5개의 평균을 낸 결과가 k 폴드 결과로 도출된다.

사이킷런을 이용하여 k 폴드 교차 검증 프로세스를 구현해보자

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

# 5개의 폴드 세트로 분리할 KFold 객체과 폴드 세트별 정확도를 담을 리스트 객체 생성

# k개 폴드 설정
kfold = KFold(n_splits=5)

# 정확도 담을 리스트 생성
cv_accuracy = []
print('붓꽃 데이터셋 크기:', features.shape[0])
```

```
붓꽃 데이터셋 크기: 150
```

이제 생성한 KFold객체를 통해 전체 붓꽃 데이터셋을 5개의 폴드 데이터셋으로 분리한다.  
KFold 객체의 split()을 호출하여 교차검증 수행을 통해 학습과 검증을 반복하여 예측 정확도를 측정한다.

전체 붓곷 데이터셋은 150개이므로 학습용 데이터셋은 4/5인 120개, 검증용 데이터셋은 1/5인 30개로 분할된다.

```python
# 반복횟수 확인
n_iter = 0

# KFold객체의 split을 호출하면, 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 array로 반환
for train_index, test_index in kfold.split(features):
    # kfold.split()으로 반환된 index를 이용하여 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    # fit, pred
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1
    
    # 정확도 측정
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차검증 정확도: {1}, 학습데이터 크기: {2}, 검증데이터 크기: {3}'
         .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스: {1}'.format(n_iter, test_index))
    cv_accuracy.append(accuracy)
    
# 평균 정확도 계산
print('\n## 평균 검증 정확도:', np.mean(cv_accuracy))
```

![](https://velog.velcdn.com/images/adastra/post/0803c969-e8a7-4f5a-a724-cff4a1d6f855/image.png)

5개의 교차 검증 결과 평균 검증 정확도는 0.9로 나타났다.  
결과를 보면 교차 검증시 검증 세트의 index가 달라지는 것을 확인할 수 있다.  
이를 통해 split()함수가 어떻게 인덱스를 할당하는지 직관적으로 파악할 수 있다.