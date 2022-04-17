---
title: scikit learn
description: 사이킷런(scikit learn)은 파이썬 머신러닝 라이브러리중 가장 많이 사용되는 라이브러리다.
tags:
- Machine learning
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

## 이제 iris 데이터셋의 피처, 피처명과 레이블을 각각 확인해보자

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

## 최종적으로 피처와 레이블 합쳐서 데이터프레임을 생성해보자

```python
import pandas as pd

iris_df = pd.DataFrame(data=iris_features, columns = iris.feature_names)
# iris_df
iris_df['label'] = iris.target
iris_df
```
![](https://velog.velcdn.com/images/adastra/post/735d7cda-3744-4c56-b78e-66d9f17d388f/image.png)

위와 같이 iris 데이터프레임이 완성되었다.

> 참고자료: 파이썬 머신러닝 완벽 가이드 - 권철민