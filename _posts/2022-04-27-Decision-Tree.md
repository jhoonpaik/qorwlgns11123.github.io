---
title: Decision Tree
description: 앙상블의 기본 알고리즘인 결정트리는 매우 쉽고 유연하게 적용할 수 있는 알고리즘이다.
tags:
- Machine learning
- classification
---


# 분류의 개요

지도학습은 레이블(명확한 정답)이 있는 데이터가 주어진 상태에서 학습하는 머신러닝 학습방식이다. 반대로 비지도학습인 정답이 없는 상태에서 학습하는 머신러닝 방식이다.

지도학습의 대표적인 유형은 분류(Classification)은 학습데이터로 주어진 피처, 레이블 값을 이용해 머신러닝을 학습하여 모델을 생성하고 이렇게 생성된 모델을 통해 새로운 데이터 값에 대한 미지의 레이블 값을 예측하는 것이다.
분류는 아래와 같이 다양한 머신러닝 알고리즘으로 구현할 수 있다.

- 나이브 베이즈
- 로지스틱 회귀
- 결정트리
- 서포트 벡터 머신
- 최소 근접(Nearest Neighbor) 알고리즘
- 신경망(Neural Network)
- 서로 다른(또는 같은) 머신러닝 알고리즘을 결합한 앙상블(Ensemble)


# 결정트리

앙상블은 서로 같은 또는 다른 알고맂므을 단순히 결합한 형태이지만 보통 배깅과 부스팅으로 나뉜다. 배깅의 대표적인 방식은 랜덤포레스트로 뛰어난 예측성능과 빠른 수행시간을 갖는다. 다른 방식인 부스팅의 경우 지속적으로 발전하고 있다.

앙상블의 기본 알고리즘인 결정트리는 매우 쉽고 유연하게 적용할 수 있는 알고리즘이다.
결정트리는 데이터에 있는 규칙을 학습을 통해 자동을 찾아내는 트리기반의 분류규칙을 만든다.
가장 간단하게 생각하면 데이터에서 if/else를 자동으로 찾아내 예측을 위한 규칙을 생성하는 방식이다.

그러므로 데이터를 바탕으로 어떤 기준을 통해 규칙을 생성해야 효율적인 분류가 되는가가 알고리즘 성능의 요체이다.

![](https://velog.velcdn.com/images/adastra/post/bc6cc2cd-f39c-4914-a081-a6e81505f16b/image.png)

위와 같은 구조가 트리 구조의 기본 방식이다. 여기서 규칙노트는 규칙조건이 되는 노드이고 리프노드는 결정된 클래스값이다. 그리고 새로운 규칙노드가 생성될 때마다 서브트리가 생성된다. 하지만 계속 아래로 많은 규칙이 생성될 때마다 분류 결정방식이 복잡해지면서 과적합으로 이어질 수 있다. 즉, 트리 깊이가 깊어질 수록 예측성능이 저하될 가능성이 높다.

즉, 결정트리가 높은 예측성능을 가지려면 최대한 많은 데이터셋이 분류에 속하도록 규칙이 정해져야 하며 이런 방식으로 트리가 분할되기 위해선 최대한 균일한 데이터셋을 구성하도록 분할하는 것이 중요하다.

그렇다면 균일할 데이터셋이라는 것은 무엇인가?
![](https://velog.velcdn.com/images/adastra/post/e399ce08-53b8-4063-b861-609bff2fc7cf/image.png)

위의 세 가지 데이터셋을 균일도가 높은 순으로 나열하면 C > B > A 순이다.

결정노드는 균일도가 높은 데이터셋을 먼저 선택할 수 있도록 규칙조건을 만든다.
즉, 균일도가 높은 데이터셋을 규칙조건으로 나누고 나눈 데이터를 또 균일도가 높은 데이터셋을 규칙조건을 통해 나누는 방식으로 진행된다.

예를 들어 30개의 블록이 있고 색깔은 빨강, 노랑, 파랑, 모양은 세모, 네모, 원으로 되어있다고 가정할 경우
이 중 노랑색 블록은 모두 원으로 구성되어있고 나머지 빨랑, 파랑 블록은 골고루 섞여있다고 가정한다면 가장 첫 번째로 형태와 색깔 속성을 분류할 때 첫 번째 규칙조건은 if color = "yellow"가 될 것이다. 그 이유는 노란색 블록은 모두 원으로 구성되어있으므로 가장 쉽게 분할할 수 있기 때문이다.

이렇게 데이터의 균일도를 측정하는 대표적인 방식은 엔트로피를 이용한 정보이득 지수와 지니계수가 있다.

정보이득은 엔트로피라는 개념을 기반으로 한다. 엔트로피는 데이터 집합의 혼잡도를 의미한다.
즉, 서로 다른 값이 섞여있을 수록 엔트로피가 높고 같은 값이 섞여있을수록 엔트로피가 낮다.
정보이득지수는 1-엔트로피 지수이다.
결정트리는 이 정보이득이 높은 속성을 기준으로 분할한다.

지니계수는 경제학에서 불평등 지수를 나타낼 때 사용한 계수로 머신러닝에 적용시 지니계수가 낮을수록 데이터
균일도가 높은 것으로 해석하여 지니계수가 낮은 속성을 기준으로 분할한다.


## 결정트리 파라미터

결정트리는 정보의 균일도만 신경쓰면 되므로 특별한 경우가 아니면 피처 스케일링과 정규화 같은 전처리 작업이 필요없다.
대신 앞서 언급한 과적합의 문제를 극복하기 위해 트리의 크기를 사전에 제한하는 방식으로 성능 튜닝을 진행한다.

결정트리 파라미터는 아래와 같다.
- min_samples_split: 노드 분할을 위한 초소한의 샘플데이터수(디폴트=2). 작게 설정할 수록 분할노드가 많아져 과적합 가능성 증가
- min_samples_leaf: 말단노드가 되기 위한 최소한의 샘플 데이터 수
- max_features: 최적의 분할을 위해 고려할 최대 피처 개수
- max_depth: 트리의 최대 깊이 규정
- max_leaf_nodes: 말단노드 최대개수


## 결정트리 모델 시각화

graphviz패키지를 통해 결정트리알고리즘의 규칙 생성과정을 직관적으로 확인할 수 있다.
사이킷런의 export_graphviz() API는 이러한 Graphviz패키지를 쉽게 인터페이스하는 기능을 갖고 있다.

graphviz 설치방법참고: https://aeir.tistory.com/entry/graphviz-%EA%B0%84%EB%8B%A8-%EC%84%A4%EC%B9%98-%EB%B0%A9%EB%B2%95-%EC%9C%88%EB%8F%84%EC%9A%B010-1

iris dataset를 통해 DecisionTree를 학습하고 어떤 형태로 트리가 만들어지는지 확인해보자

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 결정트리 생성
df_clf = DecisionTreeClassifier(random_state=132)

# iris 데이터 로딩, 분리
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target,
                                                   test_size=0.2, random_state=11)

# 학습
dt_clf.fit(X_train, y_train)

# grpahviz
from sklearn.tree import export_graphviz

# export_graphviz() 호출결과로 out_file로 지정된 tree.dot 파일 생성
export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names, \
               feature_names = iris_data.feature_names, impurity=True, filled=True)
               # impurity=True: gini 출력, filled_True: 노드색깔 다르게

```

이렇게 생성된 출력파일 tree.dot을 Grapviz 파이썬 래퍼모듈을 호출해 결정트리 규칙을 시각적으로 표현할 수 있다.

```python
import graphviz

# treedot 파일을 graphviz가 읽어서 시각화
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
```
![](https://velog.velcdn.com/images/adastra/post/686b9b62-378e-40d6-b3dd-1f47b2a3b4e8/image.png)

위와 같이 트리의 노드가 리프노드까지의 깊이 형태가 어떻게 구성되는지 한 눈에 확인할 수 있다.

노드의 구성형태를 보면,  
petal length(cm) <= 2.45와 같이 조건이 있는 것이 자식 노드 생성을 위한 규칙 조건이고 이 조건이 없으면 리프노드이다.  
samples는 현 규칙에 해당되는 데이터 건수를 의미한다.  
value = [0, 0, 2]는 클래스값 기간의 데이터 건수이다.  
iris데이터셋은 클래스값을 0, 1, 2를 갖고 있기 때문에 [0, 0, 2]는 Setosa 0개, Versicolor 0개, Virginica 39개로 데이터 구성된 것을 파악할 수 있다.

> 참고자료: 파이썬 머신러닝 완벽 가이드 - 권철민