# python_data

## 데이터준비
csv 파일을 준비하고 그 중 'Age', 'KM'를 data, 'Price'를 target으로 설정한다.
```python
from pyexpat import features
import numpy as np
import pandas as pd
cars = pd.read_csv('c:\\workspace\\downloads\\ToyotaCorolla.csv')
print(cars.head())
data = cars[['Age', 'KM']].to_numpy()
target = cars['Price'].to_numpy()
```


data와 target이 제대로 들어갔는지 확인한다.
```python
print(data.shape)
print(data[:5])
print(target[:5])
```

## 훈련, 테스트 데이터 준비
train_test_split를 이용한다.
```python
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, random_state=42)
```

표준화를 해준다.
```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```


## K 근접 이웃 회귀
neighbors에서 KNeighborsRegressor를 사용한다.
```python
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_scaled, train_target)
```

train과 test 각각의 score를 계산한다.
```python
print(knr.score(train_scaled, train_target))
print(knr.score(test_scaled, test_target))
```

Age와 KM를 넣고 Price를 예측해본다. 여기서는 각 모델에 [[20, 40000]], [[20, 50000]]을 예측해봤다.
```python
print("KNN 20, 40000 predic result:", knr.predict([[20, 40000]]))
print("KNN 20, 50000 predic result:", knr.predict([[20, 50000]]))
```


## Linear 회귀
linear_model에서 LinearRegression을 사용한다.
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_scaled, train_target)
```


train과 test 각각의 score를 계산한다.
```python
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

[[20, 40000]], [[20, 50000]]을 예측한다.
```python
print(lr.predict([[20, 40000]]))
print(lr.predict([[20, 50000]]))
```

결과가 음수가 나왔다. 표준화하지 않은 train_input을 사용해서 다시 LinearRegression을 사용해보았다.
```python
lr2 = LinearRegression()
lr2.fit(train_input, train_target)
print("lr2 train score", lr2.score(train_input, train_target))
print("lr2 test score", lr2.score(test_input, test_target))
print(lr2.predict([[26, 48000]]))
print(lr2.coef_, lr2.intercept_)
print("L 20, 40000 predic result:", lr2.predict([[20, 40000]]))
print("L 20, 50000 predic result:", lr2.predict([[20, 50000]]))
```
제대로 된 결과가 나왔다.
