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


```python
print("KNN 20, 40000 predic result:", knr.predict([[26, 40000]]))
print("KNN 20, 50000 predic result:", knr.predict([[26, 50000]]))
```
