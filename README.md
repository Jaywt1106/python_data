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

## 다중회귀
다중회귀를 위해서 새로운 변수를 추가해준다. 이때 PolynomialFeatures를 사용한다. 
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print("test_poly:", test_poly.shape)
```

linear_model에서 LinearRegression을 사용한다.
```python
lr = LinearRegression()
lr.fit(train_poly, train_target)
```

train과 test 각각의 score를 계산한다. test_name, "score: "를 넣어주면 어떤 score인지 쉽게 알 수 있다.
```python
print(test_name, "score: ", lr.score(train_poly, train_target))
print(test_name, "score: ",lr.score(test_poly, test_target))
```
[[20, 40000]], [[20, 50000]]을 예측한다.
```python
predict_input_poly1 = poly.transform([[20, 40000]])
print(test_name, "lr(20, 40000)====>", lr.predict(predict_input_poly1))
predict_input_poly2 = poly.transform([[20, 50000]])
print(test_name, "lr(20, 50000)====>", lr.predict(predict_input_poly2))
```
## 다중회귀decisiontree회귀
tree에서 DecisionTreeRegressor를 사용한다.
```python
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(max_depth = 3)
reg.fit(train_scaled, train_target)
```

train과 test 각각의 score를 계산한다. 
```python
print(reg.score(train_scaled, train_target))
print(reg.score(test_scaled, test_target))
```

[[20, 40000]], [[20, 50000]]을 예측한다.
```python
print(reg.predict([[20, 40000]]))
print(reg.predict([[20, 50000]]))
```

max_depth를 10으로 두고 계산해봤다.
```python
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(max_depth = 10)
reg.fit(train_scaled, train_target)
print(reg.score(train_scaled, train_target))
print(reg.score(test_scaled, test_target))
print(reg.predict([[20, 40000]]))
print(reg.predict([[20, 50000]]))
```

## 그리드서치로 최적의 매개변수 찾기
params를 min_impurity_decrease로 두고 최적의 매개변수를 찾아봤다.
```python
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeRegressor(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(gs.best_params_)
print(gs.cv_results_['mean_test_score'])
```

params를 머신러닝 책에 나왔던 것과 똑같게 min_impurity_decrease, max_depth, min_samples_split로 두었다. 숫자도 책과 같다.
```python
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])
params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
        ', max_depth' : range(5, 20, 1),
        'min_samples_split' : range(2, 100, 10)}
gs = GridSearchCV(DecisionTreeRegressor(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(gs.best_params_)
```

## 최상 매개변수로 돌리기
위에서 나온 최상의 매개변수를 넣어 분석을 돌렸다.
```python
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': [0.0001], 'max_depth' : [7], 'min_samples_split' : [52]}
gs = GridSearchCV(DecisionTreeRegressor(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(reg.predict([[20, 40000]]))
print(regr.predict([[20, 50000]]))
```

## randomforest회귀
```python
cross_validate를 활용해 교차검증을 한다. ensemble에서 RandomForestRegressor를 사용한다.
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(random_state=42, n_jobs=-1)
```

train_score와 test_score의 평균을 계산한다.
```python
scores = cross_validate(regr, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
regr.fit(train_input, train_target)
```

가장 중요한 변수를 찾아본다. [[20, 40000]], [[20, 50000]]을 예측한다.
```python
print(regr.feature_importances_)
print(regr.predict([[20, 40000]]))
print(regr.predict([[20, 50000]]))
```
