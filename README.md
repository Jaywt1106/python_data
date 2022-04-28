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
print("knr train score", knr.score(train_scaled, train_target))
print("knr test score",knr.score(test_scaled, test_target))
```
_결과: 0.888451451339183, 0.8214245569556612_

Age와 KM를 넣고 Price를 예측해본다. 여기서는 각 모델에 [[20, 40000]], [[20, 50000]]을 예측해봤다.
KNN 20, 40000 predic result를 써두면 어떤 결과인지 한눈에 알기 쉽다
```python
print("KNN 20, 40000 predic result:", knr.predict([[20, 40000]]))
print("KNN 20, 50000 predic result:", knr.predict([[20, 50000]]))
```
_결과: 6460, 6460_
k neighbors이기 때문에 둘의 결과가 똑같은 것이 맞다.

## Linear 회귀
linear_model에서 LinearRegression을 사용한다.
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_scaled, train_target)
```

train과 test 각각의 score를 계산한다.
```python
print("lr train score", lr.score(train_scaled, train_target))
print("lr test score", lr.score(test_scaled, test_target))
```
_결과: 0.7971133926944495, 0.7698166134656372_

[[20, 40000]], [[20, 50000]]을 예측한다.
```python
print("L 20, 40000 predic result:", lr.predict([[20, 40000]]))
print("L 20, 50000 predic result:", lr.predict([[20, 50000]]))
```
_결과: -24303688.35360172, -30368071.69322803_
결과가 음수가 나왔다. 표준화하지 않은 train_input을 사용해서 다시 LinearRegression을 사용해보았다.
```python
from sklearn.linear_model import LinearRegression
lr2 = LinearRegression()
lr2.fit(train_input, train_target)
print("lr2 train score", lr2.score(train_input, train_target))
print("lr2 test score", lr2.score(test_input, test_target))
print(lr2.predict([[26, 48000]]))
print(lr2.coef_, lr.intercept_)
print("L2 20, 40000 predic result:", lr2.predict([[20, 40000]]))
print("L2 20, 50000 predic result:", lr2.predict([[20, 50000]]))
```
_score 결과: 0.797113392694449, 0.7698166134656372
predict 결과: 16678.69087867, 16516.20084828_
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
print("poly train score: ", lr.score(train_poly, train_target))
print("poly test score: ",lr.score(test_poly, test_target))
```
_score 결과: 0.8358698063795749, 0.8277360186977837_

[[20, 40000]], [[20, 50000]]을 예측한다.
```python
predict_input_poly1 = poly.transform([[20, 40000]])
print("poly 20, 40000 predic result:", lr.predict(predict_input_poly1))
predict_input_poly2 = poly.transform([[20, 50000]])
print("poly 20, 50000 predic result:", lr.predict(predict_input_poly2))
```
_predict 결과: 17402.56704151, 17248.08602732_

## decisiontree회귀
tree에서 DecisionTreeRegressor를 사용한다.
```python
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(max_depth = 3)
reg.fit(train_scaled, train_target)
```

train과 test 각각의 score를 계산한다. 
```python
print("reg train score", reg.score(train_scaled, train_target))
print("reg test score", reg.score(test_scaled, test_target))
```
_결과: 0.8390166377883685, 0.8176098739870958_

[[20, 40000]], [[20, 50000]]을 예측한다.
```python
print("REG 20, 40000 predic result:", reg.predict([[20, 40000]]))
print("REG 20, 50000 predic result:", reg.predict([[20, 50000]]))
```
_결과: 7952.34246575, 7952.34246575_

max_depth를 10으로 두고 계산해봤다.
```python
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(max_depth = 10)
reg.fit(train_scaled, train_target)
print("reg train10 score", reg.score(train_scaled, train_target))
print("reg test10 score", reg.score(test_scaled, test_target))
print("REG10 20, 40000 predic result:", reg.predict([[20, 40000]]))
print("REG10 20, 50000 predic result:", reg.predict([[20, 50000]]))
```
_score 결과: 0.9584342271815486, 0.7448760749696672
predict 결과: 6950, 6950_

## 그리드서치로 최적의 매개변수 찾기
params를 min_impurity_decrease로 두고 최적의 매개변수를 찾아봤다.
```python
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeRegressor(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print("DT train score", dt.score(train_input, train_target))
print("DT test score", dt.score(train_input, train_target))
print(gs.best_params_)
print(gs.cv_results_['mean_test_score'])
```
_score결과: 0.9996482084930518, 0.9996482084930518
{'min_impurity_decrease': 0.0001}_

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
print("DT2 train score", dt.score(train_input, train_target))
print("DT2 test score", dt.score(train_input, train_target))
print(gs.best_params_)
```
_score 결과: 0.8686622946615553, 0.8686622946615553
{'max_depth': 7, 'min_impurity_decrease': 0.0001, 'min_samples_split': 52}_

## 최상 매개변수로 돌리기
위에서 나온 최상의 매개변수를 넣어 분석을 돌렸다.
```python
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': [0.0001], 'max_depth' : [7], 'min_samples_split' : [52]}
gs = GridSearchCV(DecisionTreeRegressor(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print("REGGS 20, 40000 predic result:", reg.predict([[20, 40000]]))
print("REGGS 20, 50000 predic result:", reg.predict([[20, 50000]]))
```
_결과: 6950, 6950_

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
_결과: 0.9755272749505878 0.8226518232374758_

가장 중요한 변수를 찾아본다. [[20, 40000]], [[20, 50000]]을 예측한다.
```python
print(regr.feature_importances_)
print("REGR 20, 40000 predic result:", regr.predict([[20, 40000]]))
print("REGR 20, 50000 predic result:", regr.predict([[20, 50000]]))
```
_중요도: [0.87743731 0.12256269]
predict 결과: 17520.48, 19905_

random - oob활용 허용한다.
regr = RandomForestRegressor(oob_score=True, random_state=42, n_jobs=-1)
regr.fit(train_input, train_target)
print("REGROBB 20, 40000 predic result:", regr.predict([[20, 40000]]))
print("REGROBB 20, 50000 predic result:", regr.predict([[20, 50000]]))
_predict 결과: 17520.48, 19905_


### 함수로 
함수로 묶어서 정리하면 결과값을 보는게 더 편하다
