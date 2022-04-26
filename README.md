# python_data
## 데이터준비
* aaa
  * bbb
* ccc  
## 알고리즘 선정

```python
from pyexpat import features
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import PolynomialFeatures

def linear_normal(test_name, train_input, train_target, test_input, test_target):
        #linear 회귀
        lr = LinearRegression()
        lr.fit(train_input, train_target)
        print(test_name, "score====>", lr.score(train_input, train_target))
        print(test_name, "26, 40000====>", lr.predict([[26, 48000]]))
        print(test_name, "26, 50000====>", lr.predict([[26, 58000]]))
        #결과 [-29172269.003776] 왜 음수..? 2차 함수라 가런가, 표준화해서 그런걸수도
        # print(lr.coef_, lr.intercept_)
```
