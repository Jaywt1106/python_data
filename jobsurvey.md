# python_data

## 데이터준비
이제는 학생(취업준비를 했으나 실패한 사람)이 실제로 취업을 위해 준비한 스펙과 취업자가 취업을 위해 준비한 스펙을 통해 실제로 각각의 스펙을 준비하는 것이 취업에 도움이 되었는 지를 확인해보려고 한다.
문 31A의 질문이 "____님께서 취업을 위해 지금까지 준비했거나 현재 준비하고 있는 스펙은 무엇입니까?", 문---로 ----인 것 같다. 항목 또한 이전에 분석했던 항목들과 일치한다.(다만 학지혈연과 ---가 빠져있다.)
문 31A와 ---에 해당하는 답변 또한 이전에 분석했던 ----과 똑같이 1~5점 척도라고 생각했다. 하지만 이번 질문들에 대한 대답은 (1) 예, (2) 아니오 두 가지로 나뉜다. 이에 따라 연속형 -> 범주형인 로지스틱 회귀에서 범주형 -> 범주형인 카이제곱 검증으로 변경하였다.
```python
from pyexpat import features
import numpy as np
import pandas as pd
cars = pd.read_csv('c:\\workspace\\downloads\\ToyotaCorolla.csv')
print(cars.head())
data = cars[['Age', 'KM']].to_numpy()
target = cars['Price'].to_numpy()
```
