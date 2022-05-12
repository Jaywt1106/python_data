# python_data
# 개요
해당 프로젝트의 목적은 학생과 취업자 각각이 주관적으로 생각하는 취업에 영향을 미치는 요소를 먼저 알아본다. 그 다음, 학생과 취업자가 실제로 취업을 위해 준비한 요소들과 그로인한 결과를 분석하여 실제로 취업에 영향을 미치는 요소를 찾는 것이 목표이다. 주관적인 요소와 객관적인 요소에 차이가 있는 지도 알아볼 예정이다. 

데이터는 고용조사 분석시스템의 '청년패널 2007 1-14차 조사 통합설문지' 응답 데이터를 활용하려고 한다. 표본수가 워낙 크고 14차까지 있기 때문에 문항의 내용이 바뀐 것들이 있다. 때문에 우선 14차의 첫번째 데이터를 활용해서 분석을 하고자 한다.

# 1 설문 데이터 확인
학생(취업준비를 했으나 실패한 사람)과 취업자가 생각하는 취업에 영향을 미치는 주관적인 요소들을 알아보려고 한다. 설문의 문31C "00님께서는 다음의 각 항목이 취업에 어느 정도 영향을 미칠 것이라고 생각하십니까?"와 ----문항을 활용한다. 

문 31A와 문6A에 해당하는 답변은 1~5점 척도이다.
 (1) 전혀 영향 없다
 (2) 영향 없다
 (3) 보통이다
 (4) 영향 있다
 (5) 매우 영향 있다

응답자들이 각 요소에 대해 매긴 점수의 평균을 plot으로 그린다. 가장 평균이 높은 3개의 요소를 도출한다.

# 2 학생 데이터 분석
## 2.0 기본 설정
```python
from pyexpat import features
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
```
## 2.1 데이터 불러오기
```python
data_s = pd.read_csv('c:\\workspace\\downloads\\ypdata_w14.csv')
col_names = ['y14a361', 'y14a362', 'y14a363', 'y14a364', 'y14a365', 'y14a366', 'y14a367', 'y14a368', 'y14a369', 'y14a370', 'y14a371', 'y14a372', 'y14a373', 'y14a374', 'y14a601']

df_sample = data_s.loc[:, col_names]
```

## 2.2 평균 구하기
```python
mean_df_array = []

for each_col_name in col_names: 
    mean_df_array.append(df_sample[each_col_name].mean())
```
