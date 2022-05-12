# python_data
# 개요
해당 프로젝트의 목적은 학생과 취업자 각각이 주관적으로 생각하는 취업에 영향을 미치는 요소를 먼저 알아본다. 그 다음, 학생과 취업자가 실제로 취업을 위해 준비한 요소들과 그로인한 결과를 분석하여 실제로 취업에 영향을 미치는 요소를 찾는 것이 목표이다. 주관적인 요소와 객관적인 요소에 차이가 있는 지도 알아볼 예정이다. 

데이터는 고용조사 분석시스템의 '청년패널 2007 1-14차 조사 통합설문지' 응답 데이터를 활용하려고 한다. 표본수가 워낙 크고 14차까지 있기 때문에 문항의 내용이 바뀐 것들이 있다. 때문에 우선 14차의 첫번째 데이터를 활용해서 분석을 하고자 한다.

# 1 설문 데이터 확인
학생(취업준비를 했으나 실패한 사람)과 취업자가 생각하는 취업에 영향을 미치는 주관적인 요소들을 알아보려고 한다. 설문의 문31C "00님께서는 다음의 각 항목이 취업에 어느 정도 영향을 미칠 것이라고 생각하십니까?"와 ----문항을 활용한다. 

문 31C와 문6A에 해당하는 답변은 1~5점 척도이다.
(1) 전혀 영향 없다 (2) 영향 없다 (3) 보통이다 (4) 영향 있다 (5) 매우 영향 있다

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
pd.read_csv를 통해 14차 조사의 첫번째 데이터 파일을 불러온다. 
그중 문 31C에 해당하는 'y14a351'~'y14a374'와 'y14a601'에 해당하는 column만을 사용하려고 한다. 해당 column들을 col_names를 통해 묶어준 뒤, loc을 활용해 해당 column들만 df_sample로 뽑아준다.
```python
data_s = pd.read_csv('c:\\workspace\\downloads\\ypdata_w14.csv')
col_names = ['y14a361', 'y14a362', 'y14a363', 'y14a364', 'y14a365', 'y14a366', 'y14a367', 'y14a368', 'y14a369', 'y14a370', 'y14a371', 'y14a372', 'y14a373', 'y14a374', 'y14a601']

df_sample = data_s.loc[:, col_names]
```

## 2.2 평균 구하기
각 항목에 대한 평균을 구한다. 똑같은 구문을 반복하는 것이기 때문에 for 문을 활용한다.
mean_df_array 리스트를 미리 만들어두고, 각 요소의 평균을 구한 뒤 리스트에 append를 통해 추가한다.
```python
mean_df_array = []

for each_col_name in col_names: 
    mean_df_array.append(df_sample[each_col_name].mean())
```
## 2.3 그래프로 표현하기
바 그래프를 통해 평균이 높은 요소들을 시각적으로 볼 수 있게 한다. plt.bar를 활용한다.
title은 '14th student sub'로 14차 학생의 주관적인 답변이다.
xlabel은 'Questions'로 각 요소이다.
ylabel은 'Score'로 학생이 생각하는 중요도 점수이다.

질문을 'y14a**'로 표현하면 어떤 요소에 대한 질문인지 알기 어렵기 때문에 labels를 활용했다.
```python
labels = ['학벌', '학점', '공인영어성적', '영어회화', '제2외국어', '한자능력', '컴퓨터 자격증', '직무관련 자격', '해외경험', '인턴경험', '공모전 경력', '석박사 학위', '봉사경험', '동아리경험', '학지혈연']
index = np.arange(len(col_names))
plt.bar(index, mean_df_array)
plt.title('14th student sub', fontsize=20)
plt.xlabel('Questions', fontsize=18)
plt.ylabel('Score', fontsize=18)
plt.xticks(index, labels, fontsize=10)
```
```python
한글 폰트를 사용하기 위해 세팅하였다.
plt.rc('font', family='NanumGothic') # For Windows
print(plt.rcParams['font.family'])
```
## 2.4 결과

# 3 취업자 데이터 분석
## 3.1 데이터 불러오기
pd.read_csv를 통해 14차 조사의 첫번째 데이터 파일을 불러온다. 
그중 문 ****C에 해당하는 'y14b293'~'y14b306'와 'y14b396'에 해당하는 column만을 사용하려고 한다. 해당 column들을 col_names를 통해 묶어준 뒤, loc을 활용해 해당 column들만 df_sample로 뽑아준다.
```python
data_w = pd.read_csv('c:\\workspace\\downloads\\ypdata_w14_1.csv', low_memory=False)
col_names2 = ['y14b293', 'y14b294', 'y14b295', 'y14b296', 'y14b297', 'y14b298', 'y14b299', 'y14b300', 'y14b301', 'y14b302', 'y14b303', 'y14b304', 'y14b305', 'y14b306', 'y14b396']
df_sample2 = data_w.loc[:, col_names2]
```

## 3.2 평균 구하기
각 항목에 대한 평균을 구한다. 똑같은 구문을 반복하는 것이기 때문에 for 문을 활용한다.
mean_df_array 리스트를 미리 만들어두고, 각 요소의 평균을 구한 뒤 리스트에 append를 통해 추가한다.
```python
mean_df_array2 = []

for each_col_name2 in col_names2: 
    mean_df_array2.append(df_sample2[each_col_name2].mean())
```

## 3.3 그래프로 표현하기
바 그래프를 통해 평균이 높은 요소들을 시각적으로 볼 수 있게 한다. plt.bar를 활용한다.
title은 '14th worker sub'로 14차 취업자의 주관적인 답변이다.
xlabel은 'Questions'로 각 요소이다.
ylabel은 'Score'로 학생이 생각하는 중요도 점수이다.

질문을 'y14b**'로 표현하면 어떤 요소에 대한 질문인지 알기 어렵기 때문에 labels를 활용했다. 학생의 요소와 순서는 똑같기 때문에 같은 labels를 활용한다.
```python
index2 = np.arange(len(col_names2))
plt.bar(index2, mean_df_array2)
plt.title('14th worker sub', fontsize=20)
plt.xlabel('Questions', fontsize=18)
plt.ylabel('Score', fontsize=18)
plt.xticks(index2, labels, fontsize=10)
```

## 3.4 결과
