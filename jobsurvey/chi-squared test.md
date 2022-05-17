# python_data

# 1 설문 데이터 확인
이제는 학생(취업준비를 했으나 실패한 사람)이 실제로 취업을 위해 준비한 스펙과 취업자가 취업을 위해 준비한 스펙을 통해 실제로 각각의 스펙을 준비하는 것이 취업에 도움이 되었는지를 확인해보려고 한다.

문 31A &#34;__님께서 취업을 위해 지금까지 준비했거나 현재 준비하고 있는 스펙은 무엇입니까?&#34;와 문6A로 "님께서 취업을 위해 준비한 스펙은 무엇입니까?"를 통해 각각이 준비한 스펙을 알 수 있을 것으로 보인다. 항목 또한 이전에 분석했던 항목들과 일치한다.(다만 학벌과 학/지/혈연이 빠져있다.) 

문 31A와 문6A에 해당하는 답변 또한 이전에 분석했던 "주관적으로 느낀 각 항목이 취업에 미친 영향"과 똑같이 1~5점 척도라고 생각했다. 하지만 이번 질문들에 대한 대답은 (1) 예, (2) 아니오 두 가지로 나뉜다. 이에 따라 연속형 -> 범주형인 로지스틱 회귀에서 범주형 -> 범주형인 카이제곱 검증으로 변경하였다.

# 2 데이터 준비
## 2.0 기본 설정
```python
from pyexpat import features
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
```

## 2.1 데이터 가져오기
데이터는 "y14a265"와 "y14b279"를 모두 답하지 않은 사람을 제외하여 준비하였다. pandas의 read_csv를 이용하여 ypdata_w14_fin.csv 파일을 불러와 사용하였다. 
```python
dataframe = pd.read_csv('c:\\workspace\\downloads\\ypdata_w14_fin.csv',  low_memory=False)
```
column에 Nan이나 type이 다른 데이터들이 섞여있는 경우 오류가 발생한다. 이 문제를 해결해주기 위해 low_memory=False를 썼다.

## 2.2 취업 여부 column 추가해주기
취업 여부를 알려주는 column이 없기 때문에 추가할 필요가 있다. 이때, 'y14a***'에 대해 답변을 한 사람은 미취업자, 'y14b***'에 답변을 한 사람은 취업자라고 볼 수 있다. 이에 따라 'y14a265'에 대한 답변이 Nan인 경우는 1 (취업자), Nan이 아닌 경우는 2 (미취업자)로 나타내주었다.

np.where은 조건을 만족하는 인덱스를 반환해주는 함수이다. 조건으로 Nan인지를 확인하기 위해 isna() 메서드를 사용했다.
```python
dataframe['취업성공여부'] = np.where(pd.isna(dataframe['y14a265'])==True, 1, 2)
```
## 2.3 데이터 분할
'y14a***'에 대한 답변을 data_1으로, 'y14b***'에 대한 답변을 data_2로 해당 column만 가져와서 사용하려고 한다. 1.1단계예서 read_csv() 함수로 데이터프레임을 만든 다음 to_numpy() 메서드를 사용해 넘파이 배열로 바꿔준다.
```python
data_1 = dataframe[['y14a265', 'y14a266', 'y14a267', 'y14a268', 'y14a269', 'y14a270', 'y14a271', 'y14a272', 'y14a273', 'y14a274', 'y14a275', 'y14a276', 'y14a277']].to_numpy()
data_2 = dataframe[['y14b279', 'y14b280', 'y14b281', 'y14b282', 'y14b283', 'y14b284', 'y14b285', 'y14b286', 'y14b287', 'y14b288', 'y14b289', 'y14b290', 'y14b291']].to_numpy()
target = dataframe['취업성공여부'].to_numpy()
```

# 3. 데이터 재정의
chi2_contingency를 사용해서 먼저 학점에 대한 카이제곱 검증을 시작하려고 했으나, 학점에 대한 답변이 학생은 'y14a265', 취업자는 'y14b279'로 나뉜다는 문제가 있었다. 판다스의 concat을 통해 붙여주려고 했으나, "cannot concatenate object of type '<class 'str'>'; only Series and DataFrame objs are valid" 경고 메세지가 떴다. 문자형 데이터는 concat을 통해 붙일 수 없는 것 같다. 

df_1 (미취업자의 df)에서 첫번째 column(학점)과 df_2 (취업자의 df)에서 첫번째 column을 붙이면 된다고 생각하였다. 각 df에서 첫번째 column만을 사용하기 위해 iloc 함수를 이용했다. 

```python
df_score = pd.concat([df_1.iloc[0], df_2.iloc[0]])
```

int를 통해 숫자로 데이터 타입을 int로 바꿔주려 했으나, 'y14a265'안에도 숫자와 Nan이 섞여 있어 불가능한 것으로 보인다. 때문에 Nan을 0으로 바꿔주려 한다. fillna를 통해 Nan을 0으로 바꿔주었다.
```python
df_1 = pd.DataFrame(data_1)
df_1 = df_1.fillna(0)

df_2 = pd.DataFrame(data_2)
df_2 = df_2.fillna(0)
```
*여기서 치명적인 문제를 발견했다. 단순하게 취업자와 미취업자의 응답을 밑으로 연결해 한 column으로 연결시키는 것이 아니다. 

## 3.1 data 간단하게 설정하기
data는 취업자와 미취업자를 구분하지 않고 data 그대로를 사용했다.

data를 그대로 다시 사용하는 과정에서 질문이 'y14a265'~'y14a277'까지로 dataframe에서 그대로 질문 명을 가져왔을 때 줄이 너무 길어지는 불편함이 있었다. for문을 사용해 column의 name들을 더 간단하게 써준다.
```python
def get_data_column_names():
    column_names = []
    for i in range(265, 278):
        column_names.append('y14a' + str(i))

    for i in range(279, 292):
        column_names.append('y14b' + str(i))

    return column_names
```
## 3.2 취업 여부 column 이름 변경
취업 여부의 이름은 'employed'으로 설정하였다. 위에서와 똑같은 방법으로 'employed' column을 생성한다.
```python
dataframe['employed'] = np.where(pd.isna(dataframe['y14a265'])==True, 1, 2)
```

## 3.3 data와 target을 준비
```python
column_names = get_data_column_names()
data = dataframe[column_names].to_numpy()
target = dataframe['employed'].to_numpy()
```

## 3.4 score column 생성
data의 첫번째 column과 14번째 column이 똑같이 학점과 관련된 질문이므로 둘을 비교해야 한다. 첫번째 column과 14번째 column의 데이터 길이가 같은지 확인하였다.

```python
print("the number of columns in data:", len(data[0]))
print("the length of the 1st column in data:", len(data[:, [0]]))
print("the length of the 14th column in data:", len(data[:, [13]]))

grade_data = data[:, [0, 13]]
print("the number of columns in grade_data:", len(grade_data[0]))
print("the length of the student column in grade_data:", len(grade_data[:, [0]]))
print("the length of the worker column in grade_data:", len(grade_data[:, [1]]))
```
먼저 두 column에서 Nan인 데이터를 0으로 바꾼다. 그 다음 score 리스트를 만들어 두고, 두 column을 비교하면서 더 큰 숫자를 score로 반환하도록 하면 된다. 여기서 fillna(0)(Nan인 경우 0을 채워줌)와 max(axis=1)(column 방향으로 비교해 더 큰 수 반환해줌)을 써준다. 이렇게 반환된 숫자들을 score 리스트에 append를 통해 추가해준다. (첫번째 column에 답을 하지 않아 Nan인 경우를 0으로 바꾸고, 14번째 column의 수와 비교하면 답을 한 14번째의 수가 무조건 더 크다는 것을 활용하였다.)
```python
#score = []
#df_grade_data = pd.DataFrame(grade_data)
#df_grade_merged = df_grade_data.fillna(0).max(axis=1) 
#score.append(df_grade_merged)
```
오류가 계속되고 차원이 맞지 않는 문제가 발생했다. score에 하나씩 추가를 하는 접근이 잘못된 것이다. df_grade_data.fillna(0).max(axis=1) 자체를 score로 사용하면 된다.

바로 위의 코드를 다음고 같이 바꿔준다.
```python
df_grade_data = pd.DataFrame(grade_data)
score = df_grade_data.fillna(0).max(axis=1) 
```
처음에 오류가 계속되었을 때는 fillna(0).max(axix=1) 자체의 문제라고 생각했다. 이 대신 if문을 사용해 직접 하나씩 푸는 방법도 있다.
```python
df_grade_merged = []
for i in range(0, len(grade_data)):
    col1 = 0
    col2 = 0
    # print(">>>>", grade_data[0])
    if (not math.isnan(grade_data[i][0])):
         col1 = grade_data[i][0]    
    
    if (not math.isnan(grade_data[i][1])):
         col2 = grade_data[i][1]    
    
    if (col1 > col2):
         df_grade_merged.append(col1)
    else:
         df_grade_merged.append(col2)
```

pritn(score)를 해봤을 때 계속 2만 출력되었다. score 안에 데이터가 2밖에 없는 문제가 있지 않을까라는 생각이 들어 직접 score 안의 1과 2의 개수를 확인해보았다.
```python
val1Records = 0
val2Records = 0
errors = 0

for i in range(0, len(score)):
        # print(">>>>", df_grade_merged[i])
        if (score[i]) == 1:
            val1Records += 1
        elif (score[i]) == 2:
            val2Records += 1
        else:
            errors += 1

print("score data: 1={}, 2={}, error={}".format(val1Records, val2Records,errors ))
```
