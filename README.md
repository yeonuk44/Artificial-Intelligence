# Artificial-Intelligence
인공지능을 수강하며 진행한 과제물과 문제 해결과정을 공유합니다.

## K-NN 분류기
DataSource.csv를 사용해 KNN 최근접 이웃 알고리즘을 통한 학습과 테스트를 진행하였습니다. 
### Tech Stack
- Python
### 개발 환경
- 아나콘다(주피터)
### 학습 목표
- DataSource.csv을 사용하여 학습과 테스트를 하시오. DataSource.csv는 11개의 컬럼을 가진다. 컬럼 id는 로우를 식별하는 타임스템프이다. X1 ~ X9 컬럼은 계측하는 데이터 항목이다. Y컬럼은 X1~X9 컬럼을 입력으로 하는 결과값이다. X1~X9 컬럼의 값중 '?'로 입력된 값은 그 값을 알 수 없는 경우를 나타낸다.
- 동일 폴더에 있는 csv파일로부터 데이터를 읽어와 DataFrame을 작성하라.
- 학습데이터 80%, 테스트데이터 20%의 비율로 데이터를 사용해라. random_state=4 해서 항상 동일한 데이터 선택이 되도록 하라.
- '?'값을 처리하여 학습결과에 영향을 주지않게 하라.
- 테스트 데이터를 사용하여 테스트를 하고 97% 이상의 정확도를 출력하여라.
- 두개의 입력 [4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]에 대해 Y 값을 [2 2] 로 출력하라.
### 문제 해결
- pandas 라이브러리를 통해 csv 파일을 불러옵니다.
```
df = pd.read_csv('DataSource.csv')
```
- unique 함수를 통해 결측치를 확인하였습니다. 해당 csv 파일에는 'X6'열에서 '?'의 고유 값과 다른 열과 다르게 정수형이 아닌 문자열임로 구성된 열임을 확인하였습니다.
```
uniq = df['X6'].unique()
```
- '?' 값이 학습결과에 영향을 주지 않도록 제거해야 합니다. dropna 함수를 사용해 결측치를 제거하기 위해선 '?'를 Nan으로 교체할 필요성이 있습니다. 따라서 replace 함수를 통해 '?'를 Nan으로 변경 후 결측치를 제거해줍니다. 이때 dropna의 속성을 이용하지 않는 다면 결측치가 존재하는 모든 행을 제거함으로 파이썬 라이브러리의 속성을 통해 용도에 맞게 변경하면 될 것입니다.
```
df['X6'].replace('?', np.nan, inplace = True) 
df.dropna(subset = ['X6'], axis = 0, inplace = True)
```
- unipue 함수로 확인한 바, 'X6'열 전체가 문자열이였음으로 astype 함수를 통해 다른 열과 동일한 정수형으로 변경해줍니다.
```
df['X6'] = df['X6'].astype('int')
```
- index를 포함한 11열이기 때문에 학습을 위해 index를 제외한 X와 Y를 묶도록 하겠습니다.(변수 명에 큰 의미는 없습니다.)
```
qt = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']]
qt_y = df[['Y']]
```
- 이제 train_test_spilt을 통해 학습데이터 80%와 테스트데이터 20%, random_state = 4 속성을 가진 모델을 생성합니다. 속성을 동일하게 해주어야 모든 사람이 해당 csv 로 학습 시 동일한 결과값을 얻을 수 있습니다. fit 함수를 통해 학습을 시작합니다. 문제 해결에서는 k 값에 대한 언급이 없었기 때문에, 97% 이상의 정확도가 나올 수 있도록 n_neighbors 값을 조절합니다.
```
training_X,vaildation_X,training_Y,vaildation_Y = train_test_split(qt.dropna(), qt_y.dropna(), test_size=0.2, random_state=4)
classifier = KNeighborsClassifier(n_neighbors = 4, weights = "distance") 
classifier.fit(training_X, training_Y) 
print('정확도: {0:.16f}'.format(classifier.score(vaildation_X, vaildation_Y )))
```
- 두 개의 입력 값을 변수에 담은 뒤 predicr 함수를 통해 예측 값을 출력합니다.
```
unknown_points = [[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]] 
guesses = classifier.predict(unknown_points) 
print('예측값:', guesses)
```
### 생각이 많이 된 점
- 문제를 해결하며,처음 '?' 값을 없애기 위해 != 부등 비교 연산자를 사용하여 '?' 값을 필터링하고자 하였습니다. 그 결과 정확도가 100%가 되었고, 이를 이상하게 여겨 열들의 '?' 없앤 결과 값을 출력하였습니다. 아니나 다를까 모든 값이 True | False로 불리언 자료형으로 변경되어 있었습니다... 이를 해결하기 위해 부등 비교 연산자(!=) 가 아닌 dropna() 함수를 사용해 결측치를 없애고자 하였고, 결과는 '?'는 Nan 값이 아니기 때문에 필터링 되지 않았습니다. 이후 np.nan을 활용해 '?'를 Nan 값으로 변경하였고, 문제를 해결할 수 있었습니다.
