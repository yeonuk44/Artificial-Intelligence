# Artificial-Intelligence
인공지능을 수강하며 진행한 과제물과 문제 해결과정을 공유합니다.

## K-NN 분류기
DataSource.csv를 사용해 KNN 최근접 이웃 알고리즘을 통한 학습과 테스트를 진행하였습니다. 
### 학습목표
- DataSource.csv을 사용하여 학습과 테스트를 하시오. DataSource.csv는 11개의 컬럼을 가진다. 컬럼 id는 로우를 식별하는 타임스템프이다. X1 ~ X9 컬럼은 계측하는 데이터 항목이다. Y컬럼은 X1~X9 컬럼을 입력으로 하는 결과값이다. X1~X9 컬럼의 값중 '?'로 입력된 값은 그 값을 알 수 없는 경우를 나타낸다.
- 동일 폴더에 있는 csv파일로부터 데이터를 읽어와 DataFrame을 작성하라.
- 학습데이터 80%, 테스트데이터 20%의 비율로 데이터를 사용해라. random_state=4 해서 항상 동일한 데이터 선택이 되도록 하라.
- '?'값을 처리하여 학습결과에 영향을 주지않게 하라.
- 테스트 데이터를 사용하여 테스트를 하고 정확도를 출력하여라.
- 두개의 입력 [4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]에 대해 Y 값을 [2 2] 로 출력하라.
### 문제 해결
- pandas 라이브러리를 통해 csv 파일을 불러옵니다.
```
df = pd.read_csv('DataSource.csv')
```
- unique 함수를 통해 결측치를 확인한다. 해당 csv 파일에는 'X6'열에서 '?'의 고유 값과 다른 열과 다르게 정수형이 아닌 문자열임로 구성된 열임을 확인하였습니다.
```
uniq = df['X6'].unique()
```
