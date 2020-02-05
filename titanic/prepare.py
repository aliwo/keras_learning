import pandas
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# csv 를 numpy 배열로 변환.
# 실수만 들어있는 numpy 배열로 바꾸려면 어떻게 해야 할까?
data = pandas.read_csv('./data/train.csv')


# 이름은 생존과는 상관 없다.
# Ticket 넘버 역시 의미 없다. 괜히 신경망에 넣었다가 점만 치는 꼴이다.
# fare 가 상관 있을까? 이건 시험해 봐야 할듯
# embark... 어디서 탔느냐가 상관이 있을까? 없다고 본다. 일단 이것도 뺀다.
# cabin 은 문자열이다... 생존과 관련이 꽤 있어보이긴 하지만... 단순화를 위해 뺸다.
data.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], 1, inplace=True)


# name 에서 male 은 1.0, female 은 0.0으로 바꾼다.
# 모든 값이 실수이므로 numpy 배열로 만든다.
# 나이에서 빈 값은 중간값으로 채운다.
# (사망자는 사망자의 중간값으로, 생존자는 생존자의 중간값으로 만들어 버릴까?)
imputer = SimpleImputer(strategy='median')
data['Sex'] = data['Sex'].apply(lambda sex: 1.0 if sex == 'male' else 0.)
imputer.fit(data)
data = imputer.transform(data)


# 큰 값들은 평균을 빼고 표준편차로 나눈다.
min_max_scaler = MinMaxScaler()
to_process = data[:, 2:]
min_max_scaler.fit(to_process)
processed = min_max_scaler.transform(to_process)


# label 은 survived 컬럼만 따로 빼야 할듯.
label = data[:, 1]


def load():
    return (processed[100:], label[100:]), (processed[:100], label[:100])


# '예측' 인데 정확도가 100% 나오는 애들은 뭐지? 특정 기준을 만족하면 무조건 생사가 갈린다는 거야?
