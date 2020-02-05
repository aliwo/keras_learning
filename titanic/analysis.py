from collections import Counter
import pandas

data = pandas.read_csv('./data/train.csv')
hi = list(data['Ticket'])

# Ticket 에 중복 값이 존재하긴 한다. 그러나 이게 죽고 사는 거랑 무슨 상관이지.
# print(Counter(hi).most_common()) # [('347082', 7), ('1601', 7), ('CA. 2343', 7), ('3101295', 6), ('CA 2144', 6), ...

# print((data[data['Ticket'] == '347082']).to_string())
# Andersson 가문의 온 가족이 티켓 347082 로 승선해서 싹 죽었나 본데...
# 하지만 죽을때 가족 단위로만 싹 죽는다고 가정할 수 도 없고...
# 'Andersson'은 다 죽었다 를 신경망에 알릴 방법을 찾아야 하나?
# 이름도 죽음 예측을 할 떄 필요한 정보란 말인가?


print((data[data['Ticket'] == '1601']).to_string())
# 이름으로 봐서 싹다 중국인인거 같은데, 전부 30대 초반쯤의 남자. 2명 죽고 나머진 살았다.

# print((data[data['Ticket'] == 'CA. 2343']).to_string())
# Sage 가문. 싹 다 죽었다. 이름의 첫 부분으로 클러스터링하는 방법을 찾아볼까...


# print(len(hi))
# print(len(set(hi)))
