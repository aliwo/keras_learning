
# keras 에서 모델을 저장하려면 model.save() 메소드를 호출해 h5 형식으로 저장하면 된다.
# model.save('kimchi.h5')

# 불러올 때에는
# from keras.models import load_model
# model = load_model('kimchi.h5')

# 모델이 어떻게 생겨먹었는지 확인할 때에는
# model.summary()

## 모델 따로, 가중치 따로 저장할 수 있다.
# from models import model_from_json
# json_string = model.to_json() # 모델 아키텍처를 json 형식으로 저장
# model = model_from_json(json_string) # json 파일에서 모델 아키텍처 재구성
#
# from models import model_from_yaml
# yaml_string = model.to_yaml() # 모델 아키텍처를 yaml 형식으로 저장
# model = model_from_yaml(yaml_string) # yaml 파일에서 모델 아키텍처 재구성

# model.save_weights('kimchi_weight.h5') # 가중치만 따로 저장
# model.load_weights('kimchi_weight.h5') # 가중치 불러오기
