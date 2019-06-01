#iport
from tensorflow import keras
import numpy as np

#data sets(x 를 주면 y값이 나와야함)
x_data = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

y_data = [
    [0],
    [1],
    [1],
    [0]
]

# numpy array 로 변환
x_data = np.array(x_data)
y_data = np.array(y_data)

# 시퀀셜 모델 생성: 활성화 함수는 시그모이드를 사용한다.
model = keras.Sequential()
model.add(keras.layers.Dense(8, activation="sigmoid", input_shape=(2,))) # input 레이어는 8개의 뉴런을 사용
model.add(keras.layers.Dense(1, activation="sigmoid")) # 아웃풋은 1개의 뉴런을 통과

# SGD optimizer 생성 후 모델에 적용
optimizer = keras.optimizers.SGD(lr=0.1)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])
model.summary()

# 학습시키기 : x,y, 데이터를 가지고 학습을 시킬 건데.4개 단위로 3500번 학습 시킬 것임.
# 3500번 정도 학습 시켰더니, 오차율이 0.1 정도가 되었기 때문에. 이정도는 학습이 된 걸로 친다고 함.
model.fit(x_data, y_data, batch_size=4,epochs=3500)

# 정확도 계산 : 학습 된 정보는 근사값이기 때문에 반올림해서 보정해주자.
predict = model.predict(x_data)
print(np.round(predict)) # 0, 1, 1 , 0 으로 xor 회로 학습을 완료하였다.