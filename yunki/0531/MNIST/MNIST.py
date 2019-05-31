# import: 텐서플로 중에 케라스 모듈을 사용한다.
from tensorflow import keras

# load Data : 구글에서 데이터셋을 다운로드 해서 메모리에 불러온다.
# x_data, y_data 들이 데이터들이고 test에는 우리가 모르는 값들이 들어간다.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 어떠한 데이터인지 알아본다. : 출력해보면 구글에서 6만개의 데이터셋을 가져온다.
print(y_train.shape)

# 앞에 5개만 출력해보자.
for i in range(5):
    print(y_train[i])

# 이제 숫자를 인식시켜보자.
# 그런데 뉴럴넷은 숫자를 인식하지 못 해서 이걸 리스트로 바꿔서 줘야하는데.
# 0부터 9까지 총 9개의 카데고리가 있으므로 0이면 리스트의 0번째 원소에 1을
# 넣고 나머지 9개에는 0을 넣는다.

# 즉 아래와 같이
# 0은 [1,0,0,0,0,0,0,0,0,0] 로 나타내야하고
# 1은 [0,1,0,0,0,0,0,0,0,0] 로 나타낸다.
# 그렇다면 5는? [ 0,0,0,0,0,1,0,0,0,0 ] 가 되겠지?

# 하지만, 케라스에는 이게 유틸로써 존재하고 있다. 다음과 같이~
# 카테고리를 10개로 할 것이고 y_train 을 10개로 나눠라라는 뜻이다.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

for i in range(5):
    print(y_train[i])

# 사실 이 데이터들은 28*28 사이즈의 배열로 된 픽셀이미지다.
# 이제 x 데이터를 뉴럴넷이 읽을 수 있게 하나의 리스트로 바꿔주자.
print(x_train.shape, x_test.shape)

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

# 아래가 바로 원하는 데이터의 형태다
print(x_train.shape, x_test.shape)

#####################################
# 학습시키기
#####################################
model = keras.Sequential()
model.add(keras.layers.Dense(32, activation="sigmoid", input_shape=(28*28, )))
model.add(keras.layers.Dense(32, activation="sigmoid"))
model.add(keras.layers.Dense(10, activation="sigmoid"))

optimizer = keras.optimizers.SGD(lr=0.1)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

model.summary()

# 정확도 검사
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# 학습을 더 시켜서 정확도를 올려보기
model.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# 그러나 이러한 방식보다는 좋은 데이터 셋을 사용하는 게 좋다.
