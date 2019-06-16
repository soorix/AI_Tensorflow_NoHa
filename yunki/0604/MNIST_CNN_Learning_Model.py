#----------------------------------------import----------------------------------------
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

np.random.seed(7)

#--------------------------------------- constant ---------------------------------------
# 이미지의 크기 및 모양
img_rows = 28
img_cols = 28

batch_size = 128 # 뉴런의 개수
num_classes = 10 # 카데고리 개수
epochs = 12      # 학습 횟수

#------------------------------------ MNIST dataset ------------------------------------
# train : 6만개의 트레이닝 데이터
# test : 신경망(nn)이 지금까지 본 적 없는 1만개의 데이터, 학습검증용
# x : 학습데이터로, 28 * 28사이즈의 2차원 리스트 ( 이미지 )
# y : 정답으로, 0~9까지의 카테고리
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#------------------------------------데이터 정제------------------------------------------
# 신경망(nn)이 이해할 수 있게 변환해줘야만 한다.
# 1). x : 6만개의 2차원 배열로 된 인풋 이미지, 케라스의 nn 이 받아들일 수 있게 4차원 배열로 바꿔줘야 한다.  (=reshape)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# 2). y : x에 대한 답으로, 0~9까지의 숫자이다. nn 이 이해할 수 있도록 one-hot-encoding 으로 바꿔줘야한다. (=categorical)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes) 

# ---------------------------------------model---------------------------------------------
model = keras.Sequential()   #32->채널 수(이미지 장 수)

# Convolution : 이미지의 특징 검출, 데이터양이 크다.
# MaxPooling : 정해진 구간(ex. 2*x) 안에서 가장 큰 값만 남기고 나머지는 버리는 방식, 데이터양이 획기적으로 줄어든다.
# Layer 1
model.add(keras.layers.Conv2D(32, kernel_size=(5,5), strides=(1,1), activation='relu', padding='same', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# Layer 2
model.add(keras.layers.Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

# dropout: 레이어가 많아 발생할 수 있는 과적합을 방지하기 위해 학습데이터의 0.25% 는 버린다.
model.add(keras.layers.Dropout(0.25))
# flatten: 2차원 데이터를 1차원으로 바꾼다.
model.add(keras.layers.Flatten())

# fully connected node 인풋 뉴런들이 relu로 들어옴.
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_classes, activation='softmax'))

# Summary
model.summary()

#-------------------------------------Training----------------------------------------------
# 손실함수: categorical_crossentropy는 MNIST 처럼 카테고리를 나눌 때 사용하는 함수
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

#--------------------------------------검증--------------------------------------------------
# 인풋 데이터는 28*28의 그레이 스케일(Greys)로 되어있음.
# x_test 를 신경망에 학습 시킬 때 1차원으로 바꿨음으로 그림으로 뿌리기 위해 2차원으로 다시 바꿔줘야함.
n = 0
plt.imshow(x_test[n].reshape(img_rows, img_cols), cmap='Greys', interpolation='nearest')
plt.show()

print('휴먼.. 이 사진은 음... 제 생각엔 ', model.predict_classes(x_test[n].reshape(1,img_rows, img_cols, 1)), '인 것 같군요?!')

#--------------------------------------정확도 출력------------------------------------------
model.evaluate(x_test, y_test)
#--------------------------------------학습모델 저장----------------------------------------
model.save("MNIST_CNN_DATA.h5")