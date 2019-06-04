import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 지난 번에 학습했던 모델을 불러오기
model = keras.models.load_model('MNIST_CNN_model.h5')
model.summary()

# 누스타 5 테스트
test_num = plt.imread('./my_num/nustar_five.png')
test_num = test_num[:,:,0]
test_num = (test_num > 125) * test_num
test_num = test_num.astype('float32') / 255.

plt.imshow(test_num, cmap='Greys', interpolation='nearest');

test_num = test_num.reshape((1, 28, 28, 1))

print('The Answer is ', model.predict_classes(test_num))
