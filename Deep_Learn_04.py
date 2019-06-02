import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.append(os.pardir)
from mnist import load_mnist

# MNIST '2'일때의 데이터 (예시)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 ] #신경망의 출력(신경망이 추정한 값)(softmax function을 통한 확률)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] #정답 레이블

# 손실함수
# MSE : mean squared error (평균제곱오차)
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 사용상황 1 : '2'일 확률이 가장 높다고 추정
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t))) # 0.0975000...

# 사용상황 2 : '7'일 확률이 가장 높다고 추정
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t))) # 0.597500....


# CEE : cross entropy error  (교차엔트로피)
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

# 사용상황 1 : '2'일 확률이 가장 높다고 추정
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t))) # 0.510825...

# 사용상황 2 : '7'일 확률이 가장 높다고 추정
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t))) # 2.302584....


#mnist데이터셋에서 무작위로 데이터 읽어오기
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# normalize : 입력 이미지의 픽셀 값을 0.0 ~ 1.0 사이의 값으로 정규화
# flatten : True시 이미지 1차원 배열로, False시 이미지 1 * 28 * 28 의 3차원배열
# one_hot_label : [0,0,0,1,0,0,0,0,0,0] 과 같이 정답을 뜻하는 원소만 1인 (hot) 나머지는 모두 0인 배열의 형태로 저장

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

# 훈련데이터에서 무작위로 10장만 빼기
train_size = x_train.shape[0]

#미니배치 : 무작위로 선정한 표본데이터
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask] # 무작위로 10개만 추출한 훈련데이터
t_batch = t_train[batch_mask] # 무작위로 10개만 추출한 시험데이터

print(np.random.choice(60000,10))


# 미니배치용 교차 엔트로피
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7))/ batch_size

# 기울기
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원

    return grad


# 경사법(기울기를 이용해 함수의 최소값(최적해)을 찾는 방법
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    # f : 최적화 하려는 함수
    # init_x : 초기값
    # lr : learning_rate(학습률)
    # step_num : 경사법에 따른 반복 횟수

    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad
    return x


#경사법으로 f(x0,x1) = x0^2 + x1^2 의 최솟값 구하기
def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0]) #초기값
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
#결과 : array([-6.1110793e-18, 8.14814391e-10])

#학습률이 클때 발생하는 문제 : lr=10.0
init_x = np.array([-3.0, 4.0]) #초기값
print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))
#결과 : array([-2.58983747e+18, -1.29524862e+12]) 너무 큰값으로 발산

#학습률이 작을 때 발생하는 문제 : lr=1e-10
init_x = np.array([-3.0, 4.0]) #초기값
print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))
#결과 : array([-2.99999994, 3.99999992]) 갱신되지 않은 채 끌나버림

# ※ 하이퍼파라미터 : 학습률(lr)과 같이 컴퓨터가 아닌 사람이 직접 설정해야 하는 매개변수
# 여러 후보 값 중에서 시험을 통해 가장 잘 학습하는 값을 찾는 과정을 거쳐야 한다


# softmax function
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y



# 신경망에서 기울기 구현
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t) #손실함수 값

        return loss


net = simpleNet()
print(net.W) # 가중치 매개변수
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
np.argmax(p) #최댓값의 인덱스 '2'

t = np.array([0, 0, 1]) # 정답 레이블
net.loss(x, t)

# 신경망 학습의 절차

# 전체 : 신경망에는 적용 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응 하도록 조정하는 과정을 '학습'이라 한다.

# 1단계 - 미니배치
# 훈련 데이터 중 일부를 무작위로 가져온다. 이렇게 선별한 데이터를 '미니배치'라 한다.
# 미니배치의 손실함수 값을 줄이는 것이 목표

# 2단계 - 기울기산출
# 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구한다.
# 기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시한다

# 3단계 - 매개변수 갱신
# 가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다.

# 4단계 - 반복
# 1 ~ 3단계를 반복한다.


















