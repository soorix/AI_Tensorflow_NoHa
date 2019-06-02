# coding: utf-8

import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 기본 활성화 함수
def sigmoid(x):
    return 1/(1+np.exp(-x))

#항등함수
def identity_function(x):
    return x


################ basic 3층 신경망 구현 부분 ######################

#각 층의 가중치(W)와 편향(B)을 초기화
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4],[0.2, 0.5],[0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

#각 층의 입력신호를 출력신호로 변환하는 factor 정의
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)+b3
    y = identity_function(a3)
    
    return y

################ basic 3층 신경망 구현 부분 ######################


# 3층 신경망 사용
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)

print(y)



