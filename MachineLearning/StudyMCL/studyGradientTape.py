import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2

# dy = 2x * dx
dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())

# tf.Varables 을 감시하는 기본 동작 비활성화 목적 -> Gradient Tape 만들 때 watch_accessed_variables=False 설정
# 변수 중 하나의 그래디언트만 연결

x0 = tf.Variable(0.0)
x1 = tf.Variable(10.0)

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(x1)
    y0 = tf.math.sin(x0)
    y1 = tf.nn.softplus(x1)
    y = y0 + y1
    ys = tf.reduce_sum(y)

    grad = tape.gradient(ys, {'x':x0, 'x1':x1 })

print('dy/dx0:', grad['x0'])
print('dy/dx1:', grad['x1'].numpy())
