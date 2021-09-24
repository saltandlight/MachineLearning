# 04. Multi-variable linear regression LAB

## Hypothesis using matrix

- H(x1, x2, x3) = w1x1 + w2x2 + w3x3

- | x1   | x2   | x3   | y    |
  | ---- | ---- | ---- | ---- |
  | 73   | 80   | 75   | 152  |
  | 93   | 88   | 93   | 185  |
  | 89   | 91   | 90   | 180  |
  | 96   | 98   | 100  | 196  |
  | 73   | 66   | 70   | 142  |

- ```python
  # data and Label
  x1 = [ 73., 93., 89., 96., 73.]
  x2 = [ 80., 88., 91., 98., 66.]
  x3 = [ 75., 93., 90., 100., 70.]
  Y  = [152., 185., 180., 196., 142.]
  
  # weights
  w1 = tf.Variable(10.)
  w2 = tf.Variable(10.)
  w3 = tf.Variable(10.)
  b  = tf.Variable(10.)
  
  hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
  ```

- [전체 실행코드]

  ```python
  # data and Label
  x1 = [ 73., 93., 89., 96., 73.]
  x2 = [ 80., 88., 91., 98., 66.]
  x3 = [ 75., 93., 90., 100., 70.]
  Y  = [152., 185., 180., 196., 142.]
  
  # random weights
  w1 = tf.Variable(tf.random_normal([1]))
  w2 = tf.Variable(tf.random_normal([1]))
  w3 = tf.Variable(tf.random_normal([1]))
  b  = tf.Variable(tf.random_normal([1]))
  
  learning_rate = 0.000001
  
  for i in range(1000+1):
      # tf.GradientTape() to record the gradient of the cost function
      with tf.GradientTape() as tape:
          hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
          cost = tf.reduce_mean(tf.square(hypothesis - Y))
      # 오차 제곱 평균
      # calculates the gradients of the cost(각각의 기울기값 구하여 할당)
      w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])
      
      #update w1, w2, w3 and b
      w1.assign_sub(learning_rate * w1_grad)
      w2.assign_sub(learning_rate * w2_grad)
      w3.assign_sub(learning_rate * w3_grad)
      b.assign_sub(learning_rate * b_grad)
      
      if i % 50 == 0:
          print("{:5} | {:12.4f}".format(i, cost.numpy()))
  ```

## Matrix를 사용한 경우

``` python
data = np.array([
    # X1 , X2 , X3 , y
    [ 73., 80., 75., 152. ],
    [ 93., 88., 93., 185. ],
    [ 89., 91., 90., 180. ],
    [ 96., 98., 100.,196. ],
    [ 73., 66., 70., 142. ]
], dtype=np.float32)

#slice data
X = data[:, :-1] #: 처음부터 앞까지   :-1   첫컬럼부터 마지막 컬럼 제외
y = data[:, [-1]] # [-1] 마지막 컬럼을 뜻함

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

#hypothesis, prediction function
def predict(X):
    return tf.matmul(X, W) + b
```

[전체 코드]

```python
data = np.array([
    # X1 , X2 , X3 , y
    [ 73., 80., 75., 152. ],
    [ 93., 88., 93., 185. ],
    [ 89., 91., 90., 180. ],
    [ 96., 98., 100.,196. ],
    [ 73., 66., 70., 142. ]
], dtype=np.float32)

#slice data
X = data[:, :-1] #: 처음부터 앞까지   :-1   첫컬럼부터 마지막 컬럼 제외
y = data[:, [-1]] # [-1] 마지막 컬럼을 뜻함

W = tf.Variable(tf.random_normal([3, 1]))
# X의 컬럼 갯수 뜻함, 출력은 한 개니까 W는 3X1 행렬이 됨
b = tf.Variable(tf.random_normal([1]))

learning_rate = 0.000001

#hypothesis, prediction function
def predict(X):
    return tf.matmul(X, W) + b

n_epochs = 2000
for i in range(n_epochs + 1):
    # record the gradient of the cost function
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X) - y)))
        
    # calculates the gradients of the loss
    W_grad, b_grad = tape.gradient(cost, [W, b])
    
    # updates parameters ( W and b)
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    
    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))
```

- Matrix 사용 안 하면 변수를 다 설정해야 하고 hypothesis가 굉장히 복잡해짐
- Matrix 사용이 성능 면에서도 간편함에서도 상당히 유리함

