# [TensorFlow] Lab-06-2 Softmax Classifier (fancy version): Animal classification

## 목차

- Softmax function
- Softmax_cross_entropy_with_lgits
- Sample Dataset
- tf.one_hot_and_reshape
- Implementation

## Softmax_cross_entropy_with_logits(logits이 활용됨)

```python 
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
```

1.

```python
# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
```

2.

```python
# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y_one_hot)

cost = tf.reduce_mean(cost_i)
```

## Implementation

```python
xy = np.loadtxt('data-04-zoo.csv', delimeter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7

# Make Y data as onehot shape
Y_one_hot = tf.one_hot(list(y_data), nb_r)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

#Weight and bias setting
W = tfe.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tfe.Variable(tf.random_normal([nb_classes]), name='bias')
variables = [W, b]

# tf.nn.softmax computes softmax activations
def logit_fn(X):
    return tf.matmul(X, W) + b

def hypothesis(X):
    return tf.nn.softmax(logit_fn(X))
	# 정확도 맞추기 위해 활용됨
    
def cost_fn(X, Y):
    logits = logit_fn(X)
    cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
    # logit 입력 받기 위함
    
    cost = tf.reduce_mean(cost_i)
    return cost

def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        loss = cost_fn(X, Y)
        grads = tape.gradient(loss, variables)
        return grads
    
def prediction(X, Y):
    pred = tf.argmax(hypothesis(X), 1)
    correct_prediction = tf.equal(pred, tf.argmax(Y, 1))
    # 예측값과 Y이 일치하는 지 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #  얼마나 맞았는지 평균
    return accuracy

# 정답 잘 구했는지 확인 위해 만든 함수
```

### Implementation-Training

```python
def fit(X, Y, epochs=500, verbose=50):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    
    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i==0) | ((i+1)%verbose==0):
            acc = prediction(X, Y).numpy()
            loss = tf.reduce_sum(cost_fn(X, Y)).numpy()
            
            print('Loss & Acc at {} epoch {}, {}'.format(i+1, loss, acc))
            
fit(x_data, Y_one_hot)
```



