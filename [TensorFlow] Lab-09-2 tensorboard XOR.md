# [TensorFlow] Lab-09-2 tensorboard XOR

- tensorboard: weight값이나 b를 시각화하는 도구

## Tensorboard for XOR NN

```python 
pip install tensorboard
tensorboard -logdir=./log/xor_logs
```

```python
# [Eager Execution] 모든 tensorboard 저장 위해 contrib 밑에 선언해야 함
writer = tf.contrib.summary.FileWriter("./log/xor_logs")
with tf.contrib.summary.record_summaries_every_n_global_steps(1):
    tf.contrib.summary.scalar('loss', cost)
    
# [Keras] callbacks라는 곳 아래에 Tensorboard 선언
# 실제 모델 학습 과정에서 tb_hist 지정함
tb_hist = tf.keras.callbacks.TensorBoard(log_dir="./logs/xor_logs", histogram_freq=0, write_graph=True, write_images=True)
model.fit(x_data, y_data, epochs=5000, callbacks=[tb_hist])
```

```python
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

# 실제 테스트할 데이터(XOR 관련)
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
         [1],
         [1],
         [0]]

dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))

def preprocess_data(features, labels):
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.float32)
    return features, labels

W1 = tf.Variable(tf.random_normal([2,1]), name='weight1')
b1 = tf.Variable(tf.random_normal([1]), name='bias1')

W4 = tf.Variable(tf.random_normal([2, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')

def neural_net(features):
    layer1 = tf.sigmoid(tf.matmul(features, W1) + b1)
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
    hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
    
	with tf contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.histogram("weights1", W1)
        tf.contrib.summary.histogram("biases1", b1)
        tf.contrib.summary.histogram("layer1", layer1)
        					...
        tf.contrib.summary.histogram("weights3", W3)
        tf.contrib.summary.histogram("biases3", b3)
        tf.contrib.summary.histogram("layer3", layer3)
        tf.contrib.summary.histogram("weights4", W4)
        tf.contrib.summary.histogram("biases4", b4)
        tf.contrib.summary.histogram("hypothesis", hypothesis)
```

- IP:6006 접속하면 화면 확인 가능

```python
def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels*tf.log(hypothesis)+(1-labels)*tf.log(1-hypothesis))
    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('loss', cost) # cost를 loss에 담음
    return cost

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.float32))
    return accuracy

def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(neural_net(features), labels)
    return tape.gradient(loss_value, [W1, W2, W3, W4, b1, b2, b3, b4])

EPOCHS = 3000
log_path = "./log/xor_eager_logs_r0_01"
writer = tf.contrib.summary.create_file_writer(log_path)
global_step = tf.train.get_or_create_global_step()
writer.set_as_default()

for step in range(EPOCHS):
    global_step.assign_add(1)
    for features,labels in tfe.Iterator(dataset)
   		features, label = preprocess_data(features, labels)
    	grads = grad(neural_net(features), features, labels)
        optimizr.apply_gradients(grads_and_vars=zip(grads, [W1, W2, W3, W4, b1, b2, b3, b4]))
        if step % 50 == 0:
            loss_value = loss_fn(neural_net(features), labels)
            print("Iter: {} Loss: {: 4f}".format(step, loss_value))
x_data, y_data = preprocess_data(x_data, y_data)
test_acc = accuracy_fn(neural_net(x_data), y_data)
print("Testset Accuracy: {:.4f}".format(test_acc))
```

- [keras]

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# 실제 테스트할 데이터(XOR 관련)
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
         [1],
         [1],
         [0]]

# Keras 관련 모델 선언
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_dim=2, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
tb_hist = tf.keras.callbacks.TensorBoard(log_dir="./logs/xor_logs_r0_01", histogram_freq=0, write_graph=True, write_images=True)
model.fit(x_data, y_data, epochs=5000, callbacks=[tb_hist])

model.predict_classes(x_data)
```

