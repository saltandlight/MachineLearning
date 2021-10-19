import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
# tf.enable_eager_execution()

def load_mnist() :
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    train_data = np.expand_dims(train_data, axis = -1) # 출력해서 확인해보기
    test_data = np.expand_dims(test_data, axis = -1)

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # print(train_data)
    return train_data, train_labels, test_data, test_labels

def normalize(train_data, test_data) :
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0

    return train_data, test_data

def flatten() :
    return tf.keras.layers.Flatten()

def dense(channel, weight_init) :
    return tf.keras.layers.Dense(units=channel, use_bias=True, kernel_initializer=weight_init)

def relu() :
    return tf.keras.layers.Activation(tf.keras.activations.relu)

def dropout(rate) :
    return tf.keras.layers.Dropout(rate)

def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return loss

def accuracy_fn(model, images, labels):
    logits = model(images, training=True)
    prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy

def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)



class create_model(tf.keras.Model) :
    def __init__(self, label_dim):
        super(create_model, self).__init__()

        weight_init = tf.keras.initializers.glorot_uniform() # weight 초기화
        self.model = tf.keras.Sequential() # 선형 모델 만듦

        self.model.add(flatten()) # 차원 낮춰줌

        for i in range(2):
            self.model.add(dense(256, weight_init))
            self.model.add(relu())

        self.model.add(dense(label_dim, weight_init))

    def call(self, x, training=None, mask=None):
        x = self.model(x)

        return x

    def create_model(label_dim) :
        weight_init = tf.keras.initializers.glorot_uniform()

        model = tf.keras.Sequential()
        model.add(flatten())

        for i in range(2) :
            model.add(dense(256, weight_init))
            model.add(relu())
            model.add(dropout(rate=0.5))

        model.add(dense(label_dim, weight_init))

        return
