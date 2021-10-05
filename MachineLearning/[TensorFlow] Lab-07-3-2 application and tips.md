# [TensorFlow] Lab-07-3-2 application and tips

## Sample Data

**Fashion MNIST-Image Classification**

```python
# Tensorflow Code
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 정규화
train_images = train_images / 255.0 # (60000, 28, 28)
test_images = test_images / 255.0 # (10000, 28, 28)

model = keras.Sequential([
    # 모델 펴줌
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 만든 모델 컴파일(optimizer, loss 설정, 정확도 어떻게 측정할 건지 정하여 모델 훈련)
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrixs=['accuracy'])

# 모델 훈련
model.fit(train_images, train_labels, epochs=5)
# 테스트 데이터 넣어서 test_loss, test_acc 출력
test.loss, test_acc = model.evaluate(test_images, test_labels)
# 새로운 데이터 넣어봄
predictions = model.predict(test_images)
np.argmax(predctions[0]) # 9 label
```

**IMDB-Text Classficiation**

- 영화평 잘 분류하기 위한 데이터들

- [데이터들]

- | entence                                                      | pos/neg |
  | ------------------------------------------------------------ | ------- |
  | worst mistake of my life br br i picked this movie up at target for 5 | 0(neg)  |
  | this firm was just brilliant casting location scenery story direction | 1(pos)  |
  | this has to be one of the worst films of the 1990s           | 0(neg)  |

```python
# Tensorflow Code
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
# The first indices are reserved , 전처리작업
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0 # 공백 벡터 선언
word_index["<START>"] = 1 # 시작값
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3 # 사용되지 않는 값
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 어떻게 학습할 것인지
model.fit(partial_x_train, partial_y_train, epochs=40, validation_data=(x_val, y_val))
```

**CIFAR-100**

```python
# Tensorflow Code
from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cfar100.load_data(label_mode='fine')
```

## Summary

- Data sets
  - Training(학습)/Validation(평가)/Testing
    - 여러 hyper parameter 들을 변경 -> 원하는 모델(99% 목표) 만들어낼 수 있음
  - Evaluating a hypothesis(가설 자체를 잘 평가해야 함)
    - Testing 통해 검증과정 거침
  - Anomaly Detection
    - 정상 데이터 학습 -> 특이 데이터 감지
      - 위조지폐분별법과 유사
- Learning
  - Online Learning vs Batch Learning
  - Fine tuning  / Feature Extraction
    - 백인 구분은 잘 작동, 황인 구분은 잘 안 되어서
    - Fine tuning: 기존 만들어진 모델에서 새로운 데이터 넣어서 미세하게 weight값 조정
    - Feature Extraction: 기존 모델 갖고 와서 특징만 뽑아와서 새로운 레이어 만들어서 학습하거나 거리만 측정하는 방법도 있음
  - Efficient Models:
    - 경량화된 모델이 중요함
    - 기존 모델 자체를 one by one colvolution 통해 경량화함
- Sample Data
  - Fasion MNIST / IMDB / CIFAR-100

