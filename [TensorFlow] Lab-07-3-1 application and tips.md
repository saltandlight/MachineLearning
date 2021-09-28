# [TensorFlow] Lab-07-3-1 application and tips

- 실제 데이터 구성방법, 학습 위한 여러 사례들 위주

## Application & Tips

- Data sets
  - Training / Validation / Testing
  - Evaluating a hypothesis
  - Anomaly Detection
- Learning
  - Online Learning vs Batch Learning(Offline-Learning)
  - Fine tuning(모델 재학습)
  - Efficient Models(모델 경량화)
- Sample Data(공개된 데이터들)
  - Fashion MNIST / IMDB / CIFAR-100
- Summary

## Data sets

- 데이터 구성이 모델에서 가장 중요
- 학습과 평가

**Good Case**

-  99%의 모델을 만드는 것이 목적

```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model.fit(x_train, y_train, validation_split=0.2, epochs=5) # 20% val data
```

**Evaluating a hypothesis**

- layer, learning_rate, optimizer 잘 선언, 결정 후에는 모델 잘 나오게 됨
- 어느정도 모델 선택된 후에는 새로운 데이터 만들어서 모델 테스트

- 새로운 테스트 데이터 넣었을 때도 잘 맞아야 함

```python
test_acc = accuracy_fn(softmax_fn(x_test), y_test)
model.evaluate(x_test, y_test) # Keras
```

### Anomaly detection(이상 감지)

- 특이한 데이터 감지하는 것
- 정상적인 데이터 학습을 확실하게 해서 특이한 데이터를 잘 감지해야 함

### Learning

**Online vs Batch**

|                  | Online Learning                              | Batch(Offline) Learning        |
| ---------------- | -------------------------------------------- | ------------------------------ |
| Data             | Fresh                                        | Static                         |
| Network          | connected                                    | disconnected                   |
| Model            | Updating(데이터유입 통한 업데이트)           | Static(정적 상태 유지)         |
| Weight           | Tunning                                      | initialize                     |
| Infra(GPU)       | Always(하드웨어 인프라 언제나 필요)          | Per call(요청 시에만 사용)     |
| Application      | Realtime Process(실시간 처리 환경 되어야 함) | Stopping(모델 바뀔 수 있게 함) |
| Priority(중요도) | Speed                                        | Correctness                    |

**Fine Tuning / Feature Extraction**

(a) Original Model

기존 얼굴 구분하는 모델

(b) Fine-tuning

새로운 데이터 넣어서 학습

weight값 미세 조절 -> 잘 구분해내게 하는 학습법

(c) Feature Extraction

기존 모델 잘 만들어놓고 새로운 task에 대해서만 학습

-> ex)황인, 백인 잘 구분해내는  것

```python
savor = tf.train.import_meta_graph('mv-model-1000.meta')
savor.restore(tf.train.latest_checkpoint('./'))
```

**Efficient Models**

- 실제 모델에 대한 퍼포먼스도 굉장히 중요함
- 100명을 구분하는 모델이 필요
- 인퍼런스 시간을 최소화하는 것이 중요, 모델의 weight를 경량화 하는 것 중요
  - fully connected layers 를 1X1 convolutions으로 대체하는 기법 많이 사용
  - (Squeezenet, Mobilenet)

- ```python
  tf.nn.depthwise_conv2d(input, filter, strides, padding)
  ```