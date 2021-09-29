# [TensorFlow] Lab10-1 Relu

## Relu activation function

- Problem of Sigmoid
- Why Relu?
- Code
  - load dataset
  - create network
  - define loss function
  - experiments
    - parameters
    - model
    - eager mode
- What's Next

### Problem of Sigmoid

Input --> Network --> output

​										ground-truth - output = loss

​										d(loss) = gradient

<================================Backpropagation

<img src="pic/relu_1.png" style="zoom:50%;" />

- 사각형 부근의 그래프 접선의 기울기는 0보다 매우 큼
- 극단 좌표계쪽은 0에 접선의 기울기가 가깝다
- gradient 전달 받아서 학습하는데 gradient가 매우 작으면 안 됨
- 만약 network가 딥하면 ... 시그모이드가 여러 개 있을 거고 곱해질 때 gradient가 9에 가까워서 전달받을 gradient가 없어질 수 있음(**Vanishing Gradient**)
- 이게 바로 Sigmoid의 문제점

### Why Relu?

- f(x) = max(0, x)
  - x가 0보다 큰 양수값을 가지면 x를 추출해서 output으로 만들어라
  - x가 0보다 작은 값을 가지면 그대로 0을 추출해서 output으로 만들어라

- <img src="pic/relu_2.png" style="zoom:67%;" />
  - 0보다 클 때는 gradient가 1
  - gradient가 잘 전달이 됨
  - 그러나 0보다 작은 음수의 값일 때, gradient가 0(아예 전달이 안 됨, 이건 문제점)
  - 그래도 이걸 사용하는 이유는 성능이 좋기 때문임
  - tf.keras.activations ---------> sigmoid, tanh, relu, elu, selu
  - 문제점 해결 위해 leaky relu 사용하기도 하는데 keras의 layers쪽에 있다.
    - tf.keras.layers -----------> leaky relu
  - leaky relu: 0보다 작은 음수 값 가질 때 0.01 과 값 곱해서 역전파하는 방식임

## Code

### Load mnist(데이터셋)

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to categorical
from tensorflow.keras.datasets import mnist # fashion_mnist, cifar10, cifar100
tf.enable_eager_execution() # eager 모드로 
```

