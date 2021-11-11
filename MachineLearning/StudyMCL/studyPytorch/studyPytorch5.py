import torch
import math

# 입력값과 출력값을 갖는 텐서들을 생성
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 출력 y는 (x, x^2, x^3)의 선형 함수, 선형 계층 신경망으로 간주 가능
# (x, x^2, x^3) 을 위한 텐서 준비
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# x.unsqueeze(-1)은 (2000, 1) 의 shape, p는 (3,)의 shape를 가짐
# 이 경우 브로드캐스트가 적용 (2000, 3)의 shape를 갖는 텐서 얻음

# nn 패키지 사용해서 모델을 순차적 계층으로 정의
# nn.Sequential은 다른 Module을 포함하는 Module로, 포함되는 Module들을 순차적으로 적용해서 출력 생성
# 각각의 Linear Module은 선형 함수(linear function)를 사용하여 입력으로부터 출력 계산
# 내부 tensor 에 가중치와 편향 저장(weight, bias)
# Flatten 계층은 선형 계층의 출력을 'y'의 shape과 맞도록 1D 텐서로 폄
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

# 또한 nn 패키지에서는 주로 사용되는 loss function들에 대한 정의도 포함되엉 ㅣㅆ음
# 여기에서는 평균 제곱 오차(MSE)를 손실함수로사용
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):
    # 순전파 단계: x를 모델에 전달해서 예측값 y를 계산. Module 객체는 __call__ 연산자를 오버라이드 함수처럼 호출 가능하게 함
    # 이렇게 해서 입력 데이터의 텐서를 Module에 전달, 출력 데이터의 텐서를 생성함
    y_pred = model(xx)

    # 손실을 계산하고 출력, 예측한 y와 정답인 y를 갖는 텐서들을 전달
    # 손실 함수는 손실(loss)을 갖는 텐서를 반환
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 역전파 단계를 실행하기 전에 변화도(gradient)를 0으로 만듬
    model.zero_grad()

    # 역전파 단계: 모델의 학습 가능한 모든 매개변수에 대해 손실의 변화도를 계산
    # 내부적으로 각 Module의 매개변수는 requires_grad=True일 때 텐서에 저장됨
    # 아래 호출은 모델의 모든 학습 가능한 매개변수의 변화도를 계산하게 됨
    loss.backward()

    # 경사하강법을 사용하여 가중치 갱신
    # 각 매개변수는 텐서, 이전에 했던 것처럼 변화도에 접근 가능
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# list의 첫번째 항목에 접근하는 것처럼 'model'의 첫 번째 계층(layer)에 접근 가능
linear_layer = model[0]

# 선형 계층에서 매개변수는 'weights'와 'bias'로 저장됨
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1]} x^2 + {linear_layer.weight[:, 2].item()} x^3')
# 참조: https://tutorials.pytorch.kr/beginner/nn_tutorial.html