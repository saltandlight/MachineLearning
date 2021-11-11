import torch
import math

dtype = torch.float
device = torch.device("cpu")

# 입력값과 출력값 갖는 텐서들 생성
# requires_grad=False 가 기본값으로 설정, 역전파 단계 중에 변화도 계산 필요 없듬
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 가중치 갖는 임의의 텐서 생성, 3차 다항식 => 4개의 가중치 필요
# y = a + b x + c x^2 + d x^3
# requires_grad = True로 설정 -> 역전파 단계 중에 이 텐서들에 대한 변화도 계산 필요 있음을 나타냄
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    # 순전파 단계: 텐서들 간의 연산 사용하여 예측값 y를 계산
    y_pred = a + b * x + c * x **2 + d * x ** 3

    # 텐서들간의 연산 사용하여 손실 계산하고 출력
    # 이 때 손실은 (1,) shape 갖는 텐서임
    # loss.item()으로 손실이 갖고 있는 스칼라 값 가져올 수 있음
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # autograd 사용하여 역전파 단계 계산, requires_grad=True 갖는 모든 텐서들에 대한 손실의 변화도 계산
    # 이후 a.grad와 b.grad, c.gard, d.grad 는 각각 a,b,c,d에 대한 손실의 변화도 갖는 텐서가 됨
    loss.backward()

    # 경사하강법을 사용하여 가중치 직접 갱신
    # torch.no_grad()로 감싸는 이유: 가중치들이 requires_grad = True지만
    # autograd에서는 이를 추적 안 할 것이기 때문임
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # 가중치 갱신 후에는 변화도를 직접 0으로 만듬
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
# 참조: https://tutorials.pytorch.kr/beginner/nn_tutorial.html
