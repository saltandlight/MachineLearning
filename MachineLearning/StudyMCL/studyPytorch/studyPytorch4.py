import torch
import math

class LegendrePolynomial3(torch.autograd.Function):
    # torch.autograd.Function 상속받음
    # 사용자 정의 autograd Function 구현
    # 텐서 연산 하는 순전파 단계와 역전파 단계 구현

    @staticmethod
    def forward(ctx, input):
        # 순전파 단계에서는 입력 갖는 텐서 받아 출력 갖는 텐서 반환
        # ctx는 컨텍스트 객체로 역전파 연산 위한 정보 저장에 사용
        # ctx.save_for_backward 메소드 사용 -> 역전파 단계에서 사용할 어떤 객체도 저장해둘 수 있음
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        # 역전파 단계에서는 출력에 대한 손실(loss)과 변화도(grad)를 갖는 텐서 받고,
        # 입력에 대한 손실의 변화도 계산해야 함
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 -1)

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # GPU에서 실행하려면 이거 쓰기

# 입력값과 출력값 갖는 텐서들 생성
# requires_grad=False가 기본 설정 -> 역전파 단계 중에 이 텐서들에 대한 변화도 계산 필요 없음
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 가중치 갖는 임의의 텐서를 생성, 3개의 다항식 => 4개의 가중치 필요
# y = a + b * P3(c + d*x)
# 이 가중치들이 수렴하기 위해 정답으로부터 너무 멀리 떨어지지 않은 값으로 초기화되어야 함
# requires_grad=True로 설정 -> 역전파 단계 중에 이 텐서들에 대한 변화도 계산 필요가 있음
a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
for t in range(2000):
    # 사용자 정의 Function 적용 위해 Function.apply 메소드 사용
    # P3 이라고 이름 붙임
    P3 = LegendrePolynomial3.apply

    # 순전파 단계: 연산 하며 예측값 y 계산
    # 사용자 정의 autograd 연산 사용 -> P3 계산
    y_pred = a + b * P3(c + d * x)

    # 손실 계산하고 출력
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # autograd 사용하여 역전파 단계 계산
    loss.backward()

    # 경사하강법을 사용하여 가중치 갱신
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

print(f'Result: y = {a.item()} + {b.item()} * P3({c.item} + {d.item()} x)')

# 참조: https://tutorials.pytorch.kr/beginner/nn_tutorial.html