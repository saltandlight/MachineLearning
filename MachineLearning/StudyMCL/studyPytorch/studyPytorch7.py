import torch
import math

dtype = torch.float
device = torch.device("cpu")

class Polynomial3(torch.nn.Module):
    def __init__(self):
        # 생성자에서 4개의 매개변수 생성, 멤버 변수로 지정

        super().__init__()
        self.a = torch.nn.Parameter(torch.randn((), device=device, dtype=dtype, requires_grad=True))
        self.b = torch.nn.Parameter(torch.randn((), device=device, dtype=dtype, requires_grad=True))
        self.c = torch.nn.Parameter(torch.randn((), device=device, dtype=dtype, requires_grad=True))
        self.d = torch.nn.Parameter(torch.randn((), device=device, dtype=dtype, requires_grad=True))

    def forward(self, x):
        # 순전파 함수에서는 입력 데이터의 텐서 받고 출력 데이터의 텐서 반환
        # 텐서들 간의 임의의 연산 말고도 생성자에서 정의한 모듈 사용 가능
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        # Tostring 같은 역할..?
        return f'y = {self.a.item()} + {self.b.item()} x+ {self.c.item()} x^2 + {self.d.item()} x^3'

# 입력값과 출력값을 갖는 텐서들 생성
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 모델 생성
model = Polynomial3()

# 손실 함수와 optimizer 생성. SGD 생성자에 model.parameters()를 호출해주면
# 모델의 멤버 학습 가능한 매개변수들이 포함됨
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
    # 순전파 단계: 모델에 x를 전달해서 예측값 y 계산
    y_pred = model(x)

    # 손실 계산하고 출력
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')

# 참조: https://tutorials.pytorch.kr/beginner/nn_tutorial.html