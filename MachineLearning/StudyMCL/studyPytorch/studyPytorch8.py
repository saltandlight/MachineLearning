import random
import torch
import math

dtype = torch.float
device = torch.device("cpu")

class DynamicNet(torch.nn.Module):
    def __init__(self):
        # 생성자에서 5개의 매개변수 생성, 멤버 변수로 지정
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn((), device=device, dtype=dtype, requires_grad=True))
        self.b = torch.nn.Parameter(torch.randn((), device=device, dtype=dtype, requires_grad=True))
        self.c = torch.nn.Parameter(torch.randn((), device=device, dtype=dtype, requires_grad=True))
        self.d = torch.nn.Parameter(torch.randn((), device=device, dtype=dtype, requires_grad=True))
        self.e = torch.nn.Parameter(torch.randn((), device=device, dtype=dtype, requires_grad=True))

    def forward(self, x):
        # 무작위로 4, 5 중 하나 선택해서매개변수 e 재사용 -> 이 차수들의 기여도 계산
        # 각 순전파 단계는 동적 연산 그래프 구성하므로 모델의 순전파 단계 정의 시 반복문이나 조건문 같은
        # 일반적인 Python 제어-흐름 연산자 사용 가능
        # 연산 그래프 정의 시 동일한 매개변수 여러 번 사용하는 것이 완벽히 안전함
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y

    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + ' \
               f'{self.e.item()} x^5 ?'

# 입력값과 출력값 갖는 텐서들 생성
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = DynamicNet()

# 손실 함수와 optimizer 생성. SGD 학습이 어려우므로 모멘텀 사용
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
for t in range(30000):
    # 순전파 단계: 모델에 x 전달해서 예측값 y 계산
    y_pred = model(x)

    # 손실 계산, 출력
    loss = criterion(y_pred, y)
    if t % 2000 == 1999:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')

# 참조: https://tutorials.pytorch.kr/beginner/nn_tutorial.html