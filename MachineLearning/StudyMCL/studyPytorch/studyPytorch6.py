import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 입력 텐서 (x, x^2, x^3) 를 준비함
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# nn 패키지를 사용해서 모델과 손실 함수 정의
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# optim 패키지를 사용해서 모델의 가중치 갱신할 optimizer 정의
# RMSprop 사용; optim 패키지는 다른 다양한 최적화 알고리즘 포함함
# RMSprop 생성자의 첫 번째 인자는 어떤 텐서가 갱신되어야 하는지 알려줌
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(2000):
    # 순전파 단계: 모델에 x를 전달해서 예측값 y를 계산
    y_pred = model(xx)

    # 손실 계산하고 출력
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 역전파 단계 전에, optimizer 객체 사용해서 모델의 학습 가능한 가중치인 갱신할
    # 변수들에 대한 모든 변화도(gradient)를 0으로 만듬.
    # 이유: 이렇게 안 하면 .backward() 호출 시마다 변화도가 버퍼에 누적되서
    optimizer.zero_grad()

    # 역전파 단계: 모델의 매개변수들에 대한 손실 변화도 계산
    loss.backward()
    
    # optimizer의 step 함수 호출하면 매개변수 갱신됨
    optimizer.step()

linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + '
      f'{linear_layer.weight[:, 2].item()} x^3')
# 참조: https://tutorials.pytorch.kr/beginner/nn_tutorial.html
