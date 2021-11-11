# R- CNN

1. selective search -> proposed regions(잘라낸 이미지들)
2. proposed regions -> pretained CNN -> feature map
   - 잘라낸 이미지들을 학습된 CNN에 넣어서 feature map을 추출
3. feature map을 SVM Classifier가 객체로 분류 시
   - 같은 feature map을 regressor에 넣어 예상되는 객체의 좌표 정보를 얻음

- 문제점: proposed regions에 대한 label이 없음
  - 객체에 대한 ground truth만 달려있고 proposed regions에 대한 label이 없음
  - 잘려진 이미지를 SVM Classifier와 regressor에 넣어 학습시켜야 하지만 기준이 되는 label이 없음



=> 그래서  **IOU**를 사용함

- R-CNN에서는 ground-truth와 proposed region 사이의 IOU 값이 thr보다 높은 경우,

  해당 region을 객체로 인식

- ground-truth와 같은 class로 labelling함

- 이 정보를 모델 학습에 이용함

=> R-CNN에서는 IoU가 라벨링 과정에서 핵심적인 역할을 함



## NMS(Non Maximum Suppression)

- object detection 예측 결과가 겹치는 형태로 나타났을 때, score가 가장 높은 bounding box 한 개만 남도록 후처리함
  - 겹치는 영역 확인할 때 IoU가 사용됨