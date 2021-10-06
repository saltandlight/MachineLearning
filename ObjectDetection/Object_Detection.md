# Object Detection

## 개요

- Obect Detection(객체 검출) = Classification + Localization
  - = Multi-labeled Classification + Bounding Box Regression
  - Classification: 물체 분류, Image Classification Task라고 하기도 함
  - Localization: 물체가 어디 있는지 Bounding Box 통해 위치 정보 나타냄, 아래 그림처럼 직사각형을 그려줌
- 컴퓨터 비전, 영상 처리와 관계가 깊음
- ex) YOLO, SSD, Mask RCNN, RetinaNet
- CCTV, 자율주행차 등등에 많이 쓰임
  - 자율주행차의 경우, Object Detection을 적용한다면 자동차는 사람처럼 '눈'을 갖게 됨
    - 이렇게 되면 사고가 날 위험성도 어느 정도 방지가 가능해짐
  - 군사목적 
  - 스포츠 분야
    - 게임에서 캐릭터 분석하듯이 능력치나 상태 파악이 가능해짐
  - 마트에서 물품 재고 파악에도 활용 가능
    - ex) 요플레가 29 개 남아있음
- ![](pic/bounding.PNG)
- 검은색 박스가 바운딩 박스이고, 이런 식으로 보통 인식함

## 1-stage Detector

- Localization(물체 위치 찾기) , Classification(물체 식별) 동시에 병행
- 빠르지만 정확도가 낮음
- ex) YOLO(You Look Only Once) 계열, SSD 계열
- 최근에는 YOLO4보다 빠른 YOLOR이 나옴

## 2-stage Detector

- Localization, Classification 을 순차적으로 행하는 방법
- 느리지만 정확도가 높음
- ex) R-CNN 계열

참고: 

https://nuggy875.tistory.com/20

https://89douner.tistory.com/75

https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-what-is-object-detection/

https://deepbaksuvision.github.io/Modu_ObjectDetection/posts/01_00_What_is_Object_Detection.html

사진 참고: 

https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/

