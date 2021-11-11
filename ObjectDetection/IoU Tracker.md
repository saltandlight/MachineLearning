# IoU Tracker

## High-Speed Tracking-by-Detection Without Using Image Information

- 추적할 모든 object들에 대해 detector는 매 frame마다 detection 생성한다고 가정
- detection은 'gap'이 없거나 거의 없으며 충분히 높은 frame rate 사용 시 흔히 있는 경우인 상당히 중첩된 IoU 갖는다고 가정



- 위 두 요구사항 모두 충족 시 트래킹은 사소한 문제가 되고 이미지 정보 없이도 수행 가능

  - 특정 임계값 sigma_IOU 만족 시, 이전 frame 내 마지막 detection과 가장 높은 IOU 갖는 detection 연관시킴, track을 본질적으로 지속 가능한 단순한 IOU tracker 제안

  - 기존 track에 할당되지 않은 모든 detection은 새로운 트랙 시작, 할당된 detection이 없는 모든 트랙들은 종료됨



- 길이가 t_min 보다 짧은 모든 track들과 sigma_h 이상의 score(예측값) 가진 detection이 하나 이상 없는 track들 걸러냄 -> 성능 향상 가능
  - 짧은 track들은 보통 FP야기, 출력에 혼란 추가하므로 걸러내게 됨
  - track에 적어도 하나 이상의 높은 score의 detection 갖도록 요구, track의 완성도를 위한 낮은 score 의 detections의 이점을 가지는 동안 track이 True object interest에 속할 수 있음을 보장 가능



- 가장 잘  matching된 할당되지 않은 detection 만 track 확장하기 위한 후보로 간주됨
  - `best_match = max(_detections, key=lambda x: self.iou(t['rect'], x['rect']))`
  - detection D_f, tack T_a 사이의 최적의 연관 반드시 야기하는 건 아닌데 해결 가능(ex. 해당 frame에서 모든 IoU의 합을 최대화하는 Hungarian Algorithm 적용)
  - 보통 sigma_IoU는 detector 의 non maximum suppression 위한 IoU 임계값과 동일한 범위에서 선택됨 -> 잘 matching된 것 취하는 것은 합리적임
  - sigma_IoU 만족하는 다수의 matching된 것들은 실제로 거의 발생 안 함



- 이 방법의 복잡도는 다른 최신의 tracker 들과 비교했을 때 매우 낮음
  - frame 에 대한 시각적 정보 사용 안 됨 -> detection 수준에서 간단한 filtering 절차로서 간주 가능
  - 추가적인 계산 비용 없이 track 얻을 수 있음
  - tracker 단독 수행 시, 100K 초과하는 frame rates 달성 쉬움
  - 속도적인 이점으로 인해 출력을 이미지나 움직임 정보 사용해서 연결 가능한 tracklets으로 고려함 -> 더 많은 tracking 구성 요소 추가 가능

## Extending IOU Based Multi-Object Tracking by Visual Information

### IoU Based Object Detector

- Instance에 대한 정보를 IOU 기반으로 하는 트래커
- IoU 높다 = 인스턴스 간 겹치는 구간 많다

- 동영상 프레임에서 어떤 객체는 현재 위치에서 많이 벗어나있을 가능성이 낮음
  - IoU Object Tracker는 이런 점에 착안해서 현재 프레임에서 detection 된 객체는 다음 프레임에서 주변에 있을 것이라고 가정
  - 심플한 방법으로 좋은 성능 낼 수 있음

- 객체가 tracking되는 도중 detection 과정에서 false negative가 발생한 경우, fragment통한 id switch 문제 발생함

  - 이 문제 개선하고자 객체 추적이 끊기게 된 순간부터 일정 프레임동안 트래커의 정보를 detection에 반영하는 방법 적용

  - 일정 프레임 이상 detection 안 되서 추적 종료 시, 트래커는 마지막으로 fragment 된 장소에서 detection 종료된 것으로 파악

  - 그러나... 트래커가 갖고 있는 정보를 다시 detection에 반영하는 방법: 프레임이 늘어날수록 tracking에 참가하고 있는 객체 양이 증가한다는 문제가 있음

    IoU tracker의 근본적인 디자인 문제가 있으므로 문제의 완전한 해결은 어렵다

    

참고: https://ezobear.github.io/reatl-time,%20object%20tracking,%20mot/2020/09/01/MOT-post.html

https://neverabandon.tistory.com/16
