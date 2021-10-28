import cv2
import glob
import numpy as np

# Yolo 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

imgPathes = glob.glob('D://Shared//ConvertedData//car_images//*')
for imgPath in imgPathes:
    # 이미지 가져오기
    img = cv2.imread(imgPath)
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape

    # Detecting objects
    # 이미지를 바로 신경망으로 넘겨주지 않고 blobFromImage 함수 사용해서 이미지 가지고 4차원의 blob 만들어서 넘겨줌
    # 신경망에서 이미지를 바로 사용할 수 없으므로 blob으로 넘겨준다.
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 정보를 화면에 표시
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # confidence: 신뢰도, 1로 갈수록 인식 정확도가 높아짐, 0.5 넘기면 인식했다고 간주함
            if confidence > 0.5:
                # Object Detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 같은 객체에 여러 박스 생길 수 있으므로 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 정보를 화면에 표시
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 참조: https://blog.naver.com/PostView.naver?blogId=topblade71&logNo=222067237754&parentCategoryNo=&categoryNo=17&viewDate=&isShowPopularPosts=false&from=postView