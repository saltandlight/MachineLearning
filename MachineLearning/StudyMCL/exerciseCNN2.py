from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())

# NN 레이어 구현
# 인풋, 아웃풋 사이에 히든 레이어 둠
classifier(Dense(units=128, activation='relu'))

# 이진 분류 문제 -> 노드가 하나인 출력 레이어 구현
# 시그모이드 활성화 함수 사용
classifier.add(Dense(units=1, activation='sigmoid'))
# 옵티마이저: 경사하강 알고리즘 선택, 로스함수와 성능 매트릭 파라미터 각각 설정
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# keras.preprocessing 라이브러리 사용, 이미지 데이터 증폭
# 이미지의 레이블은 디렉터리 명 따름
# 'cats 이름 폴더 안에 있는 모든 이미지가 고양이로 학습됨..(이해 안 감)
train_datagen = ImageDataGenerator(rescale = 1./255
                                 , shear_range = 0.2
                                 , zoom_range=0.2
                                 , horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('training_set', target_size=(64, 64)
                                                               , batch_size=32
                                                               , class_mode='binary')

test_set = test_datagen.flow_from_directory('test_set'
                                           , target_size=(64, 64)
                                           , batch_size=32
                                           , class_mode='binary')
# steps_per_epoch: 훈련 횟수
# epochs: 모델을 학습시키는 단위
# 모든 데이터가 한번 훈련에 사용되면 한 epoch가 완료되었다고 함
# 모델 훈련은 하나 이상의 epoch가 실행되어야 함
# 이 경우 25번 실행
classifier.fit_generator(training_set
                       , steps_per_epoch=8000
                       , epochs=25
                       , validation_data=test_set
                       , validation_steps=2000)

test_image = image.load_img('..\\pic\\cat.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
# 참조; https://www.blog.spiderkim.com/post/%EB%94%A5%EB%9F%AC%EB%8B%9D-cnn-convolutional-neural-network-%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EA%B0%84%EB%8B%A8%ED%95%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B6%84%EB%A5%98-%EC%98%88%EC%A0%9C