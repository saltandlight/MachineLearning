import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets

digits_data = datasets.load_digits()

n_img = 10
plt.figure(figsize=(10, 4))
for i in range(n_img):
    # 입력 이미지
    ax = plt.subplot(2, 5, i+1)
    plt.imshow(digits_data.data[i].reshape(8, 8), cmap="Greys_r")
    ax.get_xaxis().set_visible(False) # 축 표시 안 함
    ax.get_yaxis().set_visible(False)
plt.show()

print("데이터 형태:", digits_data.data.shape)
print("레이블:", digits_data.target[:n_img])

input_data = np.asarray(digits_data.data)
input_data = (input_data - np.average(input_data)) / np.std(input_data)

correct = np.asarray(digits_data.target)
correct_data = np.zeros(len(correct), correct)
for i in range(len(correct)):
    correct_data[i, correct[i]] = 1 # 원핫인코딩

x_train, x_test, t_train, t_test = train_test_split(input_data, correct_data)