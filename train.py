import cv2
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 訓練圖片資料夾與模型名稱
imagedir = "data"
modelname = "carplate_model.hdf5"

# 儲存所有圖片資料與標籤
data = []
labels = []

# 逐一讀取資料夾中所有圖片
for image_file in paths.list_images(imagedir):
    image = cv2.imread(image_file)                     # 讀取圖片
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # 轉為灰階
    label = image_file.split(os.path.sep)[-2]          # 資料夾名稱即為標籤
    data.append(image)
    labels.append(label)

# 將資料轉為 numpy 陣列
data = np.array(data)
labels = np.array(labels)

# 將資料分為訓練組與測試組（15% 測試）
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.15, random_state=0)

# 正規化圖片數據：reshape 成 CNN 輸入格式 (38x18x1)，並將像素轉為 0~1 的浮點數
X_train_normalize = X_train.reshape(X_train.shape[0], 38, 18, 1).astype("float") / 255.0
X_test_normalize  = X_test.reshape(X_test.shape[0], 38, 18, 1).astype("float") / 255.0

# 將文字標籤轉換為 One-Hot 向量（如 'A' → [0,0,...,1,...,0]）
lb = LabelBinarizer().fit(Y_train)
Y_train_OneHot = lb.transform(Y_train)
Y_test_OneHot  = lb.transform(Y_test)

# 建立 CNN 模型架構
model = Sequential()

# 第一層卷積層：20 個 5x5 卷積核，使用 ReLU 激活函數，padding 保持輸入大小
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(38, 18, 1), activation="relu"))
# 最大池化層：2x2 池化核，步長為 2
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 第二層卷積層：50 個 5x5 卷積核，ReLU 激活
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 攤平成一維資料以供全連接層處理
model.add(Flatten())
# 隱藏層：500 個神經元，使用 ReLU
model.add(Dense(500, activation="relu"))
# 輸出層：34 個類別（A~Z + 0~9，不含 I、O），使用 softmax 分類
model.add(Dense(34, activation="softmax"))

# 編譯模型：使用交叉熵作為損失函數，Adam 最佳化器
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 開始訓練模型：80% 為訓練，20% 為驗證集，每批 32 張，共訓練 10 輪
model.fit(X_train_normalize, Y_train_OneHot, validation_split=0.2, batch_size=32, epochs=10, verbose=1)

# 儲存訓練完成的模型
model.save(modelname)

# 評估訓練集準確率
scores = model.evaluate(X_train_normalize, Y_train_OneHot)
print(scores[1])  # 顯示訓練集準確率

# 評估測試集準確率
scores2 = model.evaluate(X_test_normalize, Y_test_OneHot)
print(scores2[1])  # 顯示測試集準確率
