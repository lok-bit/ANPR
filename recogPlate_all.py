import os
import shutil
import time
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import glob

# 刪除並重建指定資料夾
def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
        time.sleep(2)
    os.mkdir(dirname)

# 所有可能車牌字元（不含 I、O）
labels = ['0','1','2','3','4','5','6','7','8','9',
          'A','B','C','D','E','F','G','H','J','K',
          'L','M','N','P','Q','R','S','T','U','V',
          'W','X','Y','Z']

# 暫存字元切割圖的資料夾
dirname = 'recogdata'
# 載入訓練好的 CNN 模型
model = load_model("carplate_model.hdf5")
# 取得所有要辨識的圖片
myfiles = glob.glob('predictPlate/*.jpg')

# 逐一處理每張圖片
for imgname in myfiles:
    emptydir(dirname)  # 清空暫存資料夾
    img = cv2.imread(imgname)
    
    # 使用 Haar 分類器偵測車牌位置
    detector = cv2.CascadeClassifier('haar_carplate.xml')
    signs = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))

    if len(signs) > 0:
        # 對每個偵測到的車牌區域進行處理
        for(x, y, w, h) in signs:
            image1 = Image.open(imgname)  # 以 PIL 開啟原圖
            image2 = image1.crop((x, y, x+w, y+h))  # 裁切出車牌區域
            image3 = image2.resize((140, 40), Image.LANCZOS)  # 縮放為統一尺寸
            image3.save('tem.jpg')  # 暫存為 tem.jpg

            # 轉為灰階並進行二值化處理
            image4 = cv2.imread('tem.jpg')
            gray = cv2.cvtColor(image4, cv2.COLOR_RGB2GRAY)
            _, img_thre = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            cv2.imwrite('tem.jpg', img_thre)

        # 讀取處理後的 tem.jpg 並進行字元區域切割
        img_tem = cv2.imread('tem.jpg')
        gray = cv2.cvtColor(img_tem, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # 找輪廓
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        letter_image_regions = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            letter_image_regions.append((x, y, w, h))

        # 按照 x 座標排序，從左到右
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        # 將符合大小條件的區塊切出儲存
        i = 1
        for letter_bounding_box in letter_image_regions:
            x, y, w, h = letter_bounding_box
            if w >= 4 and 20 < h < 38:
                letter_image = gray[y:y+h, x:x+w]
                letter_image = cv2.resize(letter_image, (18, 38))  # 統一為模型輸入大小
                cv2.imwrite(dirname + '/{}.jpg'.format(i), letter_image)
                i += 1

        # 顯示每個偵測到的字元區塊（用於 debug）
        #for (x, y, w, h) in letter_image_regions:
            #cv2.rectangle(img_tem, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #cv2.imshow("debug", img_tem)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        # 將所有切出的字元圖片轉為模型輸入格式
        datan = len(os.listdir(dirname))
        tem_data = []
        for index in range(1, datan+1):
            img_arr = np.array(Image.open(f"{dirname}/{index}.jpg")) / 255.0
            tem_data.append(img_arr)

        real_data = np.stack(tem_data)
        real_data1 = np.expand_dims(real_data, axis=3)  # 增加通道維度

        # 使用模型進行預測
        predictions = model.predict(real_data1)
        predicted_classes = np.argmax(predictions, axis=1)

        # 轉換為對應字元
        result = []
        for i in predicted_classes:
            result.append(labels[i])

        print(imgname + ' --> ' + ''.join(result))
    else:
        print('無法擷取車牌! ')

    # 刪除暫存圖片
    os.remove('tem.jpg')


