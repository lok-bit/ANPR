import os
import shutil
import time
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
        time.sleep(2)
    os.mkdir(dirname)

labels = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

imgname = '3M6605.jpg'
dirname = 'recogdata'
emptydir(dirname)
img = cv2.imread('predictPlate/' + imgname)
detector = cv2.CascadeClassifier('haar_carplate.xml')
signs = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
if len(signs) > 0 :
    for(x, y, w, h) in signs:
        image1 = Image.open('predictPlate/' + imgname)
        image2 = image1.crop((x, y, x+w, y+h))
        image3 = image2.resize((140, 40), Image.LANCZOS)
        image3.save('tem.jpg')
        image4 = cv2.imread('tem.jpg')
        gray = cv2.cvtColor(image4, cv2.COLOR_RGB2GRAY)
        _, img_thre = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        cv2.imwrite('tem.jpg', img_thre)

    img_tem = cv2.imread('tem.jpg')
    gray = cv2.cvtColor(img_tem, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letter_image_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        letter_image_regions.append((x, y, w, h))
    
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    i=1
    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box
        if w>=5 and h>18 and h<35:
            letter_image = gray[y:y+h, x:x+w]
            letter_image = cv2.resize(letter_image, (18, 38))
            cv2.imwrite(dirname + '/{}.jpg'.format(i), letter_image)
            i += 1

    datan = 0
    for fname in os.listdir(dirname):
        if os.path.isfile(os.path.join(dirname, fname)):
            datan += 1
    
    tem_data = []
    for index in range(1, (datan+1)):
        tem_data.append((np.array(Image.open("recogdata/" + str(index) +".jpg")))/255.0)

    real_data = np.stack(tem_data)
    real_data1 = np.expand_dims(real_data, axis=3)
    model = load_model("carplate_model.hdf5")
    predictions = model.predict(real_data1)
    predicted_classes = np.argmax(predictions, axis=1)

    print('車牌號碼為: ')
    for i in predicted_classes:
        print(labels[i], end='')
else:
    print('無法擷取車牌! ')
os.remove('tem.jpg') 