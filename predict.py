from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

labels = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
datan = 0
for fname in os.listdir('cropMono'):
    if os.path.isfile(os.path.join('cropMono', fname)):
        datan += 1
tem_data = []
for index in range(1, (datan+1)):
    tem_data.append((np.array(Image.open("cropMono/" + str(index) +".jpg")))/255.0)

real_data = np.stack(tem_data)
real_data1 = np.expand_dims(real_data, axis=3)
model = load_model("carplate_model.hdf5")
predictions = model.predict(real_data1)
predicted_classes = np.argmax(predictions, axis=1)

print('車牌號碼為:')
for i in range(len(predicted_classes)):
    print(labels[int(predicted_classes[i])], end='')