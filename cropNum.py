import cv2
import os
import shutil
import time

def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
        time.sleep(2)
    os.mkdir(dirname)

emptydir('cropMono')

image = cv2.imread('cropPlate/AWD6618.jpg')
if image is None:
    print("❌ 圖片讀取失敗！請確認路徑正確")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)


contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

letter_image_regions = []
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    letter_image_regions.append((x, y, w, h))

letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
print(f"共偵測到 {len(letter_image_regions)} 個區塊")
print(letter_image_regions)

i = 1
for letter_bounding_box in letter_image_regions:
    x, y, w, h = letter_bounding_box
    print(f'處理區塊:w={w}, h={h}')  # 查看篩選邏輯是否合適
    if w >= 3 and h > 20 and h < 35:
        letter_image = gray[y:y+h, x:x+w]
        if w < 4:
            padding = 5
            letter_image = cv2.copyMakeBorder(
            letter_image,
            top=0, bottom=0, left=padding, right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=255  
        )
        letter_image = cv2.resize(letter_image, (18, 38))
        cv2.imwrite('cropMono/{}.jpg'.format(i), letter_image)
        i += 1

if i == 1:
    print("⚠️ 沒有任何區塊符合條件被裁切")
else:
    print(f"✅ 已裁切 {i-1} 張文字圖片到 cropMono/")

