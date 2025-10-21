import cv2
import os
import shutil
import time
import glob

# 定義清空資料夾的函式（如果存在就刪除並重新建立）
def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
        time.sleep(2)
    os.mkdir(dirname)

print('開始擷取車牌數字! ')

# 清空或建立儲存字元切割結果的資料夾 cropNum/
emptydir('cropNum')

# 取得 cropPlate 資料夾中所有 .jpg 車牌圖片
myfiles = glob.glob('cropPlate\\*.jpg')

# 逐張車牌圖片處理
for f in myfiles:
    # 取得圖片檔名（不含副檔名 .jpg）
    filename = (f.split('\\'))[-1].replace('.jpg', '')

    # 為每張車牌圖片建立一個子資料夾儲存它的字元
    emptydir('cropNum/' + filename)

    # 讀取圖片
    image = cv2.imread(f)
    if image is None:
        print("圖片讀取失敗！請確認路徑正確")
        exit()

    # 轉為灰階圖片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 將灰階圖片二值化（黑白反轉）
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)


    # 偵測輪廓（使用外框模式 + 簡單輪廓壓縮）
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_image_regions = []  # 儲存所有偵測到的字元區域 (x, y, w, h)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)  # 用邊界框包住輪廓
        letter_image_regions.append((x, y, w, h)) # 加入清單

    # 根據 x 座標排序，確保字元由左到右儲存
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    print(f"共偵測到 {len(letter_image_regions)} 個區塊")
    print(letter_image_regions)

    i = 1
    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box
        print(f'處理區塊: w={w}, h={h}')  # 顯示區塊資訊供除錯檢查

        # 根據寬度與高度篩選合理字元區塊（去除雜訊）
        if w >= 3 and h > 20 and h < 35:
            # 擷取出單一字元影像
            letter_image = gray[y:y+h, x:x+w]
            if w < 5:
                padding = 5
                letter_image = cv2.copyMakeBorder(
                letter_image,
                top=0, bottom=0, left=padding, right=padding,
                borderType=cv2.BORDER_CONSTANT,
                value=255  
            )
            # 統一字元圖片大小（18×38）方便後續訓練
            letter_image = cv2.resize(letter_image, (18, 38))

            # 儲存字元圖片到對應子資料夾中
            cv2.imwrite('cropNum/' + filename + '/{}.jpg'.format(i + 1), letter_image)
            i += 1

print('擷取車牌數字結束! ')
