import cv2
import glob
import os
import shutil
import time
from PIL import Image 

# 定義清空資料夾的函式：若資料夾存在就刪除，重新建立空資料夾
def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
        time.sleep(2)           # 等待 2 秒，確保刪除乾淨
    os.mkdir(dirname)

print('開始擷取車牌! ')
print('無法擷取車牌的圖片 : ')

# 設定輸出資料夾名稱
dstdir = 'cropPlate'

# 取得 realPlate 資料夾中的所有 .JPG 檔案
myfiles = glob.glob("realPlate\\*.JPG")

# 清空/建立 cropPlate 資料夾
emptydir(dstdir)

# 逐張處理圖片
for imgname in myfiles:
    filename = (imgname.split('\\'))[-1]  # 取出檔名（不含路徑）
    img = cv2.imread(imgname)             # 讀取圖片

    # 載入訓練好的 Haar 車牌分類器
    detector = cv2.CascadeClassifier('haar_carplate.xml')

    # 偵測車牌：設定最小車牌尺寸、縮放倍率與重疊區塊門檻
    signs = detector.detectMultiScale(img, minSize=(20, 20), scaleFactor=1.1, minNeighbors=5)

    # 若有偵測到車牌
    if len(signs) > 0:
        for (x, y, w, h) in signs:
            image1 = Image.open(imgname)                         # 用 PIL 開圖
            image2 = image1.crop((x, y, x + w, y + h))           # 裁切出車牌區域
            image3 = image2.resize((140, 40), Image.LANCZOS)     # 縮放為固定尺寸
            image3.save(dstdir + '/tem.jpg')                     # 暫存成 tem.jpg

            image4 = cv2.imread(dstdir + '/tem.jpg')             # 重新讀取剛儲存的車牌圖片
            img_gray = cv2.cvtColor(image4, cv2.COLOR_RGB2GRAY)  # 轉成灰階
            _, img_thre = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)  # 二值化

            # 儲存二值化的車牌圖片，檔名與原圖相同
            cv2.imwrite(dstdir + '/' + filename, img_thre)

    else:
        # 若沒有偵測到車牌，輸出檔名以供人工檢查
        print(filename)

# 移除暫存檔 tem.jpg
os.remove(dstdir + '/tem.jpg')

print('擷取車牌結束! ')

