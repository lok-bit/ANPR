import os
import shutil
import time
import glob
import random
import cv2

# 清空指定資料夾（若已存在則刪除後重建）
def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
        time.sleep(2)
    os.mkdir(dirname)

# 定義所有可能的車牌字元（不含 I、O）
fontlist = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','J','K',
            'L','M','N','P','Q','R','S','T','U','V',
            'W','X','Y','Z']

print('開始建立訓練資料! ')

# 清空或建立 data 資料夾，用於存放訓練用圖片
emptydir('data')

# 逐一處理每個字元類別
for n in range(len(fontlist)):
    print('產生 data/' + fontlist[n] + ' 資料夾')
    
    # 為每個字元建立對應的子資料夾
    emptydir('data/' + fontlist[n])

    # 取得 platefont/ 該字元類別的所有圖片
    myfiles = glob.glob('platefont/' + fontlist[n] + '/*.jpg')
    pic_total = 500  # 每個類別目標總數為 500 張
    pic_each = int(pic_total / len(myfiles)) + 1  # 平均每張需複製次數

    # 對每張原始字元圖進行複製（擴增）
    for index, f in enumerate(myfiles):        
        for i in range(pic_each):
            img = cv2.imread(f)  # 讀取原始圖            
            # 加入隨機雜點：每張圖加入 20 個隨機黑點
            for j in range(20):
                x = random.randint(0, 17)
                y = random.randint(0, 37)
                img[y, x] = 0
            # 儲存圖像到 data/對應資料夾，檔名為四位數流水號
            cv2.imwrite('data/' + fontlist[n] + '/{:0>4d}.jpg'.format(index * pic_each + i + 1), img)

print('建立訓練資料結束! ')



