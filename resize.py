import os
import shutil
import glob
import time
from PIL import Image 
import PIL

# 定義清空資料夾的函式
# 如果指定的資料夾存在，就刪除後重新建立
def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)  # 刪除整個資料夾
        time.sleep(2)           # 等待系統釋放檔案資源
    os.mkdir(dirname)           # 建立新的空資料夾

# 定義圖檔縮放與儲存函式
# src 是來源資料夾、dst 是儲存資料夾
def dirResize(src, dst):
    # 取得來源資料夾內所有 .JPG 圖片檔案清單
    myfiles = glob.glob(src + '/*.JPG')
    
    # 清空（或建立）目標資料夾
    emptydir(dst)
    
    print(src + ' 資料夾 :')
    print('開始轉換圖形尺寸!')

    # 逐張處理圖片
    for f in myfiles:
        fname = f.split("\\")[-1]  # 取得檔名（不含路徑）
        img = Image.open(f)        # 開啟圖片
        
        # 縮放圖片為 300x225，使用高品質縮放
        img_new = img.resize((18, 38), PIL.Image.LANCZOS)  
        # 儲存到目標資料夾
        img_new.save(dst + '/' + fname)  

    print('轉換圖形尺寸完成!\n')

# 執行兩個資料夾的圖片縮放
#dirResize('realPlate_sr', 'realPlate')         # 處理原始實拍車牌圖片
#dirResize('predictPlate_sr', 'predictPlate')   # 處理預測用車牌圖片
dirResize('o', 'oo')