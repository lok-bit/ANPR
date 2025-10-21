import os
import shutil
import time
import glob

# 定義清空資料夾的函式：若資料夾已存在就刪除，然後重新建立
def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
        time.sleep(2)
    os.mkdir(dirname)

print('開始建立文字庫! ')

# 建立主資料夾 platefont，並清空舊資料
emptydir('platefont')

# 定義所有車牌可能用到的字元（不含 I、O）
fontlist = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','J','K',
            'L','M','N','P','Q','R','S','T','U','V',
            'W','X','Y','Z']

# 為每個字元建立一個子資料夾，例如 platefont/A、platefont/3
for i in range(len(fontlist)):
    emptydir('platefont/' + fontlist[i])

# 取得 cropNum 資料夾中所有車牌資料夾名稱（例如 ABC1234）
dirs = os.listdir('cropNum')

picnum = 1  # 圖片的流水編號

# 逐一處理每個車牌資料夾
for d in dirs:
    if os.path.isdir('cropNum/' + d):
        # 取得該車牌資料夾中的所有字元圖片
        myfiles = glob.glob('cropNum/' + d + '/*.jpg')

        # 將每張字元圖片根據車牌文字的順序分類
        for i, f in enumerate(myfiles):
            # 取得對應字元（例如 d="ABC1234"，第0張對應 A，第1張對應 B）

            # 複製這張圖片到對應字元的資料夾中，檔名為流水號
            shutil.copyfile(f, 'platefont/{}/{}.jpg'.format(d[i], picnum))
            picnum += 1

print('建立文字庫結束! ')
