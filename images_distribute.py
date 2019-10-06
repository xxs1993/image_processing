##divide images into two parts
import os
import shutil
import random

source_path = "E:/BaiduNetdiskDownload/5"

pre_path = os.path.join(source_path,"pre")

suff_path = os.path.join(source_path,"suff")

dirs = os.listdir(source_path)

os.makedirs(name=pre_path, exist_ok=True)
os.makedirs(name=suff_path, exist_ok=True)
i = 0
for dir in dirs:
    ori_dir = os.path.join(source_path, dir)
    pre_dir = os.path.join(pre_path, dir)
    suff_dir = os.path.join(suff_path,dir)
    os.makedirs(name=pre_dir, exist_ok=True)
    os.makedirs(name=suff_dir, exist_ok=True)
    list = os.listdir(ori_dir)
    for file in list:
        r = random.randint(0, 1)
        new_path = suff_dir
        if r == 0:
            new_path = pre_dir
        file_dir = os.path.join(ori_dir, file)
        shutil.copyfile(file_dir,os.path.join(new_path, file))
        i = i+1






