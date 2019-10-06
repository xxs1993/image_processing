import os;
import shutil

path = "E:/BaiduNetdiskDownload/3/a"
target_path = "E:/BaiduNetdiskDownload/3"
os.makedirs(os.path.join(target_path,"0"))
os.makedirs(os.path.join(target_path,"1"))
os.makedirs(os.path.join(target_path,"2"))
os.makedirs(os.path.join(target_path,"3"))
os.makedirs(os.path.join(target_path,"4"))
os.makedirs(os.path.join(target_path,"5"))
os.makedirs(os.path.join(target_path,"6"))
os.makedirs(os.path.join(target_path,"7"))
dirs = os.listdir(path)
dict = {'正常':0,'大头':1,'小头':3,'圆头':2,'其他':4,'背景':5,'多个':6,'不确定':7,'其他畸形':4}
for dir in dirs:
    dir_path = os.path.join(path,dir)
    for type in os.listdir(dir_path):
        target_dir_id = dict[type]
        type_dir_path = os.path.join(dir_path,type)
        for file in os.listdir(type_dir_path):
            target_file_path = os.path.join(target_path,str(target_dir_id), file)
            file_path = os.path.join(type_dir_path,file)
            shutil.copyfile(file_path,target_file_path)

