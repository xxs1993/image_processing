## used to combine 1 2 3 4 to one packages
import os
import shutil
import random
source_path = "E:/BaiduNetdiskDownload/5"
normal_size = len(os.listdir(os.path.join(source_path,"0")))
new_path = os.path.join(source_path,"new")
if os.path.exists(new_path):
    shutil.rmtree(new_path)
os.makedirs(new_path)
abnormal_files={}
abnormal_files["1"] = os.listdir(os.path.join(source_path,"1"))
abnormal_files["2"] = os.listdir(os.path.join(source_path,"2"))
abnormal_files["3"] = os.listdir(os.path.join(source_path,"3"))
abnormal_files["4"] = os.listdir(os.path.join(source_path,"4"))
size1= len(abnormal_files["1"])
size2=len(abnormal_files["2"])
size3=len(abnormal_files["3"])
size4=len(abnormal_files["4"])
abnormal_propertie=[min(1,normal_size/(4*size1)),min(1,normal_size/(4*size2)),min(1,normal_size/(4*size3)),min(1,normal_size/(4*size4))]
for i in abnormal_files:
    n = int(i)
    property = abnormal_propertie[n-1]
    for file in abnormal_files[i]:
        r = random.random()
        file_path = os.path.join(source_path,i,file)
        if r< property:
            shutil.copyfile(file_path,os.path.join(new_path,file))

