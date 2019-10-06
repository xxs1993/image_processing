import os
## change path to your local path
path = "E:/sperm_image/HuSHem/01"
files = os.listdir(path)
lenth = len(files)
suff = '.tiff'
counting = 1
for i in files:
    file_path = os.path.join(path,i)
    new_path = os.path.join(path,str(counting)+suff)
    os.rename(file_path,new_path)
    counting = counting+1