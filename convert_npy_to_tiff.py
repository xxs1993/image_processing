##  convert npy file to tiff images
import cv2
import os
import numpy as np
from PIL import Image

arr = "D:/git/data/mhsma-dataset-master"
arr1 = os.path.join(arr,"mhsma")
pre = "x_128_"
arr2 = "sample"
types = ["train.npy","valid.npy","test.npy"]
# path1 = os.path.join(arr,arr1)
# sample_path = os.path.join(arr,arr2)
for type in types:
    label_acrosome_path = os.path.join(arr1,"y_acrosome_"+type)
    label_head_path = os.path.join(arr1,"y_head_"+type)
    label_vacuole_path = os.path.join(arr1, "y_vacuole_" + type)
    src_path = os.path.join(arr1,pre+type)
    acrosome_label = np.asarray(np.load(label_acrosome_path))
    head_label = np.asarray(np.load(label_head_path))
    vacuole_label = np.asarray(np.load(label_vacuole_path))
    arrs = np.asarray(np.load(src_path))
    sample_file_path = os.path.join(arr,arr2,type)
    count = 0

    for i in arrs:
        package = 0
        if acrosome_label[count]== 1 or head_label[count] == 1 or vacuole_label[count]==1:
            package = 1
        r = np.asarray(Image.fromarray(i))
    # c = cv2.cvtColor(r,cv2.COLOR_RGB2GRAY)
        name = os.path.join(sample_file_path,str(package),str(count)+".tiff")
        r = cv2.resize(r,(71,71))
        cv2.imwrite(name,r)
        count = count+1
# dir="/home/jobs/Desktop/data_self/"
# dest_dir="/home/jobs/Desktop/jpg_self/"
# def npy2jpg(dir,dest_dir):
#     if os.path.exists(dir)==False:
#         os.makedirs(dir)
#     if os.path.exists(dest_dir)==False:
#         os.makedirs(dest_dir)
#     file=dir+'test_data.npy'
#     con_arr=np.load(file)
#     count=0
#     for con in con_arr:
#         arr=con[0]
#         label=con[1]
#         print(np.argmax(label))
#         arr=arr*255
#         #arr=np.transpose(arr,(2,1,0))
#         arr=np.reshape(arr,(3,112,112))
#         r=Image.fromarray(arr[0]).convert("L")
#         g=Image.fromarray(arr[1]).convert("L")
#         b=Image.fromarray(arr[2]).convert("L")
#
#         img=Image.merge("RGB",(r,g,b))
#
#         label_index=np.argmax(label)
#         img.save(dest_dir+str(label_index)+"_"+str(count)+".jpg")
#         count=count+1