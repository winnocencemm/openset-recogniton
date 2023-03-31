import os
import fnmatch
import random
import shutil

'''
分离已知设备与未知设备的数据copy到文件夹
'''
feature = 'START'
dir = 'F:/殷鹏程/LTE数据/时间1/Uploadfile/Indoor/30dB/autocollectaftercoursesync_DCTF_'+feature
for f_name in os.listdir(dir):
    if fnmatch.fnmatch(f_name, 'lte[4-8]*.png'):
        path_img = os.path.join(dir,f_name)
        # shutil.copy(path_img, 'F:/王敏/认证/Train/LTE4-8/autocollectaftercoursesync_DCTF_'+feature)
    else:
        path_img = os.path.join(dir, f_name)
        shutil.copy(path_img, 'F:/王敏/认证/Test/LET3/autocollectaftercoursesync_DCTF_'+feature+'_lte3')


input_path='F:/王敏/认证/Train/LTE4-8/autocollectaftercoursesync_DCTF_'+feature
save_train='F:/王敏/认证/Train/LTE4-8/autocollectaftercoursesync_DCTF_'+feature+'80%'
save_val = 'F:/王敏/认证/Train/LTE4-8/autocollectaftercoursesync_DCTF_'+feature+'20%'
if not os.path.exists(save_train):
    os.makedirs(save_train)
if not os.path.exists(save_val):
    os.makedirs(save_val)

pathDir = os.listdir(input_path)  # 取图片的原始路径
random.seed(1)
filenumber = len(pathDir)  # 原文件个数
print("源文件个数："+str(filenumber))
rate = 0.2  # 抽取的验证集的比例，占总数据的多少
picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
print("测试文件个数："+str(picknumber))
numlist = random.sample(range(0, len(pathDir)), picknumber)  # 生成随机数列表a
# sample = random.sample(pathDir, picknumber)  # 随机选取需要数量的样本图片
# print(len(sample))
# val_list=[]
# train_list = []
#
# for i in range(len(sample)):
#     val_list.append(sample[i])
#
# train_list = list(set(pathDir).difference(set(val_list)))
# print(len(train_list))

for n in range(len(pathDir)):
    if n in numlist:
            filename = pathDir[n]
            oldpath = os.path.join(input_path, filename)
            newpath = os.path.join(save_val, filename)
            shutil.copy(oldpath, newpath)
            newpath = os.path.join('F:/王敏/认证/Test/LET3/autocollectaftercoursesync_DCTF_' + feature, filename)
            shutil.copy(oldpath, newpath)
    else:
            filename = pathDir[n]
            oldpath = os.path.join(input_path, filename)
            newpath = os.path.join(save_train, filename)
            shutil.copy(oldpath, newpath)


# for f_name in os.listdir(input_path):
#     if f_name in sample:
#         path_img = os.path.join(input_path, f_name)
#         shutil.copy(path_img, save_val)
#         shutil.copy(path_img, 'F:/王敏/认证/Test/LET3/autocollectaftercoursesync_DCTF_' + feature)
#     else:
#         path_img = os.path.join(input_path, f_name)
#         shutil.copy(path_img, save_train)


# for flie_name in val_list:
#     path_img=os.path.join(input_path,flie_name)
#     shutil.copy(path_img,save_val)
#     shutil.copy(path_img, 'F:/王敏/认证/Test/LET3/autocollectaftercoursesync_DCTF_' + feature)
#
# for flie_name in train_list:
#     path_img=os.path.join(input_path,flie_name)
#     shutil.copy(path_img,save_train)



