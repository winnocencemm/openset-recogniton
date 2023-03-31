from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk
import matplotlib.image as mimg
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
IMAGESIZE=129

import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus'] = False



'''
导入训练集
'''

#TrainPath='D:/文档/项目相关资料代码/4GLTE/LTE-RACH-FROM-WZN/diff-loc/Train and Test/DCTF/train'
TrainPath='D:/文档/项目相关资料代码/4GLTE/LTE-RACH-FROM-WZN/final/Indoor/train'

'''
读取训练集图片，得到矩阵TrainMat 和其转置 TrainMatT
'''

##使用一个DCTF图像

switch='/ALL'
swich=["/START","/STATE","/END"]

TrainFilesName = os.listdir(TrainPath+switch)
TrainMat=[]
trainLabel=[]
for filename in TrainFilesName:
    e=swich[0]
    image = mimg.imread(TrainPath + e + '/' + filename)
    imageresize1 = image.reshape(image.size, 1)

    e = swich[1]
    image = mimg.imread(TrainPath + e + '/' + filename)
    imageresize2 = image.reshape(image.size, 1)

    e = swich[2]
    image = mimg.imread(TrainPath + e + '/' + filename)
    imageresize3 = image.reshape(image.size, 1)

    oneimage=np.vstack((imageresize1,imageresize2,imageresize3))

    TrainMat.append(oneimage)

    if filename[4:5].isdigit():
        trainLabel.append(10)
    else:
        trainLabel.append(int(filename[3:4]))

TrainMat=np.array(TrainMat)
TrainMat = TrainMat.reshape(TrainMat.shape[0], TrainMat.shape[1])

n_components=100
pca=PCA(n_components=n_components)
MYPCA=pca.fit(TrainMat)
TrainFeatureVects=MYPCA.transform(TrainMat)
trainLabel=np.array(trainLabel)

'''
画出特征脸，也就是特征向量组合重构成的图片
'''
# igenvector=MYPCA.components_
# for i in range(n_components):
#     V1=igenvector[i]
#     V_scale=[]
#     V_min=min(V1)
#     V_max=max(V1)
#     for vi in V1:
#         vi=(vi-V_min)/(V_max-V_min)
#         V_scale.append(vi)
#     V_scale_Mat=np.array(V_scale).reshape(IMAGESIZE,IMAGESIZE,4)
#     plt.figure()
#     plt.imshow(V_scale_Mat)

'''
三维
'''
fig=plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
ax.scatter(TrainFeatureVects[trainLabel==3,0],TrainFeatureVects[trainLabel==3,1],TrainFeatureVects[trainLabel==3,2],edgecolors='y',color='y',marker='o',alpha=1)
ax.scatter(TrainFeatureVects[trainLabel==4,0],TrainFeatureVects[trainLabel==4,1],TrainFeatureVects[trainLabel==4,2],edgecolors='b',color='b',marker='s',alpha=1)
ax.scatter(TrainFeatureVects[trainLabel==5,0],TrainFeatureVects[trainLabel==5,1],TrainFeatureVects[trainLabel==5,2],edgecolors='r',color='r',marker='^',alpha=1)
ax.scatter(TrainFeatureVects[trainLabel==6,0],TrainFeatureVects[trainLabel==6,1],TrainFeatureVects[trainLabel==6,2],edgecolors='g',color='g',marker='v',alpha=1)
ax.scatter(TrainFeatureVects[trainLabel==7,0],TrainFeatureVects[trainLabel==7,1],TrainFeatureVects[trainLabel==7,2],edgecolors='m',color='m',marker='D',alpha=1)
ax.scatter(TrainFeatureVects[trainLabel==8,0],TrainFeatureVects[trainLabel==8,1],TrainFeatureVects[trainLabel==8,2],edgecolors='c',color='c',marker='p',alpha=1)
ax.set_xlabel('X轴',fontsize=14)
ax.set_ylabel('Y轴',fontsize=14)
ax.set_zlabel('Z轴',fontsize=14)
ax.set_xticklabels(np.array(range(-15,30,5)),fontsize=14)
ax.set_yticklabels(np.array(range(-15,20,5)),fontsize=14)
ax.set_zticklabels(np.array(range(-15,20,5)),fontsize=14)
ax.legend(labels=['手机1','手机2','手机3','手机4','手机5','手机6'],bbox_to_anchor=(0.67, 0.55),fontsize=15)

'''
二维空间
'''
plt.figure()
plt.scatter(TrainFeatureVects[trainLabel==3,0],TrainFeatureVects[trainLabel==3,1],edgecolors='y',color='',marker='o',alpha=1)
plt.scatter(TrainFeatureVects[trainLabel==4,0],TrainFeatureVects[trainLabel==4,1],edgecolors='b',color='',marker='s',alpha=1)
plt.scatter(TrainFeatureVects[trainLabel==5,0],TrainFeatureVects[trainLabel==5,1],edgecolors='r',color='',marker='^',alpha=1)
plt.scatter(TrainFeatureVects[trainLabel==6,0],TrainFeatureVects[trainLabel==6,1],edgecolors='g',color='',marker='v',alpha=1)
plt.scatter(TrainFeatureVects[trainLabel==7,0],TrainFeatureVects[trainLabel==7,1],edgecolors='m',color='',marker='D',alpha=1)
plt.scatter(TrainFeatureVects[trainLabel==8,0],TrainFeatureVects[trainLabel==8,1],edgecolors='c',color='',marker='p',alpha=1)
plt.legend(labels=['手机1','手机2','手机3','手机4','手机5','手机6'],loc='best',fontsize=15)
plt.xlabel('X轴',fontsize=15)
plt.ylabel('Y轴',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.tight_layout()
plt.show()


'''
一维
'''
plt.figure()
list0=[0]*len(TrainFeatureVects[trainLabel==3])
plt.plot(list0,TrainFeatureVects[trainLabel==3,0],'o',c='y')

list1=[1]*len(TrainFeatureVects[trainLabel==4])
plt.plot(list1,TrainFeatureVects[trainLabel==4,0],'o',c='b')

list2=[2]*len(TrainFeatureVects[trainLabel==5])
plt.plot(list2,TrainFeatureVects[trainLabel==5,0],'o',c='r')

list3=[3]*len(TrainFeatureVects[trainLabel==6])
plt.plot(list3,TrainFeatureVects[trainLabel==6,0],'o',c='g')

list4=[4]*len(TrainFeatureVects[trainLabel==7])
plt.plot(list4,TrainFeatureVects[trainLabel==7,0],'o',c='m')

list5=[5]*len(TrainFeatureVects[trainLabel==8])
plt.plot(list5,TrainFeatureVects[trainLabel==8,0],'o',c='c')

x=list(range(0,6,1))
x_index=['手机1','手机2','手机3','手机4','手机5','手机6']
plt.xticks(x,x_index,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('设备标签',fontsize=14)
plt.ylabel('一维特征数值',fontsize=14)
# tempsum=0
# feature_percent=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99]
# j=0
# feature_weishu=[]
# for i in range(len(pca.explained_variance_ratio_)):
#     tempsum=tempsum+pca.explained_variance_ratio_[i]
#     if j<=11 and tempsum>=feature_percent[j] :
#         j+=1
#         feature_weishu.append(i)
#
# '''
# 识别过程
# '''
#
# '''
# 首先生成测试集的特征向量
# '''
# #TestPath='D:/文档/项目相关资料代码/4GLTE/LTE-RACH-FROM-WZN/diff-loc/Train and Test/DCTF/test'
# TestPath='D:/文档/项目相关资料代码/4GLTE/LTE-RACH-FROM-WZN/final/Indoor/test'
#
# TestFileNames = os.listdir(TestPath+switch)
#
# trueLabel=[]
#
# TestFeatureVects=[]
# TestMat=[]
# for filename in TestFileNames:
#     if filename[4:5].isdigit():
#         trueLabel.append(10)
#     else:
#         trueLabel.append(int(filename[3:4]))
#
#     e = swich[0]
#     image = mimg.imread(TestPath + e + '/' + filename)
#     imageresize1 = image.reshape(image.size, 1)
#
#     e = swich[1]
#     image = mimg.imread(TestPath + e + '/' + filename)
#     imageresize2 = image.reshape(image.size, 1)
#
#     e = swich[2]
#     image = mimg.imread(TestPath + e + '/' + filename)
#     imageresize3 = image.reshape(image.size, 1)
#
#     oneimage = np.vstack((imageresize1, imageresize2, imageresize3))
#     #
#
#     TestMat.append(oneimage)
#
# TestMat=np.array(TestMat)
# TestMat=TestMat.reshape(TestMat.shape[0],TestMat.shape[1])
# TestFeatureVects=MYPCA.transform(TestMat)
# #
# #
# '''
# 根据余弦相似度判断
# '''
# def cos_sim(a,b):
#     a_norm=np.linalg.norm(a)
#     b_norm=np.linalg.norm(b)
#     cos = np.dot(a, b) / (a_norm * b_norm)
#     return cos
#
#
# preLabel=[]
# for testvect in TestFeatureVects:
#
#     Similarity=[]
#     for trainvect in TrainFeatureVects:
#         Similarity.append(cos_sim(testvect,trainvect))
#
#     max_Sim=max(Similarity)
#     index=Similarity.index(max_Sim)
#     preLabel.append(trainLabel[index])
#
#
# trueLabel=np.array(trueLabel)
# preLabel=np.array(preLabel)
#
# cm1 = metrics.confusion_matrix(trueLabel, preLabel)
# ACC1 = metrics.accuracy_score(trueLabel, preLabel)
# Precision1 = metrics.precision_score(trueLabel, preLabel,average='weighted')
# Recall1 = metrics.recall_score(trueLabel, preLabel,average='weighted')
# F1_score1 = metrics.f1_score(trueLabel, preLabel,average='weighted')
#
#
#
# '''
# 根据范数(2范数，1范数,无穷范数)
# '''
#
# preLabel=[]
# for testvect in TestFeatureVects:
#
#     Distance=[]
#     for trainvect in TrainFeatureVects:
#         Distance.append(np.linalg.norm(testvect-trainvect))
#
#     min_Dis=min(Distance)
#     index=Distance.index(min_Dis)
#     preLabel.append(trainLabel[index])
#
#
# trueLabel=np.array(trueLabel)
# preLabel=np.array(preLabel)
#
# cm2 = metrics.confusion_matrix(trueLabel, preLabel)
# ACC2 = metrics.accuracy_score(trueLabel, preLabel)
# Precision2 = metrics.precision_score(trueLabel, preLabel,average='weighted')
# Recall2 = metrics.recall_score(trueLabel, preLabel,average='weighted')
# F1_score2 = metrics.f1_score(trueLabel, preLabel,average='weighted')
#
# preLabel=[]
# for testvect in TestFeatureVects:
#
#     Distance=[]
#     for trainvect in TrainFeatureVects:
#         Distance.append(np.linalg.norm(testvect-trainvect,ord=1))
#
#     min_Dis=min(Distance)
#     index=Distance.index(min_Dis)
#     preLabel.append(trainLabel[index])
#
#
# trueLabel=np.array(trueLabel)
# preLabel=np.array(preLabel)
#
# cm3 = metrics.confusion_matrix(trueLabel, preLabel)
# ACC3 = metrics.accuracy_score(trueLabel, preLabel)
# Precision3 = metrics.precision_score(trueLabel, preLabel,average='weighted')
# Recall3 = metrics.recall_score(trueLabel, preLabel,average='weighted')
# F1_score3 = metrics.f1_score(trueLabel, preLabel,average='weighted')
#
# preLabel=[]
# for testvect in TestFeatureVects:
#
#     Distance=[]
#     for trainvect in TrainFeatureVects:
#         Distance.append(np.linalg.norm(testvect-trainvect,ord=np.inf))
#
#     min_Dis=min(Distance)
#     index=Distance.index(min_Dis)
#     preLabel.append(trainLabel[index])
#
#
# trueLabel=np.array(trueLabel)
# preLabel=np.array(preLabel)
#
# cm4 = metrics.confusion_matrix(trueLabel, preLabel)
# ACC4 = metrics.accuracy_score(trueLabel, preLabel)
# Precision4 = metrics.precision_score(trueLabel, preLabel,average='weighted')
# Recall4 = metrics.recall_score(trueLabel, preLabel,average='weighted')
# F1_score4 = metrics.f1_score(trueLabel, preLabel,average='weighted')
#
#
#
# '''
# KNN分类器,使用欧氏距离
# '''
# from sklearn.neighbors import KNeighborsClassifier
# knn=KNeighborsClassifier(n_neighbors=6)
# knn.fit(TrainFeatureVects,trainLabel)
# preLabel=knn.predict(TestFeatureVects)
#
# preLabel=np.array(preLabel)
# trueLabel=np.array(trueLabel)
#
# cm5 = metrics.confusion_matrix(trueLabel, preLabel)
# ACC5 = metrics.accuracy_score(trueLabel, preLabel)
# Precision5 = metrics.precision_score(trueLabel, preLabel,average='weighted')
# Recall5 = metrics.recall_score(trueLabel, preLabel,average='weighted')
# F1_score5 = metrics.f1_score(trueLabel, preLabel,average='weighted')
#
# ###测试集降维结果，颜色表示正确标签
# fig=plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(TestFeatureVects[trueLabel==3,0],TestFeatureVects[trueLabel==3,1],TestFeatureVects[trueLabel==3,2],color='red',marker='o',alpha=1)
# ax.scatter(TestFeatureVects[trueLabel==4,0],TestFeatureVects[trueLabel==4,1],TestFeatureVects[trueLabel==4,2],color='blue',marker='o',alpha=1)
# ax.scatter(TestFeatureVects[trueLabel==5,0],TestFeatureVects[trueLabel==5,1],TestFeatureVects[trueLabel==5,2],color='yellow',marker='o',alpha=1)
# ax.scatter(TestFeatureVects[trueLabel==6,0],TestFeatureVects[trueLabel==6,1],TestFeatureVects[trueLabel==6,2],color='green',marker='o',alpha=1)
# ax.scatter(TestFeatureVects[trueLabel==7,0],TestFeatureVects[trueLabel==7,1],TestFeatureVects[trueLabel==7,2],color='pink',marker='o',alpha=1)
# ax.scatter(TestFeatureVects[trueLabel==8,0],TestFeatureVects[trueLabel==8,1],TestFeatureVects[trueLabel==8,2],color='gray',marker='o',alpha=1)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# ###测试集中分类错误的点和分类正确的点
# fig=plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(TestFeatureVects[trueLabel!=preLabel,0],TestFeatureVects[trueLabel!=preLabel,1],TestFeatureVects[trueLabel!=preLabel,2],color='red',marker='o',alpha=1)
# ax.scatter(TestFeatureVects[trueLabel==preLabel,0],TestFeatureVects[trueLabel==preLabel,1],TestFeatureVects[trueLabel==preLabel,2],color='blue',marker='o',alpha=1)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')