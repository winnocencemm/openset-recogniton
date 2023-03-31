from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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





'''
导入训练集
'''

#TrainPath='D:/文档/项目相关资料代码/4GLTE/LTE-RACH-FROM-WZN/diff-loc/Train and Test/DCTF/train'
#TrainPath='D:/文档/项目相关资料代码/4GLTE/lte_rach/final/Indoor/30dB/TRAIN'
#TrainPath='D:/文档/项目相关资料代码/4GLTE/LTE-RACH-FROM-WZN/final/Indoor/train'
TrainPath='D:/文档/项目相关资料代码/4GLTE/lte_rach/final/Indoor/30dB'

'''
读取训练集图片，得到矩阵TrainMat 和其转置 TrainMatT
'''

##使用一个DCTF图像

switch='/ALL'
switch='/autocollectaftercoursesync_DCTF_ALL'
TrainFilesName = os.listdir(TrainPath+switch)
TrainMat=[]
trainLabel=[]
for filename in TrainFilesName:
    image=mimg.imread(TrainPath+switch+'/'+filename)
    imageresize = image.reshape(image.size, 1)
    TrainMat.append(imageresize)
    trainLabel.append(int(filename[3:4]))

TrainMat=np.array(TrainMat)
TrainMat = TrainMat.reshape(TrainMat.shape[0], TrainMat.shape[1])
trainLabel=np.array(trainLabel)

n_components=130
#n_components=1500
pca=PCA(n_components=n_components)
MYPCA=pca.fit(TrainMat)
TrainFeatureVects=MYPCA.transform(TrainMat)


n_components=5
lda=LinearDiscriminantAnalysis(n_components=n_components)
MYLDA=lda.fit(TrainFeatureVects,trainLabel)
TrainFeatureVects=MYLDA.transform(TrainFeatureVects)


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

# fig=plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(TrainFeatureVects[trainLabel==3,0],TrainFeatureVects[trainLabel==3,1],TrainFeatureVects[trainLabel==3,2],color='red',marker='o',alpha=1)
# ax.scatter(TrainFeatureVects[trainLabel==4,0],TrainFeatureVects[trainLabel==4,1],TrainFeatureVects[trainLabel==4,2],color='blue',marker='o',alpha=1)
# ax.scatter(TrainFeatureVects[trainLabel==5,0],TrainFeatureVects[trainLabel==5,1],TrainFeatureVects[trainLabel==5,2],color='yellow',marker='o',alpha=1)
# ax.scatter(TrainFeatureVects[trainLabel==6,0],TrainFeatureVects[trainLabel==6,1],TrainFeatureVects[trainLabel==6,2],color='green',marker='o',alpha=1)
# ax.scatter(TrainFeatureVects[trainLabel==7,0],TrainFeatureVects[trainLabel==7,1],TrainFeatureVects[trainLabel==7,2],color='pink',marker='o',alpha=1)
# ax.scatter(TrainFeatureVects[trainLabel==8,0],TrainFeatureVects[trainLabel==8,1],TrainFeatureVects[trainLabel==8,2],color='gray',marker='o',alpha=1)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')



# plt.scatter(TrainFeatureVects[trainLabel==3,0],TrainFeatureVects[trainLabel==3,1],color='red',marker='o',alpha=1)
# plt.scatter(TrainFeatureVects[trainLabel==4,0],TrainFeatureVects[trainLabel==4,1],color='blue',marker='o',alpha=1)
# plt.scatter(TrainFeatureVects[trainLabel==5,0],TrainFeatureVects[trainLabel==5,1],color='yellow',marker='o',alpha=1)
# plt.scatter(TrainFeatureVects[trainLabel==6,0],TrainFeatureVects[trainLabel==6,1],color='green',marker='o',alpha=1)
# plt.scatter(TrainFeatureVects[trainLabel==7,0],TrainFeatureVects[trainLabel==7,1],color='black',marker='o',alpha=1)
# plt.scatter(TrainFeatureVects[trainLabel==8,0],TrainFeatureVects[trainLabel==8,1],color='pink',marker='o',alpha=1)
# plt.ylabel('pc1')
# plt.ylabel('pc2')
# plt.tight_layout()
# plt.show()


# print(lda.explained_variance_ratio_)

'''
识别过程
'''

'''
首先生成测试集的特征向量
'''
#TestPath='D:/文档/项目相关资料代码/4GLTE/LTE-RACH-FROM-WZN/diff-loc/Train and Test/DCTF/test'
#TestPath='D:/文档/项目相关资料代码/4GLTE/lte_rach/final/Indoor/30dB/TEST'
#TestPath='D:/文档/项目相关资料代码/4GLTE/LTE-RACH-FROM-WZN/final/Indoor/test'
#TestPath='D:/文档/项目相关资料代码/4GLTE/lte_rach/final/Mix/30dB/TEST/coarsesync'
TestPath='D:/文档/项目相关资料代码/4GLTE/lte_rach/final/Outdoor/30dB'
TestFileNames = os.listdir(TestPath+switch)

trueLabel=[]

TestFeatureVects=[]
TestMat=[]
for filename in TestFileNames:
    if filename[4:5].isdigit():
        trueLabel.append(10)
    else:
        trueLabel.append(int(filename[3:4]))


    """
    一个DCTF
    """

    image = mimg.imread(TestPath+ switch + '/' + filename)
    imageresize = image.reshape(image.size, 1)
    TestMat.append(imageresize)

TestMat=np.array(TestMat)
TestMat=TestMat.reshape(TestMat.shape[0],TestMat.shape[1])

TestFeatureVects=MYPCA.transform(TestMat)
TestFeatureVects=MYLDA.transform(TestFeatureVects)
#
#
'''
根据余弦相似度判断
'''
def cos_sim(a,b):
    a_norm=np.linalg.norm(a)
    b_norm=np.linalg.norm(b)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos


preLabel=[]
for testvect in TestFeatureVects:

    Similarity=[]
    for trainvect in TrainFeatureVects:
        Similarity.append(cos_sim(testvect,trainvect))

    max_Sim=max(Similarity)
    index=Similarity.index(max_Sim)
    preLabel.append(trainLabel[index])


trueLabel=np.array(trueLabel)
preLabel=np.array(preLabel)

cm1 = metrics.confusion_matrix(trueLabel, preLabel)
ACC1 = metrics.accuracy_score(trueLabel, preLabel)
Precision1 = metrics.precision_score(trueLabel, preLabel,average='weighted')
Recall1 = metrics.recall_score(trueLabel, preLabel,average='weighted')
F1_score1 = metrics.f1_score(trueLabel, preLabel,average='weighted')



'''
根据范数(2范数，1范数)
'''

preLabel=[]
for testvect in TestFeatureVects:

    Distance=[]
    for trainvect in TrainFeatureVects:
        Distance.append(np.linalg.norm(testvect-trainvect))

    min_Dis=min(Distance)
    index=Distance.index(min_Dis)
    preLabel.append(trainLabel[index])


trueLabel=np.array(trueLabel)
preLabel=np.array(preLabel)

cm2 = metrics.confusion_matrix(trueLabel, preLabel)
ACC2 = metrics.accuracy_score(trueLabel, preLabel)
Precision2 = metrics.precision_score(trueLabel, preLabel,average='weighted')
Recall2 = metrics.recall_score(trueLabel, preLabel,average='weighted')
F1_score2 = metrics.f1_score(trueLabel, preLabel,average='weighted')

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


'''
KNN分类器,使用欧氏距离
'''
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=20)
knn.fit(TrainFeatureVects,trainLabel)
preLabel=knn.predict(TestFeatureVects)

preLabel=np.array(preLabel)
trueLabel=np.array(trueLabel)

cm4 = metrics.confusion_matrix(trueLabel, preLabel)
ACC4 = metrics.accuracy_score(trueLabel, preLabel)
Precision4 = metrics.precision_score(trueLabel, preLabel,average='weighted')
Recall4 = metrics.recall_score(trueLabel, preLabel,average='weighted')
F1_score4 = metrics.f1_score(trueLabel, preLabel,average='weighted')



###测试集降维结果，颜色表示正确标签
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
