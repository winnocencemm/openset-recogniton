import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk
import matplotlib.image as mimg
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei']
IMAGESIZE=129

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
TrainPath='D:/文档/项目相关资料代码/4GLTE/LTE-RACH-FROM-WZN/final/Indoor/train'
'''
读取训练集图片，得到矩阵TrainMat 和其转置 TrainMatT
'''

##使用一个DCTF图像

switch='/ALL'
#switch='/autocollectaftercoursesync_DCTF_ALL'
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


n_components=100
#n_components=1500
pca=PCA(n_components=n_components)
MYPCA=pca.fit(TrainMat)
TrainFeatureVects=MYPCA.transform(TrainMat)





# n_components=6
# pca=PCA(n_components=n_components)
# MYPCA=pca.fit(TrainMat[trainLabel!=5])
# TrainFeatureVects=MYPCA.transform(TrainMat[trainLabel!=5])




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


#print(pca.explained_variance_ratio_)

# tempsum=0
# feature_percent=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99]
# j=0
# feature_weishu=[]
# for i in range(len(pca.explained_variance_ratio_)):
#     tempsum=tempsum+pca.explained_variance_ratio_[i]
#     if j<=11 and tempsum>=feature_percent[j] :
#         j+=1
#         feature_weishu.append(i)


'''
识别过程
'''

'''
首先生成测试集的特征向量
'''
#TestPath='D:/文档/项目相关资料代码/4GLTE/LTE-RACH-FROM-WZN/diff-loc/Train and Test/DCTF/test'
#TestPath='D:/文档/项目相关资料代码/4GLTE/lte_rach/final/Indoor/30dB/TEST'
TrueTestPath='D:/文档/项目相关资料代码/4GLTE/LTE-RACH-FROM-WZN/final/Indoor/test'
FalseTestPath='D:/文档/项目相关资料代码/4GLTE/LTE-RACH-FROM-WZN/final/phone9-12'
TrueTestFileNames = os.listdir(TrueTestPath+switch)
FalseTestFileNames = os.listdir(FalseTestPath+switch)

trueLabel=[]

TestFeatureVects=[]
TestMat=[]
for filename in TrueTestFileNames:
    ##为合法设备
    trueLabel.append(1)
    """
    一个DCTF
    """

    image = mimg.imread(TrueTestPath+ switch + '/' + filename)
    imageresize = image.reshape(image.size, 1)
    TestMat.append(imageresize)

for filename in FalseTestFileNames:
    ##为非法设备
    trueLabel.append(0)
    """
    一个DCTF
    """

    image = mimg.imread(FalseTestPath+ switch + '/' + filename)
    imageresize = image.reshape(image.size, 1)
    TestMat.append(imageresize)


TestMat=np.array(TestMat)
TestMat=TestMat.reshape(TestMat.shape[0],TestMat.shape[1])

TestFeatureVects=MYPCA.transform(TestMat)


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
SimilarityList=[]
for testvect in TestFeatureVects:

    Similarity=[]
    for trainvect in TrainFeatureVects:
        Similarity.append(cos_sim(testvect,trainvect))

    max_Sim=max(Similarity)
    SimilarityList.append(max_Sim)

SimilarityList=np.array(SimilarityList)
trueLabel=np.array(trueLabel)

fpr, tpr, thresholds = metrics.roc_curve(trueLabel,SimilarityList)

AUC1=metrics.auc(fpr, tpr)

#根据ROC曲线确定阈值
fnr=1-tpr
eer_index=np.nanargmin(np.absolute(fnr - fpr))
threshold = thresholds[eer_index]


#计算ACC,Precision,Recall,F1-score
preLabel=[]
for sim in SimilarityList:
    if sim>=threshold:
        preLabel.append(1)
    else:
        preLabel.append(0)
preLabel=np.array(preLabel)

ACC1=metrics.accuracy_score(trueLabel,preLabel)
Precision1=metrics.precision_score(trueLabel,preLabel)
Recall1=metrics.recall_score(trueLabel,preLabel)
F1_score1=metrics.f1_score(trueLabel,preLabel)

plt.figure()
plt.xlabel('FPR',fontsize=15)
plt.ylabel('TPR',fontsize=15)
x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(fpr,tpr,linewidth=2.5)

x1 = np.arange(0, 1, 0.01)
y1=[]
for t in x1:
    yt=1-t
    y1.append(yt)

plt.plot(y1,x1,linewidth=2.5)
plt.plot(fpr[eer_index],tpr[eer_index],'.',ms=15)

##画P-R曲线
precision, recall, thresholds = metrics.precision_recall_curve(trueLabel, SimilarityList)

plt.figure()
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision',fontsize=15)

x=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(recall,precision,linewidth=2.5)

'''
根据2-范数+KNN思想
'''

preLabel=[]
SimilarityList=[]
for testvect in TestFeatureVects:

    Distance=[]
    for trainvect in TrainFeatureVects:
        Distance.append(np.linalg.norm(testvect-trainvect))

    Distance=sorted(Distance)
    min_Dis=np.mean(Distance[:20])
    SimilarityList.append(-min_Dis)

SimilarityList=np.array(SimilarityList)
trueLabel=np.array(trueLabel)

fpr, tpr, thresholds = metrics.roc_curve(trueLabel,SimilarityList)

AUC2=metrics.auc(fpr, tpr)

#根据ROC曲线确定阈值
fnr=1-tpr
eer_index=np.nanargmin(np.absolute(fnr - fpr))
threshold = thresholds[eer_index]


#计算ACC,Precision,Recall,F1-score
preLabel=[]
for sim in SimilarityList:
    if sim>=threshold:
        preLabel.append(1)
    else:
        preLabel.append(0)
preLabel=np.array(preLabel)

ACC2=metrics.accuracy_score(trueLabel,preLabel)
Precision2=metrics.precision_score(trueLabel,preLabel)
Recall2=metrics.recall_score(trueLabel,preLabel)
F1_score2=metrics.f1_score(trueLabel,preLabel)

plt.figure()
plt.xlabel('FPR',fontsize=15)
plt.ylabel('TPR',fontsize=15)
x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(fpr,tpr,linewidth=2.5)

x1 = np.arange(0, 1, 0.01)
y1=[]
for t in x1:
    yt=1-t
    y1.append(yt)

plt.plot(y1,x1,linewidth=2.5)
plt.plot(fpr[eer_index],tpr[eer_index],'.',ms=15)

##画P-R曲线
precision, recall, thresholds = metrics.precision_recall_curve(trueLabel, SimilarityList)

plt.figure()
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision',fontsize=15)

x=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(recall,precision,linewidth=2.5)




'''
2-范数
'''
preLabel=[]
SimilarityList=[]
for testvect in TestFeatureVects:

    Distance=[]
    for trainvect in TrainFeatureVects:
        Distance.append(np.linalg.norm(testvect-trainvect))

    min_Dis=min(Distance)
    SimilarityList.append(-min_Dis)

SimilarityList=np.array(SimilarityList)
trueLabel=np.array(trueLabel)

fpr, tpr, thresholds = metrics.roc_curve(trueLabel,SimilarityList)

AUC3=metrics.auc(fpr, tpr)

#根据ROC曲线确定阈值
fnr=1-tpr
eer_index=np.nanargmin(np.absolute(fnr - fpr))
threshold = thresholds[eer_index]


#计算ACC,Precision,Recall,F1-score
preLabel=[]
for sim in SimilarityList:
    if sim>=threshold:
        preLabel.append(1)
    else:
        preLabel.append(0)
preLabel=np.array(preLabel)

ACC3=metrics.accuracy_score(trueLabel,preLabel)
Precision3=metrics.precision_score(trueLabel,preLabel)
Recall3=metrics.recall_score(trueLabel,preLabel)
F1_score3=metrics.f1_score(trueLabel,preLabel)

plt.figure()
plt.xlabel('FPR',fontsize=15)
plt.ylabel('TPR',fontsize=15)
x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(fpr,tpr,linewidth=2.5)

x1 = np.arange(0, 1, 0.01)
y1=[]
for t in x1:
    yt=1-t
    y1.append(yt)

plt.plot(y1,x1,linewidth=2.5)
plt.plot(fpr[eer_index],tpr[eer_index],'.',ms=15)

##画P-R曲线
precision, recall, thresholds = metrics.precision_recall_curve(trueLabel, SimilarityList)

plt.figure()
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision',fontsize=15)

x=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(recall,precision,linewidth=2.5)


'''
1-范数
'''
preLabel=[]
SimilarityList=[]
for testvect in TestFeatureVects:

    Distance=[]
    for trainvect in TrainFeatureVects:
        Distance.append(np.linalg.norm(testvect-trainvect,ord=1))


    min_Dis=min(Distance)
    SimilarityList.append(-min_Dis)

SimilarityList=np.array(SimilarityList)
trueLabel=np.array(trueLabel)

fpr, tpr, thresholds = metrics.roc_curve(trueLabel,SimilarityList)

AUC4=metrics.auc(fpr, tpr)

#根据ROC曲线确定阈值
fnr=1-tpr
eer_index=np.nanargmin(np.absolute(fnr - fpr))
threshold = thresholds[eer_index]


#计算ACC,Precision,Recall,F1-score
preLabel=[]
for sim in SimilarityList:
    if sim>=threshold:
        preLabel.append(1)
    else:
        preLabel.append(0)
preLabel=np.array(preLabel)

ACC4=metrics.accuracy_score(trueLabel,preLabel)
Precision4=metrics.precision_score(trueLabel,preLabel)
Recall4=metrics.recall_score(trueLabel,preLabel)
F1_score4=metrics.f1_score(trueLabel,preLabel)

plt.figure()
plt.xlabel('FPR',fontsize=15)
plt.ylabel('TPR',fontsize=15)
x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(fpr,tpr,linewidth=2.5)

x1 = np.arange(0, 1, 0.01)
y1=[]
for t in x1:
    yt=1-t
    y1.append(yt)

plt.plot(y1,x1,linewidth=2.5)
plt.plot(fpr[eer_index],tpr[eer_index],'.',ms=15)

##画P-R曲线
precision, recall, thresholds = metrics.precision_recall_curve(trueLabel, SimilarityList)

plt.figure()
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision',fontsize=15)

x=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(recall,precision,linewidth=2.5)






'''
OneClassSVM
'''
from sklearn.svm import OneClassSVM
ocsvm=OneClassSVM(kernel='poly')
MYSVM=ocsvm.fit(TrainFeatureVects)
SimilarityList=MYSVM.score_samples(TestFeatureVects)


SimilarityList=np.array(SimilarityList)
trueLabel=np.array(trueLabel)

fpr, tpr, thresholds = metrics.roc_curve(trueLabel,SimilarityList)

AUC5=metrics.auc(fpr, tpr)

#根据ROC曲线确定阈值
fnr=1-tpr
eer_index=np.nanargmin(np.absolute(fnr - fpr))
threshold = thresholds[eer_index]


#计算ACC,Precision,Recall,F1-score
preLabel=[]
for sim in SimilarityList:
    if sim>=threshold:
        preLabel.append(1)
    else:
        preLabel.append(0)
preLabel=np.array(preLabel)

ACC5=metrics.accuracy_score(trueLabel,preLabel)
Precision5=metrics.precision_score(trueLabel,preLabel)
Recall5=metrics.recall_score(trueLabel,preLabel)
F1_score5=metrics.f1_score(trueLabel,preLabel)

plt.figure()
plt.xlabel('FPR',fontsize=15)
plt.ylabel('TPR',fontsize=15)
x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(fpr,tpr,linewidth=2.5)

x1 = np.arange(0, 1, 0.01)
y1=[]
for t in x1:
    yt=1-t
    y1.append(yt)

plt.plot(y1,x1,linewidth=2.5)
plt.plot(fpr[eer_index],tpr[eer_index],'.',ms=15)

##画P-R曲线
precision, recall, thresholds = metrics.precision_recall_curve(trueLabel, SimilarityList)

plt.figure()
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision',fontsize=15)

x=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(recall,precision,linewidth=2.5)


'''
LOF 
'''
from sklearn.neighbors import LocalOutlierFactor
clf=LocalOutlierFactor(novelty=True)
clf.fit(TrainFeatureVects)
SimilarityList=clf.score_samples(TestFeatureVects)


SimilarityList=np.array(SimilarityList)
trueLabel=np.array(trueLabel)

fpr, tpr, thresholds = metrics.roc_curve(trueLabel,SimilarityList)

AUC6=metrics.auc(fpr, tpr)

#根据ROC曲线确定阈值
fnr=1-tpr
eer_index=np.nanargmin(np.absolute(fnr - fpr))
threshold = thresholds[eer_index]


#计算ACC,Precision,Recall,F1-score
preLabel=[]
for sim in SimilarityList:
    if sim>=threshold:
        preLabel.append(1)
    else:
        preLabel.append(0)
preLabel=np.array(preLabel)

ACC6=metrics.accuracy_score(trueLabel,preLabel)
Precision6=metrics.precision_score(trueLabel,preLabel)
Recall6=metrics.recall_score(trueLabel,preLabel)
F1_score6=metrics.f1_score(trueLabel,preLabel)

plt.figure()
plt.xlabel('FPR',fontsize=15)
plt.ylabel('TPR',fontsize=15)
x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(fpr,tpr,linewidth=2.5)

x1 = np.arange(0, 1, 0.01)
y1=[]
for t in x1:
    yt=1-t
    y1.append(yt)

plt.plot(y1,x1,linewidth=2.5)
plt.plot(fpr[eer_index],tpr[eer_index],'.',ms=15)

##画P-R曲线
precision, recall, thresholds = metrics.precision_recall_curve(trueLabel, SimilarityList)

plt.figure()
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision',fontsize=15)

x=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(recall,precision,linewidth=2.5)



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

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(TestFeatureVects[trueLabel==1,0],TestFeatureVects[trueLabel==1,1],TestFeatureVects[trueLabel==1,2],color='blue',marker='o',alpha=1,label='测试集中的合法手机')
ax.scatter(TestFeatureVects[trueLabel==0,0],TestFeatureVects[trueLabel==0,1],TestFeatureVects[trueLabel==0,2],color='red',marker='o',alpha=1,label='测试集中的非法手机')
ax.scatter(TrainFeatureVects[:,0],TrainFeatureVects[:,1],TrainFeatureVects[:,2],color='green',marker='o',alpha=1,label='训练集，均为合法手机')
ax.legend()

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(TestFeatureVects[trueLabel==preLabel,0],TestFeatureVects[trueLabel==preLabel,1],TestFeatureVects[trueLabel==preLabel,2],color='blue',marker='o',alpha=1,label='测试集中识别正确的手机')
ax.scatter(TestFeatureVects[trueLabel!=preLabel,0],TestFeatureVects[trueLabel!=preLabel,1],TestFeatureVects[trueLabel!=preLabel,2],color='red',marker='o',alpha=1,label='测试集中识别错误的手机')
ax.legend()