import logging
import scipy.io as scio
import dspFunctions as dspf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from scipy import signal
import math
import os
from scipy.spatial.distance import pdist
from sklearn import metrics
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score, average_precision_score
'''
定义计算余弦相似度函数
'''
def cos_sim(a,b):
    a_norm=np.linalg.norm(a)
    b_norm=np.linalg.norm(b)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos

'''
定义计算欧式距离函数
'''
def euclid_distance(a,b):
    distance=np.sqrt(np.sum(np.square(a - b)))
    return distance

'''
定义KNN
'''
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=20)



'''
计算向量之间余弦相似度
画ROC曲线
画PR曲线
'''
#
trainSetVect=np.load('./vec_train_lte4-8.npy')
trueTestSetVect=np.load('./vec_test_lte4-8.npy')
falseTestSetVect=np.load('./vec_test_lte3.npy')


#
Similarity=[]
labelSet=[]

for vect in trueTestSetVect:
    tempSim=-1
    for trainvect in trainSetVect:
        tempSim=max(tempSim,cos_sim(vect,trainvect))
    Similarity.append(tempSim)
    labelSet.append(1)


for vect in falseTestSetVect:
    tempSim=-1
    for trainvect in trainSetVect:
        tempSim=max(tempSim,cos_sim(vect,trainvect))
    Similarity.append(tempSim)
    labelSet.append(0)



# '''
# KNN思想+欧式
# 画ROC曲线
# 画PR曲线
# '''
# #
# trainSetVect=np.load('./vec_train_lte4-8.npy')
# trueTestSetVect=np.load('./vec_test_lte4-8.npy')
# falseTestSetVect=np.load('./vec_test_lte3.npy')
#
#
# #
# Similarity=[]
# labelSet=[]
#
# for vect in trueTestSetVect:
#     Distance = []
#     for trainvect in trainSetVect:
#         Distance.append(np.linalg.norm(vect - trainvect))
#     Distance = sorted(Distance)
#     min_Dis = np.mean(Distance[:20])
#     Similarity.append(-min_Dis)
#     labelSet.append(1)
#
#
# for vect in falseTestSetVect:
#     Distance = []
#     for trainvect in trainSetVect:
#         Distance.append(np.linalg.norm(vect - trainvect))
#     Distance = sorted(Distance)
#     min_Dis = np.mean(Distance[:20])
#     Similarity.append(-min_Dis)
#     labelSet.append(0)


# '''
#     欧式
#     画ROC曲线
#     画PR曲线
# '''
# #
# trainSetVect=np.load('./vec_train_lte4-8.npy')
# trueTestSetVect=np.load('./vec_test_lte4-8.npy')
# falseTestSetVect=np.load('./vec_test_lte3.npy')
#
#
# #
# Similarity=[]
# labelSet=[]
#
# for vect in trueTestSetVect:
#     tempSim = -float("inf")
#     for trainvect in trainSetVect:
#         tempSim = max(tempSim, -np.linalg.norm(vect - trainvect))
#     Similarity.append(tempSim)
#     labelSet.append(1)
#
#
# for vect in falseTestSetVect:
#     tempSim = -float("inf")
#     for trainvect in trainSetVect:
#         tempSim = max(tempSim, -np.linalg.norm(vect - trainvect))
#     Similarity.append(tempSim)
#     labelSet.append(0)

Similarity=np.array(Similarity)
labelSet=np.array(labelSet)


## fpr:即假正率，本来为负样本的样本被预测为正样本的总样本数量÷真实结果为负样本的总样本数
##tpr:即真正率或灵敏度或召回率或查全率或真正率或功效，本来为正样本的样本被预测为正样本的总样本数量÷真实结果为正样本的总样本数
fpr, tpr, thresholds = metrics.roc_curve(labelSet,Similarity)
AUC1=metrics.auc(fpr, tpr)

#
#
#
#根据ROC曲线确定阈值
fnr=1-tpr
eer_index=np.nanargmin(np.absolute(fnr - fpr))
EER1=fpr[eer_index]
threshold = thresholds[eer_index]

#计算ACC,Precision,Recall,F1-score
labelSet_pred=[]
for sim in Similarity:
    if sim>=threshold:
        labelSet_pred.append(1)
    else:
        labelSet_pred.append(0)
labelSet_pred=[]
for sim in Similarity:
    if sim>=threshold:
        labelSet_pred.append(1)
    else:
        labelSet_pred.append(0)
labelSet_pred=np.array(labelSet_pred)

ACC1=metrics.accuracy_score(labelSet,labelSet_pred)
Precision1=metrics.precision_score(labelSet,labelSet_pred)
Recall1=metrics.recall_score(labelSet,labelSet_pred)
F1_score1=metrics.f1_score(labelSet,labelSet_pred)
auc_score = roc_auc_score(labelSet,labelSet_pred)

'''
画ROC曲线
'''
plt.figure()
plt.xlabel('FPR',fontsize=15)
plt.ylabel('TPR',fontsize=15)
x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.title('lte3_'+'AUC='+str(auc_score))
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(fpr,tpr,linewidth=2.5)
plt.show()

'''
画P-R曲线
'''
precision, recall, thresholds = metrics.precision_recall_curve(labelSet, Similarity)
AP=metrics.average_precision_score(labelSet,labelSet_pred)

plt.figure()
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision',fontsize=15)

x=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

plt.xlim([0,1])
plt.ylim([0,1])
plt.title('lte3_'+'AP='+str(AP))
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(recall,precision,linewidth=2.5)
plt.show()













