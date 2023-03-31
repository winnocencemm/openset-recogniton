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
计算概率大小
'''
#
# trueTestSetVect=np.load('./phone3-8-gailv.npy')
# falseTestSetVect1=np.load('./phone9-gailv.npy')
# falseTestSetVect2=np.load('./phone10-gailv.npy')
# falseTestSetVect3=np.load('./phone11-gailv.npy')
# falseTestSetVect4=np.load('./phone12-gailv.npy')
# falseTestSetVect_USRP=np.load('./USRP-gailv.npy')

# Possibility=[]
# labelSet=[]
# for vect in trueTestSetVect:
#     tempPos=max(vect)
#     Possibility.append(tempPos)
#     labelSet.append(1)

# for vect in falseTestSetVect1:
#     tempPos=max(vect)
#     Possibility.append(tempPos)
#     labelSet.append(0)
#
# for vect in falseTestSetVect2:
#     tempPos=max(vect)
#     Possibility.append(tempPos)
#     labelSet.append(0)
# #
# for vect in falseTestSetVect3:
#     tempPos=max(vect)
#     Possibility.append(tempPos)
#     labelSet.append(0)
#
# for vect in falseTestSetVect4:
#     tempPos=max(vect)
#     Possibility.append(tempPos)
#     labelSet.append(0)

# for vect in falseTestSetVect_USRP:
#     tempPos=max(vect)
#     Possibility.append(tempPos)
#     labelSet.append(0)


# Possibility=np.array(Possibility)
# labelSet=np.array(labelSet)
#
# fpr, tpr, thresholds = metrics.roc_curve(labelSet,Possibility)
#
# AUC1=metrics.auc(fpr, tpr)
# #确定阈值
# maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
# threshold = thresholds[maxindex]
#
# plt.figure()
# plt.xlabel('FPR',fontsize=15)
# plt.ylabel('TPR',fontsize=15)
# x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# y=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# plt.xticks(x,fontsize=15)
# plt.yticks(y,fontsize=15)
# plt.plot(fpr,tpr,linewidth=2.5)
# plt.plot(fpr[maxindex],tpr[maxindex],'.',ms=15)



'''
计算向量之间余弦相似度
画ROC曲线
'''
#
# trainSetVect=np.load('./trainset-lastout.npy')
#
# trueTestSetVect=np.load('./phone3-8-lastout.npy')
# falseTestSetVect1=np.load('./phone9-lastout.npy')
# falseTestSetVect2=np.load('./phone10-lastout.npy')
# falseTestSetVect3=np.load('./phone11-lastout.npy')
# falseTestSetVect4=np.load('./phone12-lastout.npy')
# falseTestSetVect_USRP=np.load('./USRP-lastout.npy')
#
# Similarity=[]
# labelSet=[]
# for vect in trueTestSetVect:
#     tempSim=-1
#     for trainvect in trainSetVect:
#         tempSim=max(tempSim,cos_sim(vect,trainvect))
#     Similarity.append(tempSim)
#     labelSet.append(1)
#
#
# for vect in falseTestSetVect1:
#     tempSim=-1
#     for trainvect in trainSetVect:
#         tempSim=max(tempSim,cos_sim(vect,trainvect))
#     Similarity.append(tempSim)
#     labelSet.append(0)
# #
# for vect in falseTestSetVect2:
#     tempSim=-1
#     for trainvect in trainSetVect:
#         tempSim=max(tempSim,cos_sim(vect,trainvect))
#     Similarity.append(tempSim)
#     labelSet.append(0)
# #
# for vect in falseTestSetVect3:
#     tempSim=-1
#     for trainvect in trainSetVect:
#         tempSim=max(tempSim,cos_sim(vect,trainvect))
#     Similarity.append(tempSim)
#     labelSet.append(0)
# #
# for vect in falseTestSetVect4:
#     tempSim=-1
#     for trainvect in trainSetVect:
#         tempSim=max(tempSim,cos_sim(vect,trainvect))
#     Similarity.append(tempSim)
#     labelSet.append(0)
#
# for vect in falseTestSetVect_USRP:
#     tempSim=-1
#     for trainvect in trainSetVect:
#         tempSim=max(tempSim,cos_sim(vect,trainvect))
#     Similarity.append(tempSim)
#     labelSet.append(0)
# #
# Similarity=np.array(Similarity)
# labelSet=np.array(labelSet)
#
# ## fpr:即假正率，本来为负样本的样本被预测为正样本的总样本数量÷真实结果为负样本的总样本数
# ##tpr:即真正率或灵敏度或召回率或查全率或真正率或功效，
# # 本来为正样本的样本被预测为正样本的总样本数量÷真实结果为正样本的总样本数
# fpr, tpr, thresholds = metrics.roc_curve(labelSet,Similarity)
#
#
# AUC2=metrics.auc(fpr, tpr)
#
#
#
# #根据ROC曲线确定阈值
# fnr=1-tpr
# eer_index=np.nanargmin(np.absolute(fnr - fpr))
# threshold = thresholds[eer_index]
#
# # maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
# # threshold = thresholds[maxindex]
#
# #计算ACC,Precision,Recall,F1-score
# labelSet_pred=[]
# for sim in Similarity:
#     if sim>=threshold:
#         labelSet_pred.append(1)
#     else:
#         labelSet_pred.append(0)
# labelSet_pred=np.array(labelSet_pred)
#
# ACC=metrics.accuracy_score(labelSet,labelSet_pred)
# Precision=metrics.precision_score(labelSet,labelSet_pred)
# Recall=metrics.recall_score(labelSet,labelSet_pred)
# F1_score=metrics.f1_score(labelSet,labelSet_pred)
#
#
#
#
# plt.figure()
# plt.xlabel('FPR',fontsize=15)
# plt.ylabel('TPR',fontsize=15)
# x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# y=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# plt.xticks(x,fontsize=15)
# plt.yticks(y,fontsize=15)
# plt.plot(fpr,tpr,linewidth=2.5)
#
# x1 = np.arange(0, 1, 0.01)
# y1=[]
# for t in x1:
#     yt=1-t
#     y1.append(yt)
#
# plt.plot(y1,x1,linewidth=2.5)
# plt.plot(fpr[eer_index],tpr[eer_index],'.',ms=15)
#
#
# '''
# 画P-R曲线
# '''
# precision, recall, thresholds = metrics.precision_recall_curve(labelSet, Similarity)
#
# plt.figure()
# plt.xlabel('Recall',fontsize=15)
# plt.ylabel('Precision',fontsize=15)
#
# x=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# y=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
#
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.xticks(x,fontsize=15)
# plt.yticks(y,fontsize=15)
# plt.plot(recall,precision,linewidth=2.5)




'''
计算测试向量到训练向量集之间的马氏距离
'''
# trainSetVect=np.load('./trainset-lastout.npy')
#
# trueTestSetVect=np.load('./phone3-8-lastout.npy')
# falseTestSetVect1=np.load('./phone9-lastout.npy')
# falseTestSetVect2=np.load('./phone10-lastout.npy')
# falseTestSetVect3=np.load('./phone11-lastout.npy')
# falseTestSetVect4=np.load('./phone12-lastout.npy')
# falseTestSetVect_USRP=np.load('./USRP-lastout.npy')
# #
# #
# '''
# trainSetVectT:每列代表一个样本，每个样本6个元素
# CovVect：协方差矩阵 6*6
# invCovVect:协方差矩阵的逆矩阵
# '''
# trainSetVectT=trainSetVect.T
# CovVect=np.cov(trainSetVectT)
# invCovVect=np.linalg.inv(CovVect)
# # '''
# #
# # '''
# #
# '''
# MiuVect：均值向量 6*1
# '''
# MiuVect=np.mean(trainSetVectT,axis=1)
#
# '''
# MD:Mahalanobis Distance马氏距离
# '''
#
#
#
#
# MDSet=[]
# labelSet=[]
#
# for vect in trueTestSetVect:
#     vectT=vect.T
#     DifferenceVect=vectT-MiuVect
#     DifferenceVectT=DifferenceVect.T
#     temp=np.dot(DifferenceVectT,invCovVect)
#     MD=np.sqrt(np.dot(temp,DifferenceVect))
#     MDSet.append(MD)
#     labelSet.append(1)
#
# # for vect in falseTestSetVect1:
# #     vectT=vect.T
# #     DifferenceVect=vectT-MiuVect
# #     DifferenceVectT=DifferenceVect.T
# #     temp=np.dot(DifferenceVectT,invCovVect)
# #     MD=np.sqrt(np.dot(temp,DifferenceVect))
# #     MDSet.append(MD)
# #     labelSet.append(0)
#
#
# for vect in falseTestSetVect2:
#     vectT=vect.T
#     DifferenceVect=vectT-MiuVect
#     DifferenceVectT=DifferenceVect.T
#     temp=np.dot(DifferenceVectT,invCovVect)
#     MD=np.sqrt(np.dot(temp,DifferenceVect))
#     MDSet.append(MD)
#     labelSet.append(0)
#
# labelSet=np.array(labelSet)
# MDSet=np.array(MDSet)
#
#
# fpr, tpr, thresholds = metrics.roc_curve(labelSet,MDSet)
#
#
# AUC2=metrics.auc(fpr, tpr)
#
# #根据ROC曲线确定阈值
# maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
# threshold = thresholds[maxindex]
#
# #计算ACC,Precision,Recall,F1-score
# labelSet_pred=[]
# for MD in MDSet:
#     if MD>=threshold:
#         labelSet_pred.append(1)
#     else:
#         labelSet_pred.append(0)
# labelSet_pred=np.array(labelSet_pred)
#
#
#
# ACC=metrics.accuracy_score(labelSet,labelSet_pred)
# Precision=metrics.precision_score(labelSet,labelSet_pred)
# Recall=metrics.recall_score(labelSet,labelSet_pred)
# F1_score=metrics.f1_score(labelSet,labelSet_pred)
#
#
#
#
# plt.figure()
# plt.xlabel('FPR',fontsize=15)
# plt.ylabel('TPR',fontsize=15)
# x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# y=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# plt.xticks(x,fontsize=15)
# plt.yticks(y,fontsize=15)
# plt.plot(fpr,tpr,linewidth=2.5)
# plt.plot(fpr[maxindex],tpr[maxindex],'.',ms=15)
#
#
# '''
# 画P-R曲线
# '''
# precision, recall, thresholds = metrics.precision_recall_curve(labelSet, MDSet)
#
# plt.figure()
# plt.xlabel('Recall',fontsize=15)
# plt.ylabel('Precision',fontsize=15)
#
# x=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# y=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
#
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.xticks(x,fontsize=15)
# plt.yticks(y,fontsize=15)
# plt.plot(recall,precision,linewidth=2.5)
# #


'''
计算欧氏距离
'''


# trainSetVect=np.load('../vec/trainout_MAXMINLOSS_2023_3_27.npy')

# trueTestSetVect=np.load('../vec/testout_MAXMINLOSS_2023_3_27_784test.npy')
# falseTestSetVect1=np.load('../vec/phone9out_MAXMINLOSS_2023_3_27.npy')
# falseTestSetVect2=np.load('../vec/phone10out_MAXMINLOSS_2023_3_27.npy')
# falseTestSetVect3=np.load('../vec/phone11out_MAXMINLOSS_2023_3_27.npy')
# falseTestSetVect4=np.load('../vec/phone12out_MAXMINLOSS_2023_3_27.npy')

trainSetVect=np.load('../vec/trainout_STATE_15dB_MAXMINLOSS_2023_3_29.npy')
trueTestSetVect=np.load('../vec/testout_STATE_15dB_MAXMINLOSS_2023_3_29.npy')
falseTestSetVect1=np.load('../vec/phone9out_STATE_15dB_MAXMINLOSS_2023_3_29.npy')
falseTestSetVect2=np.load('../vec/phone10out_STATE_15dB_MAXMINLOSS_2023_3_29.npy')
falseTestSetVect3=np.load('../vec/phone11out_STATE_15dB_MAXMINLOSS_2023_3_29.npy')
falseTestSetVect4=np.load('../vec/phone12out_STATE_15dB_MAXMINLOSS_2023_3_29.npy')

# falseTestSetVect_USRP=np.load('./USRP-lastout.npy')

Similarity=[]
labelSet=[]
for vect in trueTestSetVect:
    tempSim=-float("inf")
    for trainvect in trainSetVect:
        tempSim=max(tempSim,-euclid_distance(vect,trainvect))
    Similarity.append(tempSim)
    labelSet.append(1)


for vect in falseTestSetVect1:
    tempSim=-float("inf")
    for trainvect in trainSetVect:
        tempSim=max(tempSim,-euclid_distance(vect,trainvect))
    Similarity.append(tempSim)
    labelSet.append(0)
#
for vect in falseTestSetVect2:
    tempSim=-float("inf")
    for trainvect in trainSetVect:
        tempSim=max(tempSim,-euclid_distance(vect,trainvect))
    Similarity.append(tempSim)
    labelSet.append(0)
#
for vect in falseTestSetVect3:
    tempSim=-float("inf")
    for trainvect in trainSetVect:
        tempSim=max(tempSim,-euclid_distance(vect,trainvect))
    Similarity.append(tempSim)
    labelSet.append(0)
#
for vect in falseTestSetVect4:
    tempSim=-float("inf")
    for trainvect in trainSetVect:
        tempSim=max(tempSim,-euclid_distance(vect,trainvect))
    Similarity.append(tempSim)
    labelSet.append(0)

# for vect in falseTestSetVect_USRP:
#     tempSim=-float("inf")
#     for trainvect in trainSetVect:
#         tempSim=max(tempSim,-euclid_distance(vect,trainvect))
#     Similarity.append(tempSim)
#     labelSet.append(0)
#
Similarity=np.array(Similarity)
labelSet=np.array(labelSet)

## fpr:即假正率，本来为负样本的样本被预测为正样本的总样本数量÷真实结果为负样本的总样本数
##tpr:即真正率或灵敏度或召回率或查全率或真正率或功效，
# 本来为正样本的样本被预测为正样本的总样本数量÷真实结果为正样本的总样本数
fpr, tpr, thresholds = metrics.roc_curve(labelSet,Similarity)


AUC2=metrics.auc(fpr, tpr)



#根据ROC曲线确定阈值
fnr=1-tpr
eer_index=np.nanargmin(np.absolute(fnr - fpr))
threshold = thresholds[eer_index]

# maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
# threshold = thresholds[maxindex]

#计算ACC,Precision,Recall,F1-score
labelSet_pred=[]
for sim in Similarity:
    if sim>=threshold:
        labelSet_pred.append(1)
    else:
        labelSet_pred.append(0)
labelSet_pred=np.array(labelSet_pred)

ACC=metrics.accuracy_score(labelSet,labelSet_pred)
Precision=metrics.precision_score(labelSet,labelSet_pred)
Recall=metrics.recall_score(labelSet,labelSet_pred)
F1_score=metrics.f1_score(labelSet,labelSet_pred)
print("ACC:"+str(ACC))
print("Precision:"+str(Precision))
print("Recall:"+str(Recall))
print("F1_score:"+str(F1_score))



plt.figure()
plt.xlabel('FPR',fontsize=15)
plt.ylabel('TPR',fontsize=15)
plt.title('AUC='+str(AUC2))
x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks(x,fontsize=15)
plt.yticks(y,fontsize=15)
plt.plot(fpr,tpr,linewidth=2.5)
plt.show()
x1 = np.arange(0, 1, 0.01)
y1=[]
for t in x1:
    yt=1-t
    y1.append(yt)

plt.plot(y1,x1,linewidth=2.5)
plt.plot(fpr[eer_index],tpr[eer_index],'.',ms=15)


'''
画P-R曲线
'''
precision, recall, thresholds = metrics.precision_recall_curve(labelSet, Similarity)

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
plt.show()