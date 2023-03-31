import logging
import scipy.io as scio
import dspFunctions as dspf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from scipy import signal
import math
import os





'''
##2021-12-08
'''


sampleRate=16e6
Syn_Length = 1651
Syn_Interval = 12800


##导入经过同步后的前导信号
filepath='/rach'
filenamelist=os.listdir(filepath)
qiandaolist=[]


for i in range(len(filenamelist)):
    qiandao = np.load('/rach'+filenamelist[i])
    qiandao=qiandao/np.mean(np.abs(qiandao))
    # b,a=signal.butter(6,3e6/(sampleRate/2))
    # qiandao= signal.lfilter(b, a, qiandao)
    qiandaolist.append(qiandao)

featurelist1=[]
featurelist2=[]
##featurelist3=[]
for i in range(len(filenamelist)):
    feature1=qiandaolist[i][1450:1550]
    feature2=qiandaolist[i][15900:16000]
    ##feature3=qiandaolist[i][1500:1700]
    featurelist1.append(feature1)
    featurelist2.append(feature2)
    ##featurelist3.append(feature3)

# # # # # #
PLOT_TABLE_SIZE =64
PLOT_TABLE_MAX = 3

for i in range(len(filenamelist)):
#for i in range(1):
    START=(np.tile(featurelist1[i], (1, 100)))[0]
    END = (np.tile(featurelist2[i], (1, 100)))[0]
    STATE=qiandaolist[i][1550:15900]
    ALL=qiandaolist[i]

    '''
    前瞬态DCTF
    '''
    START_difference=dspf.getDifference(START,16)
    plotTable = dspf.getPlotTable(PLOT_TABLE_SIZE, PLOT_TABLE_MAX, START_difference)
    r, g, b = dspf.grayColor(plotTable)
    img = dspf.imgArray(r, g, b)
    #mimg.imsave('START/'+filenamelist[i][0:-4]+'.png',img,format='png')
    # plt.figure()
    # plt.imshow(img)


    '''
    尾瞬态DCTF
    '''
    END_difference=dspf.getDifference(END,16)
    plotTable = dspf.getPlotTable(PLOT_TABLE_SIZE, PLOT_TABLE_MAX, END_difference)
    r, g, b = dspf.grayColor(plotTable)
    img = dspf.imgArray(r, g, b)
    #mimg.imsave('END/'+filenamelist[i][0:-4]+'.png',img,format='png')
    # plt.figure()
    # plt.imshow(img)


    '''
    稳态DCTF
    '''
    STATE_difference=dspf.getDifference(STATE,16)
    plotTable = dspf.getPlotTable(PLOT_TABLE_SIZE, PLOT_TABLE_MAX, STATE_difference)
    r, g, b = dspf.grayColor(plotTable)
    img = dspf.imgArray(r, g, b)
    #mimg.imsave('STATE/'+filenamelist[i][0:-4]+'.png',img,format='png')
    # plt.figure()
    # plt.imshow(img)

    '''
    完整信号DCTF
    '''
    ALL_difference = dspf.getDifference(ALL, 16)
    plotTable = dspf.getPlotTable(PLOT_TABLE_SIZE, PLOT_TABLE_MAX, ALL_difference)
    r, g, b = dspf.grayColor(plotTable)
    img = dspf.imgArray(r, g, b)
    #mimg.imsave('ALL/'+filenamelist[i][0:-4] + '.png', img, format='png')
    # plt.figure()
    # plt.imshow(img)