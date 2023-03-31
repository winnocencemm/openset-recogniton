import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import os
def normalize(vector):
    return vector / np.sqrt(np.sum(np.square(np.abs(vector))) / vector.size)


def normalizeBurst(bursts):
    # TODO 将这部分加到提取帧的时候一并完成
    energyThreArray = np.mean(np.abs(bursts), axis=1)
    for i, burst in enumerate(bursts):
        # 排除截取的帧两端多余信号对归一化的影响
        burstTmp = burst[np.where(np.abs(burst) > energyThreArray[i])[0]]
        meanEnergy = np.sqrt(np.sum(np.square(np.abs(burstTmp))) / burstTmp.size)
        bursts[i] = bursts[i] / meanEnergy
    # # 不使用循环的实现，发现使用循环似乎更快
    # burstsTmp = [list(bursts[i][np.where(bursts[i] > energyThreArray[i])]) \
    #     for i in range(len(energyThreArray))]
    # meanEnergyArray = np.array([np.sqrt(np.sum(np.square(np.abs( \
    #     np.array(burst)))) / np.array(burst).size) \
    #         for burst in burstsTmp]).reshape(len(bursts), -1)
    # bursts = bursts / meanEnergyArray
    return bursts


def getAverageSpectrum(data, fftSize=4096, startPos=0):
    """
    返回输入数据的 fft 平均值

    Parameters
    ----------
    data : np.ndarray
        输入数据
    fftSize : int, optional
        fft 点数, by default 4906
    startPos : int, optional
        起始点位置, by default 0

    Returns
    -------
    np.ndarray
        输出结果
    """
    # fftResult = np.zeros(fftSize, dtype=np.complex64)
    # dataLength = len(data)
    # loops = np.floor(dataLength / fftSize)
    # for i in range(0, loops):
    #     fftTmps = np.fft.fft(data[i*fftSize : (i+1)*fftSize], fftSize)
    #     fftResult += np.abs(fftTmps)
    # return fftResult / loops
    fftResult = np.zeros(fftSize, dtype=np.complex64)
    counter = 0
    for i in range(startPos, len(data), fftSize):
        fftResult += np.abs(np.fft.fft(data[i:i + fftSize], fftSize))
        counter += 1
    fftResult = np.real(fftResult)
    return fftResult / counter


def getAmplitudeResponse(data, fftSize, sampleRate, representBydb=True, shift=True):
    '''
    输出频率响应
    '''
    fftResult = getAverageSpectrum(data, fftSize)
    if representBydb:
        fftResult = 10 * np.log10(fftResult)
    # 将零频移动到频谱中心
    if shift:
        fftResult = np.fft.fftshift(fftResult)
    # 频率分辨率
    deltaFreq = sampleRate / fftSize
    freq = np.arange(0, sampleRate, deltaFreq)
    freq -= sampleRate / 2
    return freq, fftResult

def getAmplitudeResponse2(data, fftSize, sampleRate, bw=None, representBydb=True, shift=True):
    '''
    输出频率响应
    '''
    if not bw:
        bw = sampleRate
    fftResult = getAverageSpectrum(data, fftSize)
    if representBydb:
        fftResult = 10 * np.log10(fftResult)
    # 将零频移动到频谱中心
    if shift:
        fftResult = np.fft.fftshift(fftResult)
    # 频率分辨率
    deltaFreq = bw / fftSize
    freq = np.arange(0, bw, deltaFreq)
    freq -= bw / 2   #相当于freq=freq-(bw/2)
    return freq, fftResult

def getDataAfterSpectrumShift(data, sampleRate, frequencyPointOffset, expectedFrequencyPoint=0.0):
    """
    频谱搬移

    Parameters
    ----------
    data : np.ndarray
        原始数据
    sampleRate : float
        采样率
    frequencyPointOffset : float
        频率偏移值
    expectedFrequencyPoint : float, optional
        期望搬移后的频点, by default 0.0

    Returns
    -------
    np.ndarray
        搬移后的数据
    """
    compensationFactor = (frequencyPointOffset - expectedFrequencyPoint) / sampleRate
    dataLength = len(data)
    # dataTmp = np.zeros(dataLength, dtype=np.complex64)
    # for n in range(dataLength):
    #     dataTmp[n] = data[n] * np.exp(-1j * 2 * np.pi * (n + 1) * compensationFactor)  # 注意 n 的取值。
    # 使用矩阵运算加快运算速度
    expTmp = np.array([np.exp(-1j * 2 * np.pi * (n + 1) * compensationFactor) for n in range(dataLength)])
    dataTmp = data * expTmp
    # data = data * expTmp
    return dataTmp


def findThre(data, a=100):
    '''
    用最大类间方差法自动寻找阈值，将data归一化到[1，a]
    在高信噪比的条件下，一般计算结果为 thre = 1，可从这点优化该函数。
    '''
    data = np.abs(data)
    k = (a - 1) / float((np.max(data) - np.min(data)))
    dataT = data
    data = np.round((data - np.min(data)) * k)
    p = np.zeros(a)
    # 计算概率分布
    for i in range(a):
        p[i] = np.where(data == i)[0].size / float(data.size)
    g = np.zeros(a)
    for i in range(a):
        n0 = np.where(data < i)[0].size
        # 计算每一类的出现概率
        w0 = n0 / float(data.size)
        w1 = 1 - w0
        # 分别计算两类的平均灰度
        u0 = 0
        for j in range(i + 1):
            u0 += (j + 1) * p[j]
        u1 = 0
        for j in range(i + 1, a):
            u1 += (j + 1) * p[j]
        # 计算类间方差的总方差
        # if w0!=0 and w1!=0:
        #     u0=u0/w0
        #     u1=u1/w1
        g[i] = w0 * w1 * (u0 - u1)**2
    thre = np.where(g == np.max(g))[0][0]
    return (thre / k + np.min(dataT))


def findAverageThre(data, thre, k=1):
    """
    返回阈值
    
    Parameters
    ----------
    data : np.ndarray
        输入数据
    thre : float
        阈值
    k : int, optional
        放大系数, by default 1
    
    Returns
    -------
    float
        平均阈值
    """
    return np.mean(np.abs(data[np.abs(data) < thre])) * k


def getValidDataPos1(data, thre, rang=50, validMinLength=3000):
    """
    获取原始采样数据中有效信号的区域范围，
    与 getValidDataPos2 相比，有可能会多提取到一个不完整的数据段

    Parameters
    ----------
    data : np.ndarray
        原始采样数据
    thre : float
        阈值
    rang : int, optional
        滑动窗口大小, by default 50
    validMinLength : int, optional
        有效数据段的最短长度, by default 300

    Returns
    -------
    np.ndarray
        有效信号的索引范围，返回 (0, -1)，表示没有获取到有效数据段
    """
    data = np.abs(data)
    boundaries = []
    lowBoundary = -1
    highBoundary = -1
    for i, d in enumerate(data):
        if i > highBoundary:
            if d > thre:
                validDataPos = np.where(data[i:i + rang] > thre)
                validDataNum = len(validDataPos[0])
                if validDataNum > int(rang * 0.5):
                    lowBoundary = i
                    for j in range(i + rang, len(data)):
                        validDataPos = np.where(data[j:j + rang] > thre)
                        validDataNum = len(validDataPos[0])
                        if validDataNum < 3:
                            highBoundary = j
                            break
                    if highBoundary - lowBoundary < validMinLength:
                        break
                    boundaries.append((lowBoundary, highBoundary))
    if len(boundaries) == 0:
        boundaries.append((0, len(boundaries) - 1))
    return np.array(boundaries)


def getValidDataPos2(data, thre, validMinLength=3000):
    """
    获取原始采样数据中有效信号的区域范围，该算法无法获取单个的帧

    Parameters
    ----------
    data : np.ndarray
        原始采样数据
    thre : float
        阈值
    validMinLength : int, optional
        有效数据段的最短长度, by default 300

    Returns
    -------
    np.ndarray
        有效信号的索引范围，返回 (0, -1)，表示没有获取到有效数据段
    """
    data = np.abs(data)
    validIndex = data > thre
    dataTemp = data[validIndex]
    indexTemp = np.zeros(len(dataTemp))
    i = 0
    for j in range(len(data)):
        if validIndex[j]:
            indexTemp[i] = j
            i += 1
    offsetOfIndex = indexTemp[1:] - indexTemp[:-1]
    tmp = indexTemp[1:]
    lowBoundary = np.insert(tmp[offsetOfIndex > 10], 0, indexTemp[0])
    lowBoundary = np.delete(lowBoundary, -1)
    tmp = indexTemp[:-1]
    highBoundary = tmp[offsetOfIndex > 10]
    lengthOfData = highBoundary - lowBoundary
    lowBoundary = lowBoundary[lengthOfData > validMinLength]
    highBoundary = highBoundary[lengthOfData > validMinLength]
    if len(lowBoundary) == 0:
        boundaries = np.array((0, -1))
    else:
        boundaries = np.dstack((lowBoundary, highBoundary))[0, :, :]
    return boundaries.astype(np.int)


def findSyncSequenceNum(data, validRanges, syncDatas, sampleRate):
    '''
    寻找训练序列序号
    '''
    k = sampleRate / 1e6
    numOfBurst = len(validRanges)
    bursts = np.zeros((numOfBurst, int(1000*k)), np.complex128)

    for burstCounter in range(numOfBurst):
        burstData = data[validRanges[burstCounter, 0]:validRanges[burstCounter, 0] + int(1000*k)]
        burstData = burstData / np.abs(burstData)
        bursts[burstCounter] = burstData
    
    corMaxAverage = []
    for i in range(64):
        syncData = syncDatas[i]
        corMax = []
        for j in range(numOfBurst):
            burst = bursts[j]
            cor = np.correlate(burst, syncData)
            corMax.append(np.abs(cor[np.argmax(np.abs(cor))]))
        corMaxAverage.append(np.mean(corMax))
    syncNum = np.argmax(corMaxAverage)
    return syncNum


def getAverageEnergy(data):
    """
    返回数据的平均能量
    """
    return np.sqrt(np.sum(np.abs(data) ** 2) / len(data))


def getEnergyArray(dataBursts):
    """
    返回各段burst的平均能量的列表
    """
    numOfBurst = len(dataBursts)
    averageEnergyArrays = np.zeros(numOfBurst)
    for burstCounter, burstData in enumerate(dataBursts):
        averageEnergyArrays[burstCounter] = getAverageEnergy(burstData)
    return averageEnergyArrays


def calculateAverageEnergy(averageEnergyArrays, percentile):
    """
    返回提取得到的的帧的能量分位值
    
    Parameters
    ----------
    averageEnergyArrays : np.ndarray
        平均能量列表
    percentile : int
        分位值
    
    Returns
    -------
    float
        能量分位值
    """
    percentileEnergy = np.percentile(averageEnergyArrays, percentile) 
    return percentileEnergy


def getCorStartIndex(data, syncData):
    """
    返回一帧数据的训练比特起始位置与相关系数值
    """
    cor = np.correlate(data, syncData)
    # 选取训练序列的部分做相关
    # cor = np.correlate(data, syncData[20:-20])
    corStartIndex = np.argmax(np.abs(cor))
    corMaxValue = np.abs(cor[corStartIndex])
    return corStartIndex, corMaxValue


def getCorStartIndexs(data, validRanges, syncData):
    """
    返回各帧的训练比特起始位置与相关系数值
    """
    numOfBurst = len(validRanges)
    corStartIndexs = np.zeros(numOfBurst, np.int)
    corMaxValues = []
    for burstCounter in range(numOfBurst):
        # 每个时隙先取 700 个点做同步
        burstTemp = data[validRanges[burstCounter, 0]:validRanges[burstCounter, 0] + 700]
        burstTemp = burstTemp / np.abs(burstTemp)
        corStartIndex, corMaxValue = getCorStartIndex(burstTemp, syncData)
        corStartIndexs[burstCounter] = corStartIndex
        corMaxValues.append(corMaxValue)
    return corStartIndexs, corMaxValues


def extractBursts(data, samplePerSymbol, validRanges, syncData, num=None, corPercentile=20, energyPercentile=50):
    """
    提取有效帧
    
    Parameters
    ----------
    data : np.ndarray
        原始采样数据
    samplePerSymbol : int
        每个符号的采样数
    validRanges : np.ndarray
        有效数据范围
    syncData : np.ndarray
        同步序列
    energyPercentile : int, optional
        能量分位值, by default 10
    corPercentile : int, optional
        相关系数分位值, by default 20
    
    Returns
    -------
    np.ndarray
        有效帧
    """
    k = samplePerSymbol / 4
    numOfBurst = len(validRanges)
    # 每个时隙取 645 个点
    dataBursts = np.zeros((numOfBurst, int(645*k)), np.complex64)
    corStartIndexs = np.zeros(numOfBurst, np.int)
    corMaxValues = []
    for burstCounter in range(numOfBurst):
        # 每个时隙先取 700 个点做同步
        burstTemp = data[validRanges[burstCounter, 0]:validRanges[burstCounter, 0] + int(700*k)]
        burstTemp = burstTemp / np.abs(burstTemp)
        corStartIndex, corMaxValue = getCorStartIndex(burstTemp, syncData)
        corStartIndexs[burstCounter] = corStartIndex
        corMaxValues.append(corMaxValue)
    for i, corStartIndex in enumerate(corStartIndexs):
        startBoundary = validRanges[i, 0] + corStartIndex - (3 + 57 + 1) * samplePerSymbol - 20*k  # 20起保护作用
        endBoundary = startBoundary + 156.25 * samplePerSymbol + 20*k
        # print(endBoundary - startBoundary)
        if startBoundary < 0:
            pass
        else:
            dataBursts[i] = data[int(startBoundary):int(endBoundary)]
    corMaxValues = np.array(corMaxValues)
    percentileCor = np.percentile(corMaxValues, corPercentile)
    dataBursts = dataBursts[np.where(corMaxValues > percentileCor)]
    averageEnergyArrays = getEnergyArray(dataBursts)
    if num:
        indexs = np.argsort(averageEnergyArrays)[::-1][:num]
        dataBursts = dataBursts[indexs, :]
    else:
        percentileEnergy = calculateAverageEnergy(averageEnergyArrays, energyPercentile)
        dataBursts = dataBursts[np.where(averageEnergyArrays > percentileEnergy)]
    return dataBursts

##修改，显示每帧的信噪比

import pandas as pd
def SNR_CAL(data_frame, beginer ,ender ):
    data1=np.zeros((data_frame.shape[0], int(beginer)))  
    data2=np.zeros((data_frame.shape[0], int(ender)))  
    
    for i in range(len(data_frame)):
        data1[i,:] = data_frame[i, :int(beginer)]
        data2[i,:] = data_frame[i, -int(ender): ]
    averageEnergyArrays = getEnergyArray(data_frame)
    averageEnergyArrays1 = getEnergyArray(data1)
    averageEnergyArrays2 = getEnergyArray(data2)
  #  print(averageEnergyArrays.shape)
    snr_frame1= averageEnergyArrays/ averageEnergyArrays1
    snr_frame2= averageEnergyArrays/ averageEnergyArrays2
    snr_frame1=10*np.log10(snr_frame1)
    snr_frame2=10*np.log10(snr_frame2)
#    print(snr_frame1,snr_frame2)
    
    
    return (snr_frame1)


def extractBursts1(phone_num, phone_dist,gsm_frequency, syncNum, data, samplePerSymbol, validRanges, syncData, num=None, corPercentile=20, energyPercentile=50):
 
    k = samplePerSymbol / 4
    numOfBurst = len(validRanges)
    framelist=[]
    
    # 每个时隙取 645 个点
    dataBursts = np.zeros((numOfBurst, int(645*k)), np.complex64)
    
    corStartIndexs = np.zeros(numOfBurst, np.int)
    corMaxValues = []
    for burstCounter in range(numOfBurst):
        # 每个时隙先取 700 个点做同步
        burstTemp = data[validRanges[burstCounter, 0]:validRanges[burstCounter, 0] + int(700*k)]
        burstTemp = burstTemp / np.abs(burstTemp)
        corStartIndex, corMaxValue = getCorStartIndex(burstTemp, syncData) ##同步的数值
        corStartIndexs[burstCounter] = corStartIndex  # corStartIndexs是相对位置
        corMaxValues.append(corMaxValue)
    
#     datadic={'startBoundary':0, 'endBoundary':0, "corStartIndex":0 }
#     dataframe = pd.DataFrame(datadic,index=[0])
#     dataframe.to_csv("gasmdata.csv",index=False, sep=',')
    corsindex1 =[]
    startindex=[]
    endindex=[]
    for i, corStartIndex in enumerate(corStartIndexs):
        startBoundary = int(validRanges[i, 0] + corStartIndex - (3 + 57 + 1) * samplePerSymbol - int(20*k) ) # 20起保护作用
        endBoundary = int( startBoundary + 156.25 * samplePerSymbol + int(20*k) )
        
        
        if startBoundary < 0:

            pass
            
        else:
            corsindex=int(corStartIndex+validRanges[i, 0])
#             print("frame1 beginning:",startBoundary, "ending:",endBoundary, "同步位置：",corStartIndex+validRanges[i, 0]) ##打印起始点 
            
            corsindex1.append(corsindex)####  同步点的位置
            startindex.append(startBoundary)
            endindex.append(endBoundary)
###############################################################SNR筛选      
            dataBursts[i] = data[int(startBoundary):int(endBoundary)]
            
    
    
################################################################   
    print(len(corsindex1)) ##所有同步的位置点
 
 #########################################  筛选
    corMaxValues = np.array(corMaxValues)
    corsindex1=np.array(corsindex1)
    startindex=np.array(startindex)
    endindex=np.array(endindex)
    
    print(len(corMaxValues))
    percentileCor = np.percentile(corMaxValues, corPercentile)
    dataBursts = dataBursts[np.where(corMaxValues > percentileCor)]  ##筛选的位置,np.where(corMaxValues > percentileCor)为索引,percentileCor
    
    ######筛选后同步起点，终点，同步点
    corsindex1=corsindex1[np.where(corMaxValues > percentileCor)]
    startindex=startindex[np.where(corMaxValues > percentileCor)]
    endindex=endindex[np.where(corMaxValues > percentileCor)]
    print(corsindex1.shape)
   
 #计算SNR 
   ###################################################################     
    data2=np.zeros((dataBursts.shape[0],int(7*k)))  
   # print(data2.shape)
    for i in range(len(dataBursts)):
        #print(i)
        data2[i,:] = dataBursts[i,:int(7*k)]
 
    averageEnergyArrays = getEnergyArray(dataBursts)
 
  ## SNR估计
    print(dataBursts.shape)
    print(data2.shape)
    averageEnergyArrays2 = getEnergyArray(data2)
  #  print(averageEnergyArrays.shape)
    snr_frame= averageEnergyArrays/ averageEnergyArrays2
    snr_frame=10*np.log10(snr_frame)
    print(snr_frame)
    
    dataBursts = normalizeBurst(dataBursts)
   ################################################################# 
    # dataBursts = dspf.compensatePhase(dataBursts, syncData)
  ###################################################################  写CSV 
#     for i in range(len(corsindex1)):
#         #data1=[dataBursts[i]]
#         datadic1={"1:":gsm_frequency,"2:":startindex[i], "3":corsindex1[i], "4":endindex[i], "5":syncNum,                          "6":snr_frame[i],"7": averageEnergyArrays[i]}
#         dataframe1 = pd.DataFrame(datadic1,index=[0])  ####
#         dataframe1.to_csv("parameters"+phone_num+".csv",index=False, mode='a+',header=False)
#         data1=[dataBursts[i]]
# #        data1 = [np.array(list(zip(np.real(dataBursts[i]),np.imag(dataBursts[i])))).flatten()]   ##I/Q分离
#         data1 = pd.DataFrame(data1,index=[0])
#         data1.to_csv('data'+phone_num+'.csv',index=False ,mode='a+',header=False)
#     f1 = pd.read_csv("parameters"+phone_num+".csv")
#     f2 = pd.read_csv('data'+phone_num+'.csv')
#     file = [f1,f2]
#     train = pd.concat(file,axis=1)
#     import datetime
#     filename = datetime.datetime.now().strftime(phone_num+'_'+phone_dist+'_'+'%y%m%d-%H%M%S.csv')
#     train.to_csv(filename, index=0, sep=',')
#     os.remove("parameters"+phone_num+".csv")
#     os.remove('data'+phone_num+'.csv')
    #"gsmdata"+str(gsm_frequency)+".csv"
         
#     if num:  ##
#         indexs = np.argsort(averageEnergyArrays)[::-1][:num]
#         dataBursts = dataBursts[indexs, :]
#     else:  ##只选前面一半的帧
#         percentileEnergy = calculateAverageEnergy(averageEnergyArrays, energyPercentile) ##高于百分比的帧能量
#         dataBursts = dataBursts[np.where(averageEnergyArrays > percentileEnergy)] ##选出来
    return dataBursts

def onlyExtractBursts(data, validRanges, num=None):
    """"
    只提取帧，不做其他操作，当num为整数时，提取能量较高的num帧
    取1050个点，避免有些帧取不全
    """
    numOfBurst = len(validRanges)
    # 每个时隙取 645 个点
    dataBursts = np.zeros((numOfBurst, 1050), np.complex64)

    for burstCounter in range(numOfBurst):
        burstTemp = data[validRanges[burstCounter, 0] - 50:validRanges[burstCounter, 0] + 1000]
        if len(burstTemp) != 1050:
            continue
        dataBursts[burstCounter] = burstTemp
    averageEnergyArrays = getEnergyArray(dataBursts)
    if num:
        indexs = np.argsort(averageEnergyArrays)[::-1][:num]
        dataBursts = dataBursts[indexs, :]
    return dataBursts


def calculatePhaseOffset(trainingSequence, syncData):
    """
    通过与本地训练序列syncData共轭相乘计算相偏
    """
    phaseOffset = np.dot(trainingSequence, np.conj(syncData))
    phaseOffset = phaseOffset / len(syncData)
    # 避免对能量的干扰，对相偏作归一化
    phaseOffset = phaseOffset / np.abs(phaseOffset)
    return phaseOffset


def compensatePhase(dataBursts, syncData):
    """
    对每个帧进行相位补偿
    
    Parameters
    ----------
    dataBursts : np.ndarray
        提取得到的帧
    syncData : np.ndarray
        本地训练序列
    
    Returns
    -------
    np.ndarray
        进行相位补偿后的帧
    """
    lenOfTrainingSeq = len(syncData)
    for i, burst in enumerate(dataBursts):
        corstartIndex, _ = getCorStartIndex(burst, syncData)
        trainingSequence = dataBursts[i, int(corstartIndex):int(corstartIndex) + lenOfTrainingSeq]
        phaseOffset = calculatePhaseOffset(trainingSequence, syncData)
        dataBursts[i] = dataBursts[i] * np.conj(phaseOffset)
    return dataBursts


def getDataIQImblanced(data, alphaI=1, alphaQ=1):
    """
    进行IQ两路数据不均衡
    
    Parameters
    ----------
    data : np.ndarray
        输入数据
    alphaI : float  
        I 路系数
    alphaQ : float  
        Q 路系数

    Returns
    -------
    np.ndarray
        进行时延后的数据
    """
    result = alphaI * np.real(data) + alphaQ * np.imag(data) * 1j
    return result


def getDataIQOffset(data, offsetInterval):
    """
    进行IQ两路数据的时延
    
    Parameters
    ----------
    data : np.ndarray
        输入数据
    offsetInterval : int
        时延长度
    
    Returns
    -------
    np.ndarray
        进行时延后的数据
    """
    result = np.roll(np.real(data), -offsetInterval) + np.imag(data) * 1j
    result = result[:data.size - offsetInterval]
    return result


def getDifference(data, offsetInterval):
    """
    计算数据的差分结果
    result(t) = data(t) * conj(data(t+n))

    Parameters
    ----------
    data : np.ndarray
        要进行差分的flatten后的数据
    offsetInterval : int
        差分间隔

    Returns
    -------
    np.ndarray
        差分结果
    """
    if (data.size - offsetInterval) > 0:
        return data * np.conj(np.roll(data, offsetInterval))
    else:
        print('差分间隔大于数据长度！')
        return data


def getPlotTable(plotTableSize, plotTableMax, data, offsetInterval=0, minPointsNum=None):
    """获取DCTF绘图表格"""
    plotTable = np.zeros((2 * plotTableSize + 1, 2 * plotTableSize + 1), \
        np.float32)
    plotScale = float(plotTableMax) / plotTableSize
    processDataLength = data.size - 2 * offsetInterval
    if processDataLength > 0:
        for i in range(offsetInterval, processDataLength - offsetInterval):
            imageIndex = int(np.clip(np.imag(data[i]) / plotScale, \
                -plotTableSize, plotTableSize))
            realIndex = int(np.clip(np.real(data[i]) / plotScale, \
                -plotTableSize, plotTableSize))
            plotTable[plotTableSize + imageIndex, \
                plotTableSize + realIndex] += 1
    plotTable[plotTableSize, plotTableSize] = 0
    if minPointsNum:
        plotTable = np.where(plotTable < minPointsNum, 0, plotTable)
    plotTable = np.sqrt(plotTable)
    return plotTable


def grayColorR(gray):
    def core(c):
        r = 0
        if 128 <= c <= 170:
            r = 255.0 / 42 * (c - 128)
        elif c > 170:
            r = 255
        return r
    core = np.frompyfunc(core, 1, 1)
    return core(gray).astype(np.int)


def grayColorG(gray):
    def core(c):
        g = 0
        if c < 84:
            g = 255.0 / 84 * c
        elif c <= 170:
            g = 255
        elif c <= 255:
            g = 255.0 / 85 * (255 - c)
        return g
    core = np.frompyfunc(core, 1, 1)
    return core(gray).astype(np.int)


def grayColorB(gray):
    def core(c):
        b = 0
        if c < 84:
            b = 255
        elif c <= 128:
            b = 255.0 / 44 * (128 - c)
        return b
    core = np.frompyfunc(core, 1, 1)
    return core(gray).astype(np.int)


def grayColor(plotTable):
    """
    根据绘图表格的值，计算相应的RGB值
    
    Parameters
    ----------
    dataTable : np.ndarray
        绘图表格
    
    Returns
    -------
    tuple(r, g, b)
        绘图表格各点RGB值
    """
    # 将绘图表格中各点的值缩放到0～255之间
    plotTable = plotTable / np.ptp(plotTable) * 255.0
    r = grayColorR(plotTable)
    g = grayColorG(plotTable)
    b = grayColorB(plotTable)
    return r, g, b


def imgArray(r, g, b):
    img = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def vfdt(data, k=0.1, W=128, S=1, K=5):
    """
    使用方差分形提取信号暂态起始点位置
    
    Parameters
    ----------
    data : np.ndarray
        信号能量
    k : float, optional
        阈值, by default 0.1
    W : int, optional
        窗口大小, by default 128
    S : int, optional
        窗口滑动偏移量, by default 1
    K : int, optional
        时间增量的书目, by default 5
    
    Returns
    -------
    int
        起始点位置索引
    """
    N = data.size
    DList = []
    meanNoise = []
    varNoise = []
    # 进行窗口滑动
    for n in range(N - W):
        tmp = data[n:n + W]
        varBList = []
        # 依次计算时间增量序列的VarB
        for i in range(K):
            deltaT = i + 1
            deltaBList = []
            # 一个窗口中所有的时间增量序列对
            for j in range(W - deltaT):
                t1 = j
                t2 = t1 + deltaT
                deltaB = tmp[t2] - tmp[t1]
                deltaBList.append(deltaB)
            varBList.append(np.cov(deltaBList))
        meanNoise.append(np.mean(tmp[:int(W // 4)]))
        varNoise.append(np.var(tmp[:int(W // 4)]))
        xk = np.log10(np.arange(1, K + 1))
        yk = np.log10(varBList)
        H2 = (K * np.sum(xk * yk) - np.sum(xk) * np.sum(yk)) / (K * np.sum(np.square(xk)) - np.square(np.sum(xk)))
        D = 2 - H2
        DList.append(D)
    meanNoise = np.array(meanNoise)
    varNoise = np.array(varNoise)
    DList = np.array(DList)
    result = np.abs(DList - meanNoise) - (k * meanNoise + varNoise)
    # 阈值判断条件
    # threCondition = (np.abs(tmpD - meanD) - varD) / meanD
    return np.where(result < 0)[0][0]


def posterior1(data):
    """
    贝叶斯步进器检测暂态起始点
    
    Parameters
    ----------
    data : np.ndarray
        信号能量，注意序列长度不能过长，否则计算数据会溢出，每次取200个点计算后验概率，以后再解决这个问题
    
    Returns
    -------
    int
        暂态信号起始点
    """
    def S1(d, m):
        a = np.sum(np.square(d[:m]))
        b = (1 / m) * (np.sum(np.square(d[:m])))
        c = np.power((a - b), (1 - m) / 2)
        return c

    def S2(d, m):
        n = d.size
        a = np.sum(np.square(d[m:]))
        b = (1 / (n - m)) * np.sum(np.square(d[m:]))
        c = np.power((a - b), (1 + m - n) / 2)
        return c

    n = data.size
    posteriorList = []
    for m in range(2, n - 1):
        a = 1 / np.sqrt(m * (n - m))
        b = gamma((m - 1) / 2)
        c = gamma((n - m - 1) / 2)
        s1 = S1(data, m)
        s2 = S2(data, m)
        posteriorValue = a * b * c * s1 * s2
        posteriorList.append(posteriorValue)
    posteriorList = np.array(posteriorList)
    return np.argmax(posteriorList)

def posterior2(data):
    """
    贝叶斯步进器检测暂态起始点
    
    Parameters
    ----------
    data : np.ndarray
        信号能量，注意序列长度不能过长，否则计算数据会溢出，每次取200个点计算后验概率，以后再解决这个问题

    Returns
    -------
    int
        暂态信号起始点
    """
    def fun(d, n, m):
        a = 1 / np.sqrt(m * (n - m))
        b = np.sum(np.square(d)) - (1 / m) * np.square(np.sum(d[:m])) - (1 / (n - m)) * np.square(np.sum(d[m:]))
        tmp = a * np.power(b, (2 - n) / 2)
        return tmp

    n = data.size
    posteriorList = []
    for m in range(1, n):
        posteriorList.append((fun(data, n, m)))
    posteriorList = np.array(posteriorList)
    return np.argmax(posteriorList)


def addNoise(data, snr=7):
    """
    对data加入噪声
    """

    snr = 10 ** (snr / 10.0)
    xpower = np.sum(np.abs((np.square(data)))) / len(data)
    npower = xpower / snr
    noise = np.random.randn(len(data)) * np.sqrt(npower)

    result = data + noise
    result = result + noise * 1j

    return result

