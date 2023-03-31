import logging
import scipy.io as scio
import dspFunctions as dspf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
import os

def F_Frequency_Offset_Estimation_Fine(Data_In, Syn_Start, Syn_Length, Syn_Interval):
    Mean_Thres = 0.2
    Syn_Results_Raw= Data_In[Syn_Start+1:Syn_Start+Syn_Length+1] * np.conj(Data_In[Syn_Start+Syn_Interval+1:Syn_Start+Syn_Length+Syn_Interval+1])
    Mean_Power = np.mean(abs(Syn_Results_Raw))
    Syn_Results_Raw_Rotated = Syn_Results_Raw * np.exp(-1j*math.pi/2)
    Syn_Results_Angle = np.angle(Syn_Results_Raw)
    Syn_Results_Rotated_Angle = np.angle(Syn_Results_Raw_Rotated)
    Temp_1 = np.mean(Syn_Results_Angle)
    Syn_Results_Angle_double=(Syn_Results_Angle - Temp_1)**2
    Var_Angle=sum(Syn_Results_Angle_double)
    Var_Angle = Var_Angle / Syn_Length
    Temp_1 = np.mean(Syn_Results_Rotated_Angle)
    Syn_Results_Rotated_Angle_double = (Syn_Results_Rotated_Angle - Temp_1) ** 2
    Var_Angle_Ratated = sum(Syn_Results_Rotated_Angle_double)
    Var_Angle_Ratated = Var_Angle_Ratated / Syn_Length
    if (Var_Angle > Var_Angle_Ratated):
        Syn_Results_Raw = abs(Syn_Results_Raw)
        Temp_1 = 0
        Temp_2 = 0
        for n in range(Syn_Length):
            if Syn_Results_Raw[n]>Mean_Power*Mean_Thres:
                Temp_2 = Temp_2 + Syn_Results_Rotated_Angle[n] + math.pi / 2
                Temp_1 = Temp_1 + 1
        Temp_3 = Temp_2 / Temp_1
        if Temp_3>math.pi+0.1:
            Temp_3=Temp_3-2*math.pi
        Est_Freq_Offset = Temp_3 / Syn_Interval
    else:
        Syn_Results_Raw = abs(Syn_Results_Raw)
        Temp_1 = 0
        Temp_2 = 0
        for n in range(Syn_Length):
            if Syn_Results_Raw[n] > Mean_Power * Mean_Thres:
                Temp_2 = Temp_2 + Syn_Results_Angle[n]
                Temp_1 = Temp_1 + 1
        Temp_3 = Temp_2 / Temp_1
        Est_Freq_Offset = Temp_3 / Syn_Interval

    return Est_Freq_Offset
def F_Frequency_Offset_Compensationn(Get_OFDM_Data, Est_Freq_Compensate):
    Data_Length = len(Get_OFDM_Data)
    explist=[np.exp(-1j*Est_Freq_Compensate*(n+1)) for n in range(Data_Length)]
    Get_OFDM_Data_Compensated=Get_OFDM_Data*explist
    return Get_OFDM_Data_Compensated
def Myself_Get_LTE_ZC_Syn_Index(Received_Data, ZC_Ref_Sequence,Syn_Start):
    deltalist = []
    for m in range(64):
        qiandao_examplefft = np.fft.fftshift(np.fft.fft(Received_Data[Syn_Start + 1650:Syn_Start + 1650 + 12800]))
        syncData_forcorfft = np.fft.fftshift(np.fft.fft(ZC_Ref_Sequence[m, 1650:14450]))
        syncData_forcorfft = np.conj(syncData_forcorfft)
        fftcor = qiandao_examplefft * syncData_forcorfft
        fftcor = np.fft.fftshift(np.fft.ifft(fftcor))
        delta = np.abs(np.argmax(np.abs(fftcor)) - 6400)
        deltalist.append(delta)
    preamble_index = np.argmin(deltalist)

    deltalist = []
    for n in range(-1000, 1000):
        qiandao_examplefft = np.fft.fftshift(np.fft.fft(Received_Data[Syn_Start + 1650 + n:Syn_Start + 1650 + 12800 + n]))
        syncData_forcorfft = np.fft.fftshift(np.fft.fft(ZC_Ref_Sequence[preamble_index, 1650:14450]))
        syncData_forcorfft = np.conj(syncData_forcorfft)
        fftcor = qiandao_examplefft * syncData_forcorfft
        fftcor = np.fft.fftshift(np.fft.ifft(fftcor))
        delta = np.abs(np.argmax(np.abs(fftcor)) - 6400)
        deltalist.append(delta)
    syn_index = np.argmin(deltalist) - 1000 + Syn_Start

    return preamble_index,syn_index


sampleRate=16e6
Syn_Length=1651
Syn_Interval=12800


##导入本地标准前导序列
trainingSequenceDataFile = '15.36e6default_ZC_64_zeroCorrelationZoneConfig=1.mat'
syncData = scio.loadmat(trainingSequenceDataFile)
syncDatas = syncData['matlabPreamble']
syncDatas_mean=np.zeros((64,16000),dtype=complex)
for i in range(64):
    syncDatas_mean[i,:]=signal.resample_poly(syncDatas[i,:],25,24)
    syncDatas_mean[i,:]=syncDatas_mean[i,:]/np.mean(np.abs(syncDatas_mean[i,:]))

##导入捕获到的原始信号
filepath='rawsignal/autocollect16e6_01_30/'
filenamelist=os.listdir(filepath)

b,a=signal.butter(8,1e6/(sampleRate/2))
indexlist=[]
preamble_index_list=[]
sync_index_list=[]
qiandao_list=[]
Est_Freq_Offset_list=[]
Temp_2_list=[]
qiandao_list_without_FOC=[]
qiandao_list_without_POC=[]
location_list=[]
preamble_list=[]
device_list=[]
for i in range(1):
    qiandao = np.load('rawsignal/autocollect16e6_01_30/' + filenamelist[i])
    # qiandao = signal.resample_poly(qiandao,4,5)
    qiandao = signal.lfilter(b, a, qiandao)

    if len(qiandao)>1536:
        ##判断是否是前导信号
        qiandaolow = signal.resample_poly(qiandao,3,25)
        percentlist=[]
        for k in range(len(qiandaolow) - 1536):
            datarange = qiandaolow[k:k + 1536]
            datarangefft = np.fft.fftshift(np.fft.fft(datarange))
            percent = np.sum(np.abs(datarangefft[349:1188])) / np.sum(np.abs(datarangefft))
            percentlist.append(percent)
        if np.max(percentlist)>0.95:
            print(str(i)+"true")##是输出true
            ##根据循环前缀同步前导的位置
            corlist = []
            dataRanges = qiandaolow/np.mean(np.abs(qiandaolow))
            for j in range(len(dataRanges) - 1536 - 198):
                datarange1 = dataRanges[j:j + 198]
                datarange2 = dataRanges[j + 1536:j + 1536 + 198]
                cor = np.correlate(datarange1, datarange2)
                corlist.append(np.abs(cor))
            corargmax = np.argmax(corlist)
            if (int(corargmax*25/3)-2000)>0:
                # dataAftersync = qiandao[int(corargmax * 25 / 3) - 2000:int(((corargmax + 1920) * 25 / 3) + 2000)]
                dataAftersync=qiandao[int(corargmax*25/3)-2000:int(((corargmax+1920)*25/3)+2000)]/np.mean(np.abs(qiandao[int(corargmax*25/3)-2000:int(((corargmax+1920)*25/3)+2000)]))
                # plt.subplot(9,10,i+1)
                # plt.plot(dataAftersync)
                # qiandaolist.append(dataAftersync)

                '''
                同步序号&&频偏补偿&&同步位置
                '''
                dataAftersync=dataAftersync/np.mean(np.abs(dataAftersync))
                preamble_index1, syn_index1 = Myself_Get_LTE_ZC_Syn_Index(dataAftersync, syncDatas_mean, 2000)

                Data_without_Compensate= dataAftersync[syn_index1 - 1500:syn_index1 - 1500 + 18000]
                qiandao_list_without_FOC.append(Data_without_Compensate)

                '''
                保存细同步后的信号
                '''
                # if len(Data_without_Compensate)!=0:
                #     if np.max(np.abs(Data_without_Compensate[1400:17000]))<3 :
                #         preamble_list.append(Data_without_Compensate)
                #         location_list.append(int(filenamelist[i][-5]))
                #         device_list.append(int(filenamelist[i][-15]))
                #         np.save("lte_rach/16e6/2021_01_30/autocollectaftercoarsesync/" +filenamelist[i][0:-4] +"_p=" + str(preamble_index1),Data_without_Compensate)


                Est_Freq_Offset = F_Frequency_Offset_Estimation_Fine(dataAftersync, syn_index1, Syn_Length, Syn_Interval)
                Data_Compensate = F_Frequency_Offset_Compensationn(dataAftersync, -Est_Freq_Offset)
                qiandao_list_without_POC.append(Data_Compensate[syn_index1 - 1500:syn_index1 - 1500 + 18000])
                Est_Freq_Offset_list.append(Est_Freq_Offset)
                preamble_index2, syn_index2 = Myself_Get_LTE_ZC_Syn_Index(Data_Compensate, syncDatas_mean, 2000)
                preamble_index_list.append(preamble_index2)
                sync_index_list.append(syn_index2)


                Temp_1 = 0
                for n in range(Syn_Length + Syn_Interval - 1):
                    Temp_1 = Temp_1 + Data_Compensate[syn_index2 + n] * np.conj(syncDatas_mean[preamble_index2, n])
                Temp_2 = np.angle(Temp_1)
                Temp_2_list.append(Temp_2)
                explist = [np.exp(-1j * Temp_2)] * len(Data_Compensate)
                Data_Compensate_final = Data_Compensate * explist

                Data_Compensate_final_final = Data_Compensate_final[syn_index2 - 1500:syn_index2 - 1500 + 18000]
                qiandao_list.append(Data_Compensate_final_final)

                '''
                保存细同步+频偏相偏补偿后的信号
                '''
                #if len(Data_Compensate_final_final) != 0:
                    #if np.max(np.abs(Data_Compensate_final_final[1400:17000])) < 3 :
                        #np.save("lte_rach/16e6/2021_01_30/autocollectafterfinesync/"+filenamelist[i][0:-4] +"_p=" + str(preamble_index2),Data_Compensate_final_final)
