import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix

##设置使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
])

class FlameSet(data.Dataset):
    def __init__(self,filenamepath1,filenamepath2,filenamepath3):
        # 所有图片的绝对路径
        imgslist1=os.listdir(filenamepath1)
        self.imgslist1=imgslist1
        self.filenamepath1=filenamepath1
        self.imgs1=[os.path.join(filenamepath1,k) for k in imgslist1]

        imgslist2 = os.listdir(filenamepath2)
        self.imgslist2 = imgslist2
        self.filenamepath2 = filenamepath2
        self.imgs2 = [os.path.join(filenamepath2, k) for k in imgslist2]
        
        imgslist3 = os.listdir(filenamepath3)
        self.imgslist3 = imgslist3
        self.filenamepath3 = filenamepath3
        self.imgs3 = [os.path.join(filenamepath3, k) for k in imgslist3]

        self.transforms=transform
    def __getitem__(self, index):
        img_path1 = self.imgs1[index]
        pil_img1 = Image.open(img_path1)
        pil_img1 = pil_img1.convert("RGB")
        
        label1 = int(self.imgslist1[index][3:4])-3
        
        if self.transforms:
            data1 = self.transforms(pil_img1)
        else:
            pil_img1 = np.asarray(pil_img1)
            data1 = torch.from_numpy(pil_img1)

        
        img_path2 = self.imgs2[index]
        pil_img2 = Image.open(img_path2)
        pil_img2 = pil_img2.convert("RGB")

        label2 = int(self.imgslist2[index][3:4]) - 3
        if self.transforms:
            data2 = self.transforms(pil_img2)
        else:
            pil_img2 = np.asarray(pil_img2)
            data2 = torch.from_numpy(pil_img2)
        
        
        img_path3 = self.imgs3[index]
        pil_img3 = Image.open(img_path3)
        pil_img3 = pil_img3.convert("RGB")
        

        label3 = int(self.imgslist3[index][3:4]) - 3
        if self.transforms:
            data3 = self.transforms(pil_img3)
        else:
            pil_img3 = np.asarray(pil_img3)
            data3 = torch.from_numpy(pil_img3)

        data = torch.cat((data1,data2,data3),dim=0)
        if label1==label2 and label2==label3:
            return data, label1

    def __len__(self):
        return len(self.imgs1)

'''
神经网络结构参数
'''
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5),
#             torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
#             torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
#             torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
#             torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            # torch.nn.MaxPool2d(kernel_size=3, stride=1)
        )
        
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
#             torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
#             torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            # torch.nn.MaxPool2d(kernel_size=3, stride=1)
        )

        self.classifier = torch.nn.Sequential(
            # torch.nn.Dropout2d(p=0.2),
            # nn.Linear(in_features=16 * 5 * 5, out_features=120),
            # nn.Linear(in_features=16 * 29 * 29, out_features=120),
            # torch.nn.Linear(in_features=16 * 29 * 29  , out_features=60),
            torch.nn.Linear(in_features=16 * 29 * 29 *3 , out_features=120),
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.Linear(in_features=84, out_features=6)
        )
    def forward(self, x):
        b, c ,e= x.split([3,3,3], dim=1)
#         d, e = c.split(3, dim=1)
        # print(b.shape)
        # print(c.shape)
        b = self.conv1(b)
        c = self.conv2(c)
#         d = self.conv2(d)
        e = self.conv3(e)
        #print(b.shape)
        #print(c.shape)
        b = b.view(b.size(0),-1)
        c = c.view(c.size(0),-1)
#         d = d.view(d.size(0),-1)
        e = e.view(e.size(0),-1)
        #print(b.shape)
        x = torch.cat((b,c,e),dim=1)

#         x=self.conv1(x)
#         x=x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
if __name__ == '__main__':
    '''
    导入训练集
    '''
    # traindataSet=FlameSet('/DCTF/TRAIN/coarsesync_STATE',
    #                       '/DCTF/TRAIN/coarsesync_START',
    #                       '/DCTF/TRAIN/coarsesync_END')
    # traindataSet = FlameSet('F:/殷鹏程/LTE数据/时间1/Uploadfile/Indoor/30dB/TRAIN/autocollectaftercoursesync_DCTF_STATE',
    #                         'F:/殷鹏程/LTE数据/时间1/Uploadfile/Indoor/30dB/TRAIN/autocollectaftercoursesync_DCTF_START',
    #                         'F:/殷鹏程/LTE数据/时间1/Uploadfile/Indoor/30dB/TRAIN/autocollectaftercoursesync_DCTF_END')

    traindataSet = FlameSet('F:/王敏/认证/Train/LTE4-8/autocollectaftercoursesync_DCTF_STATE80%',
                            'F:/王敏/认证/Train/LTE4-8/autocollectaftercoursesync_DCTF_START80%',
                            'F:/王敏/认证/Train/LTE4-8/autocollectaftercoursesync_DCTF_END80%')
    traindata_loader=data.DataLoader(traindataSet,batch_size=64,shuffle=True)

    torch.cuda.empty_cache()
    model = CNNnet()

    '''
    载入GPU，不使用GPU则注释
    '''
    # model.cuda()
    # print(model)

    '''
    损失函数和优化算法
    '''
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    #opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    '''
    训练阶段
    '''
    accuracy_last=0
    loss_count = []
    accuracy_list=[]
    loss_init=0.001
    for _ in range(1):
        ##epoch
        for epoch in range(50):
            for i, (x, y) in enumerate(traindata_loader):

                '''
                不使用GPU则注释
                '''
                # x=x.cuda()
                # y=y.cuda()


                batch_x = Variable(x)  # torch.Size([128, 1, 28, 28])
                batch_y = Variable(y)  # torch.Size([128])
                # 获取最后输出
                out = model(batch_x)  # torch.Size([128,10])
                # 获取损失
                loss = loss_func(out, batch_y)
                # 使用优化器优化损失
                opt.zero_grad()  # 清空上一步残余更新参数值
                loss.backward()  # 误差反向传播，计算参数更新值
                opt.step()  # 将参数更新值施加到net的parmeters上
                if i % 50000 == 0:
                    #loss_count.append(loss)
                    print('{}:\t'.format(i), loss.item())
                    if loss.item()<loss_init:
                        '''
                        保存模型
                        '''
                        # torch.save(model,'bestmodel')
                        torch.save(model, 'bestmodel_feature')
                        loss_init=loss.item()



    
    