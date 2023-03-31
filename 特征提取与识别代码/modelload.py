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
from sklearn import metrics


'''
选择gpu
'''
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
神经网络结构参数，和训练使用的结构保持一致
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
        # print(b.shape)
        # print(c.shape)
        b = b.view(b.size(0),-1)
        c = c.view(c.size(0),-1)
#         d = d.view(d.size(0),-1)
        e = e.view(e.size(0),-1)
        x = torch.cat((b,c,e),dim=1)

#         x=self.conv1(x)
#         x=x.view(x.size(0),-1)
        
    
    ##输出中间层输出使用的代码
#         for i in range(len(self.classifier)):
#             x = self.classifier[i](x)
#             if i==0:
#                 first_out=x
#                 np.save('/USRP-firstout',first_out.detach().numpy())
#             if i==1:
#                 second_out=x
#                 np.save('/USRP-secondout',second_out.detach().numpy())
                
    
        x = self.classifier(x)
        return x

'''
加载模型，可选加载位置CPU or GPU
'''
model = torch.load('bestmodel_1',map_location='cpu')
# testdataSet=FlameSet('/DCTF/TEST/coarsesync_STATE',
#                           '/DCTF/TEST/coarsesync_START',
#                           '/DCTF/TEST/coarsesync_END')

testdataSet=FlameSet('F:/殷鹏程/LTE数据/时间1/Uploadfile/Indoor/30dB/TEST/autocollectaftercoursesync_DCTF_STATE',
                     'F:/殷鹏程/LTE数据/时间1/Uploadfile/Indoor/30dB/TEST/autocollectaftercoursesync_DCTF_START',
                     'F:/殷鹏程/LTE数据/时间1/Uploadfile/Indoor/30dB/TEST/autocollectaftercoursesync_DCTF_END')

testdata_loader=data.DataLoader(testdataSet,batch_size=len(testdataSet),shuffle=False)
for a, b in testdata_loader:
                    '''
                    加载在CPU则注释
                    '''
                    #a = a.cuda()
                    #b = b.cuda()


                    test_x = Variable(a)
                    test_y = Variable(b)
                    test_out = model(test_x)
                    
                    

##加载在cpu时
cm = confusion_matrix(test_y.numpy(),  torch.max(test_out, 1)[1].numpy())
print(cm)
acc=metrics.accuracy_score(test_y.numpy(),torch.max(test_out, 1)[1].numpy())

# Precision=metrics.precision_score(test_y.numpy(),torch.max(test_out, 1)[1].numpy())
# Recall=metrics.recall_score(test_y.numpy(),torch.max(test_out, 1)[1].numpy())
# F1_score=metrics.f1_score(test_y.numpy(),torch.max(test_out, 1)[1].numpy())
print("acc=",acc)
# print("Precision",Precision)
# print("Recall",Recall)
# print("F1_score",F1_score)
