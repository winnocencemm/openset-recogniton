import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from deepclustering import DeepCluster, make_deepcluster_model
from PIL import Image

# 定义数据集类
class DCTFDataSet(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.images = []
        for filename in os.listdir(root_dir):
            img = Image.open(os.path.join(root_dir, filename))
            img = self.transform(img)
            self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

# 加载数据
dataset = DCTFDataSet("D:/研究生/王敏/muticlass_openset_recognition/train/STATE")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义DeepCluster模型
model = make_deepcluster_model(input_shape=(1, 256, 256), num_classes=20)

# 训练DeepCluster模型
deepcluster = DeepCluster(model, dataloader)
deepcluster.train()

# 提取特征表示
features = []
with torch.no_grad():
    for data in dataloader:
        features.append(deepcluster.model(data).detach().numpy())
features = np.concatenate(features)

# 使用K-Means聚类
kmeans = KMeans(n_clusters=20)
labels = kmeans.fit_predict(features)

# 使用t-SNE可视化聚类结果
tsne = TSNE(n_components=2, perplexity=30.0)
embedded = tsne.fit_transform(features)
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels)
plt.show()
