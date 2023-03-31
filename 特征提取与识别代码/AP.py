import os
import numpy as np
from PIL import Image

# 设置文件夹路径
folder_path = "D:/研究生/王敏/muticlass_openset_recognition/train/STATE"
# 设置射频指纹DCTF图像路径列表
image_paths = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".png"):  # 这里假设图像格式为png
            image_paths.append(os.path.join(root, file))

# 读取图像并转换为numpy数组
images = []
for image_path in image_paths:
    img = Image.open(image_path).convert("RGB")  # 转为灰度图像
    img_array = np.asarray(img).flatten()  # 展平为一维向量
    images.append(img_array)
images = np.array(images)

from sklearn.metrics import pairwise_distances

# 计算相似矩阵
similarities = pairwise_distances(images.reshape(len(images), -1), metric='cosine')

from sklearn.cluster import SpectralClustering

# 定义聚类器
n_clusters = 6
clusterer = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')

# 对相似矩阵进行聚类
labels = clusterer.fit_predict(similarities)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 使用t-SNE算法将高维特征向量降到二维
tsne = TSNE(n_components=2, random_state=0)
features = tsne.fit_transform(images.reshape(len(images), -1))

# 绘制聚类结果的散点图
plt.scatter(features[:,0], features[:,1], c=labels)
plt.show()



