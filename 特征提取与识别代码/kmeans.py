import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 定义KMeans聚类的簇数
num_clusters = 6

# 定义t-SNE的参数
perplexity = 30
learning_rate = 200

# 定义数据集和标签
X = []
y = []

# 设置文件夹路径
folder_path = "D:/研究生/王敏/muticlass_openset_recognition/train/STATE"
# 设置射频指纹DCTF图像路径列表
image_paths = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".png"):  # 这里假设图像格式为png
            image_paths.append(os.path.join(root, file))
# 读取图像并转换为numpy数组
X = []
for image_path in image_paths:
    img = Image.open(image_path).convert("L")  # 转为灰度图像
    img_array = np.asarray(img).flatten()  # 展平为一维向量
    X.append(img_array)
X = np.array(X)


# 进行KMeans聚类
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)

# 获取聚类结果
labels = kmeans.labels_

# 使用t-SNE进行降维和可视化
# tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate)
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# 绘制散点图
plt.figure(figsize=(10, 10))
colors = ["b", "g", "r", "c", "m", "y", "k"]
for i in range(num_clusters):
    plt.scatter(X_tsne[labels == i, 0], X_tsne[labels == i, 1], c=colors[i % len(colors)], label="Cluster {}".format(i+1))
plt.legend()
plt.show()
