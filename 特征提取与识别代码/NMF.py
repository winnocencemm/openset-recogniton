import os
import numpy as np
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 读取文件夹中的所有图像，并转换为矩阵
image_folder = "D:/研究生/王敏/muticlass_openset_recognition/train/STATE"
images = []
for filename in os.listdir(image_folder):
    img = plt.imread(os.path.join(image_folder, filename))
    images.append(img.reshape(-1)) # 将图像矩阵展平为一维向量
X = np.array(images)

# 将像素值归一化到0到1的范围内
X = X.astype(np.float64) / 255

# 使用非负矩阵分解进行聚类
k = 4  # 簇数目
model = NMF(n_components=k, init='nndsvd', random_state=0)
W = model.fit_transform(X)
labels = np.argmax(W, axis=1)
print(labels)
# 使用t-sne进行可视化
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne = tsne.fit_transform(X)

# 绘制可视化结果
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='rainbow')
plt.show()
