import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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
X = []
for image_path in image_paths:
    img = Image.open(image_path).convert("L")  # 转为灰度图像
    img_array = np.asarray(img).flatten()  # 展平为一维向量
    X.append(img_array)
X = np.array(X)

# 使用DBSCAN聚类
dbscan = DBSCAN(eps=10, min_samples=500)
labels = dbscan.fit_predict(X)
print(labels)
# 使用t-SNE将高维特征向量降到2维
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# 绘制聚类结果
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='rainbow')
plt.title('DBSCAN Clustering')
plt.show()
