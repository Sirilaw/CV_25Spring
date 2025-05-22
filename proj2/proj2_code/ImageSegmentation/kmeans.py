# coding: utf-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class KMeansSegmenter:
    def __init__(self, n_clusters=3, max_iter=40, tol=1e-4):
        """
        初始化K-Means分割器
        
        参数:
        - n_clusters: 聚类数量(K)
        - max_iter: 最大迭代次数
        - tol: 中心点移动容差
        """
        self.K = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
    
    def _initialize_centroids(self, X):
        """随机初始化中心点"""
        np.random.seed(42)  # 固定随机种子确保可重复性
        random_idx = np.random.permutation(X.shape[0])[:self.K]
        self.centroids = X[random_idx]
    
    def _euclidean_distance(self, X):
        """计算欧氏距离"""
        return np.sqrt(((X[:, np.newaxis] - self.centroids)**2).sum(axis=2))
    
    def _assign_clusters(self, X):
        """分配每个点到最近的中心点"""
        distances = self._euclidean_distance(X)
        self.labels = np.argmin(distances, axis=1)
    
    def _update_centroids(self, X):
        """更新中心点为簇内点的均值"""
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.K):
            new_centroids[k] = X[self.labels == k].mean(axis=0)
        return new_centroids
    
    def _has_converged(self, new_centroids):
        """检查是否收敛"""
        return np.linalg.norm(new_centroids - self.centroids) < self.tol
    
    def fit(self, X):
        """执行K-Means聚类"""
        self._initialize_centroids(X)
        
        for i in range(self.max_iter):
            self._assign_clusters(X)
            new_centroids = self._update_centroids(X)
            
            if self._has_converged(new_centroids):
                print(f"K={self.K} 聚类收敛于 {i+1} 次迭代")
                break
                
            self.centroids = new_centroids
    
    def predict(self, X):
        """预测每个点的簇标签"""
        distances = self._euclidean_distance(X)
        return np.argmin(distances, axis=1)

def kmeans_segmentation(input_path, k_values=[2, 4, 8, 16, 32], output_dir='output', t=1e-4):
    """
    使用自定义K-Means进行图像分割并可视化
    
    参数:
    - input_path: 输入图像路径
    - k_values: 要尝试的K值列表
    - output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取并预处理图像
    img = Image.open(input_path)
    img_array = np.array(img)
    original_shape = img_array.shape
    
    # 2. 准备可视化结果
    plt.figure(figsize=(15, 8))
    
    # 显示原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(img_array)
    plt.title('Original')
    plt.axis('off')
    
    # 3. 对每个K值进行分割
    results = []
    for i, K in enumerate(k_values, start=2):
        # 3.1 将图像转换为2D像素数组并归一化
        if len(original_shape) == 3:  # 彩色图像
            h, w, c = original_shape
            X = img_array.reshape(-1, c) / 255.0
        else:  # 灰度图像
            h, w = original_shape
            X = img_array.reshape(-1, 1) / 255.0
        
        # 3.2 执行K-Means聚类
        kmeans = KMeansSegmenter(n_clusters=K, max_iter=30, tol=t)
        kmeans.fit(X)
        
        # 3.3 重建分割后的图像
        segmented = kmeans.centroids[kmeans.labels]
        segmented = (segmented * 255).clip(0, 255).astype(np.uint8)
        segmented_img = segmented.reshape(original_shape)
        
        # 3.4 保存单独的分割结果
        output_path = os.path.join(output_dir, f'kmeans_k{K}.jpg')
        Image.fromarray(segmented_img).save(output_path)
        results.append(segmented_img)
        
        # 3.5 添加到可视化结果
        plt.subplot(2, 3, i)
        plt.imshow(segmented_img)
        plt.title(f'K={K}')
        plt.axis('off')
    
    # 4. 保存和显示比较结果
    comparison_path = os.path.join(output_dir, 'kmeans_comparison_tol.jpg')
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == "__main__":
    # 使用示例
    input_image = 'data/snake.jpg'  # 替换为你的图像路径
    output_directory = 'output/k-means/snake/tol_1e-4'  # 输出目录
    
    try:
        segmented_images = kmeans_segmentation(
            input_image, 
            k_values=[2, 4, 6, 8, 10],  # 自定义K值
            output_dir=output_directory
        )
        print("图像分割完成，结果已保存到:", output_directory)
    except Exception as e:
        print("处理图像时出错:", str(e))