import cv2
import numpy as np
import glob
import os
import networkx as nx
import gc

# 从two_image_stitiching.py导入所有需要的函数
from two_image_stitiching import (
    detect_sift_features, 
    match_features_robust, 
    estimate_homography_ransac,
    stitch_two_images,
    crop_black_borders
)

def resize_image_if_needed(img, max_dimension=400):  # 大幅降低到400
    """如果图像太大则缩放"""
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > max_dimension:
        scale = max_dimension / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"图像从 {w}x{h} 缩放到 {new_w}x{new_h}")
    return img

def detect_sift_features_light(img):
    """轻量级SIFT特征检测"""
    # 进一步减少特征点数量
    sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.05, edgeThreshold=30)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def compute_pairwise_matches_minimal(images):
    """极简内存占用的配对匹配计算"""
    n = len(images)
    match_matrix = np.zeros((n, n), dtype=np.uint8)  # 使用最小数据类型
    
    print("计算图像对之间的匹配...")
    
    # 分批处理，避免同时存储所有特征
    for i in range(n):
        print(f"处理图像 {i}...")
        
        # 只在需要时计算特征
        for j in range(i + 1, n):
            # 重新缩放到更小尺寸进行匹配
            img1_small = resize_image_if_needed(images[i], max_dimension=300)
            img2_small = resize_image_if_needed(images[j], max_dimension=300)
            
            kp1, des1 = detect_sift_features_light(img1_small)
            kp2, des2 = detect_sift_features_light(img2_small)
            
            if des1 is not None and des2 is not None:
                matches = match_features_robust(des1, des2)
                match_count = min(len(matches), 255)  # 限制在uint8范围内
                match_matrix[i, j] = match_count
                match_matrix[j, i] = match_count
                print(f"  与图像 {j}: {match_count} 个匹配点")
            
            # 立即清理所有临时变量
            del img1_small, img2_small, kp1, kp2, des1, des2
            if 'matches' in locals():
                del matches
            gc.collect()
    
    return match_matrix

def build_connection_graph(match_matrix, min_matches=5):  # 大幅降低阈值
    """基于匹配强度构建连接图"""
    n = match_matrix.shape[0]
    G = nx.Graph()
    
    for i in range(n):
        G.add_node(i)
    
    for i in range(n):
        for j in range(i + 1, n):
            if match_matrix[i, j] >= min_matches:
                G.add_edge(i, j, weight=int(match_matrix[i, j]))
    
    return G

def find_stitching_order_simple(G):
    """简化的拼接顺序查找"""
    if not nx.is_connected(G):
        print("警告: 图像不完全连通，使用最大连通分量")
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        nodes = list(largest_component)
    else:
        nodes = list(G.nodes())
    
    if len(G.edges()) == 0:
        return nodes
    
    # 简化策略：选择度数最高的节点开始，然后贪心选择
    degrees = dict(G.degree())
    start_node = max(degrees, key=degrees.get) if degrees else nodes[0]
    
    order = [start_node]
    remaining = set(nodes) - {start_node}
    
    while remaining:
        best_next = None
        best_weight = 0
        
        for current in order:
            if current in G:
                for neighbor in G.neighbors(current):
                    if neighbor in remaining:
                        weight = G[current][neighbor]['weight']
                        if weight > best_weight:
                            best_weight = weight
                            best_next = neighbor
        
        if best_next is not None:
            order.append(best_next)
            remaining.remove(best_next)
        else:
            # 如果找不到连接的节点，随机选择一个
            if remaining:
                next_node = remaining.pop()
                order.append(next_node)
    
    return order

def stitch_two_images_controlled(img1, img2, max_size=800):
    """受控的两图拼接，限制输出尺寸"""
    # 先缩放输入图像
    img1_resized = resize_image_if_needed(img1, max_dimension=max_size)
    img2_resized = resize_image_if_needed(img2, max_dimension=max_size)
    
    result = stitch_two_images(img1_resized, img2_resized)
    
    if result is None:
        result = stitch_two_images(img2_resized, img1_resized)
    
    # 清理中间变量
    del img1_resized, img2_resized
    gc.collect()
    
    # 确保输出不会太大
    if result is not None:
        result = resize_image_if_needed(result, max_dimension=1000)
    
    return result

def multi_image_stitching(images):
    """极简内存占用的多图像拼接"""
    if len(images) < 2:
        return None
    
    if len(images) == 2:
        return stitch_two_images_controlled(images[0], images[1])
    
    print(f"开始多图拼接，共 {len(images)} 张图像")
    
    # 预处理：大幅缩放所有图像
    processed_images = []
    for i, img in enumerate(images):
        # 进一步降低处理尺寸
        processed_img = resize_image_if_needed(img, max_dimension=300)
        processed_images.append(processed_img)
    
    # 立即清理原始图像
    del images
    gc.collect()
    
    # 如果图像太多，只处理前几张
    max_images = 6  # 限制最大图像数量
    if len(processed_images) > max_images:
        print(f"图像数量过多，只处理前{max_images}张图像")
        processed_images = processed_images[:max_images]
        gc.collect()
    
    # 极简匹配计算
    match_matrix = compute_pairwise_matches_minimal(processed_images)
    
    # 构建连接图
    G = build_connection_graph(match_matrix, min_matches=3)  # 进一步降低阈值
    
    # 找到拼接顺序
    order = find_stitching_order_simple(G)
    
    print(f"拼接顺序: {order}")
    
    # 逐步拼接，使用受控的拼接函数
    result = processed_images[order[0]]
    
    for i in range(1, min(len(order), 4)):  # 限制拼接数量
        next_img = processed_images[order[i]]
        print(f"正在拼接第 {i+1} 张图像...")
        
        # 使用受控的拼接函数
        temp_result = stitch_two_images_controlled(result, next_img, max_size=600)
        
        if temp_result is not None:
            result = temp_result
            print(f"成功拼接第 {i+1} 张图像")
        else:
            print(f"无法拼接第 {i+1} 张图像，跳过")
        
        # 每次拼接后强制清理
        if 'temp_result' in locals():
            del temp_result
        gc.collect()
    
    # 清理处理过的图像
    del processed_images
    gc.collect()
    
    return result

def stitch_case(case_name, input_dir, output_dir):
    """处理单个拼接case"""
    image_paths = sorted(glob.glob(os.path.join(input_dir, case_name, "*")))
    
    # 限制加载的图像数量
    max_load = 8
    if len(image_paths) > max_load:
        print(f"图像数量过多({len(image_paths)})，只加载前{max_load}张")
        image_paths = image_paths[:max_load]
    
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            # 加载时就进行初步缩放
            img = resize_image_if_needed(img, max_dimension=500)
            images.append(img)
    
    if len(images) < 2:
        print(f"[{case_name}] Skipped: insufficient or unreadable images.")
        return

    print(f"[{case_name}] 开始多图拼接，实际处理{len(images)}张图像...")
    stitched_image = multi_image_stitching(images)
    
    if stitched_image is None:
        print(f"[{case_name}] 拼接失败")
        return

    case_output_dir = output_dir
    os.makedirs(case_output_dir, exist_ok=True)
    output_path = os.path.join(case_output_dir, f"{case_name}.JPG")
    cv2.imwrite(output_path, stitched_image)
    print(f"[{case_name}] Done: saved to {output_path}")
    
    # 清理所有变量
    del images, stitched_image
    gc.collect()

def main():
    input_root = "data/task2_multiview"
    output_root = "output/task2_multiview"
    os.makedirs(output_root, exist_ok=True)
    
    cases = [name for name in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, name))]
    if not cases:
        print("No cases found in 'data' directory.")
        return

    for case in sorted(cases):
        print(f"\n处理case: {case}")
        stitch_case(case, input_root, output_root)
        # 每个case完成后强制清理内存
        gc.collect()

if __name__ == "__main__":
    main()