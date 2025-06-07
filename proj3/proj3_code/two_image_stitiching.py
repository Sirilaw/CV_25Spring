import cv2
import numpy as np
import glob
import os
import networkx as nx

def detect_sift_features(img):
    """使用SIFT检测特征点"""
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.03, edgeThreshold=20)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features_robust(des1, des2, ratio_threshold=0.7):
    """使用改进的特征匹配算法"""
    if des1 is None or des2 is None:
        return []
    
    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return []
    
    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches

def estimate_homography_ransac(kp1, kp2, matches):
    """使用RANSAC估计单应性矩阵"""
    if len(matches) < 10:
        return None, None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # 使用更严格的RANSAC参数
    H, mask = cv2.findHomography(
        src_pts, dst_pts, 
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.995
    )
    
    return H, mask

def create_seamless_blend_mask(img1_warped, img2, overlap_region):
    """创建用于无缝融合的渐变掩码"""
    h, w = img2.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    # 找到重叠区域的边界
    overlap_coords = np.where(overlap_region > 0)
    if len(overlap_coords[0]) == 0:
        return mask
    
    min_x, max_x = np.min(overlap_coords[1]), np.max(overlap_coords[1])
    
    # 创建从左到右的线性渐变
    for x in range(w):
        if x <= min_x:
            mask[:, x] = 0.0
        elif x >= max_x:
            mask[:, x] = 1.0
        else:
            # 在重叠区域内创建平滑过渡
            alpha = (x - min_x) / (max_x - min_x)
            mask[:, x] = alpha
    
    # 应用高斯模糊使过渡更平滑
    mask = cv2.GaussianBlur(mask, (21, 21), 10)
    
    return mask

def warp_and_blend_images(img1, img2, H):
    """图像变换和高质量融合"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 计算输出画布大小
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners1_transformed = cv2.perspectiveTransform(corners1, H)
    
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    all_corners = np.concatenate([corners1_transformed, corners2], axis=0)
    
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    
    # 添加边距
    margin = 50
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin
    
    # 平移矩阵
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    
    # 输出尺寸
    output_width = x_max - x_min
    output_height = y_max - y_min
    
    # 变换第一张图像
    img1_warped = cv2.warpPerspective(
        img1, 
        translation @ H, 
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # 创建输出画布并放置第二张图像
    result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # 计算第二张图像在输出画布中的位置
    y_offset = -y_min
    x_offset = -x_min
    
    # 确保不超出边界
    y_end = min(y_offset + h2, output_height)
    x_end = min(x_offset + w2, output_width)
    
    if y_offset >= 0 and x_offset >= 0 and y_end > y_offset and x_end > x_offset:
        result[y_offset:y_end, x_offset:x_end] = img2[:y_end-y_offset, :x_end-x_offset]
    
    # 创建掩码来识别重叠区域
    img1_mask = cv2.cvtColor(img1_warped, cv2.COLOR_BGR2GRAY) > 0
    img2_mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) > 0
    overlap_mask = img1_mask & img2_mask
    
    # 创建用于融合的渐变掩码
    if np.any(overlap_mask):
        blend_mask = create_seamless_blend_mask(img1_warped, result, overlap_mask)
        blend_mask = np.expand_dims(blend_mask, axis=2)
        
        # 在重叠区域进行加权融合
        for c in range(3):
            overlap_region = overlap_mask.astype(bool)
            result[:, :, c] = np.where(
                overlap_region,
                img1_warped[:, :, c] * (1 - blend_mask[:, :, 0]) + 
                result[:, :, c] * blend_mask[:, :, 0],
                result[:, :, c]
            )
        
        # 在非重叠区域直接使用变换后的图像
        non_overlap_img1 = img1_mask & (~overlap_mask)
        for c in range(3):
            result[:, :, c] = np.where(
                non_overlap_img1,
                img1_warped[:, :, c],
                result[:, :, c]
            )
    else:
        # 如果没有重叠，直接叠加
        img1_valid = img1_mask
        for c in range(3):
            result[:, :, c] = np.where(
                img1_valid,
                img1_warped[:, :, c],
                result[:, :, c]
            )
    
    return result

def crop_black_borders(img):
    """裁剪黑边"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # 找到非零像素的边界
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # 添加小的边距
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        
        return img[y:y+h, x:x+w]
    
    return img

def stitch_two_images(img1, img2):
    """主要的图像拼接函数"""
    # 检测特征点
    kp1, des1 = detect_sift_features(img1)
    kp2, des2 = detect_sift_features(img2)
    
    if des1 is None or des2 is None:
        print("无法检测到足够的特征点")
        return None
    
    # 匹配特征点
    matches = match_features_robust(des1, des2)
    
    if len(matches) < 10:
        print(f"匹配点不足: {len(matches)}")
        return None
    
    print(f"找到 {len(matches)} 个特征匹配点")
    
    # 计算单应性矩阵
    H, mask = estimate_homography_ransac(kp1, kp2, matches)
    
    if H is None:
        print("无法计算单应性矩阵")
        return None
    
    # 变换和融合图像
    result = warp_and_blend_images(img1, img2, H)
    
    # 裁剪黑边
    result = crop_black_borders(result)
    
    return result

def stitch_case(case_name, input_dir, output_dir):
    image_paths = sorted(glob.glob(os.path.join(input_dir, case_name, "*")))
    images = [cv2.imread(p) for p in image_paths]
    if any(img is None for img in images) or len(images) != 2:
        print(f"[{case_name}] Skipped: insufficient or unreadable images.")
        return

    # 实现拼接算法
    print(f"[{case_name}] 开始拼接...")
    
    # 尝试两种拼接顺序
    stitched_image = stitch_two_images(images[0], images[1])
    
    if stitched_image is None:
        print(f"[{case_name}] 尝试反向拼接...")
        stitched_image = stitch_two_images(images[1], images[0])
    
    if stitched_image is None:
        print(f"[{case_name}] 拼接失败")
        return

    case_output_dir = output_dir
    os.makedirs(case_output_dir, exist_ok=True)
    output_path = os.path.join(case_output_dir, f"{case_name}.JPG")
    cv2.imwrite(output_path, stitched_image)
    print(f"[{case_name}] Done: saved to {output_path}")

def main():
    input_root = "data/task1_pairwise"
    output_root = "output/task1_pairwise"
    os.makedirs(output_root, exist_ok=True)
    cases = [name for name in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, name))]
    if not cases:
        print("No cases found in 'data' directory.")
        return

    for case in sorted(cases):
        stitch_case(case, input_root, output_root)

if __name__ == "__main__":
    main()