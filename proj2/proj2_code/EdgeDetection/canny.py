import cv2
import numpy as np

def StandCanny(img, lowThreshold, highThreshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, lowThreshold, highThreshold)
    return edges

def CustomCanny(img, lowThreshold, highThreshold):
    # Step 1: Grayscale + Gaussian Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Step 2: Sobel gradient on blurred image (CV_16S → float32)
    grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
    grad_x = grad_x.astype(np.float32)
    grad_y = grad_y.astype(np.float32)

    grad_mag = np.hypot(grad_x, grad_y)
    grad_dir = np.arctan2(grad_y, grad_x)  # in radians

    # Step 3: Non-Maximum Suppression (improved version)
    def NMS(mag, angle):
        H, W = mag.shape
        Z = np.zeros((H, W), dtype=np.uint8)
        angle = np.rad2deg(angle) % 180

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255
                    a = angle[i, j]

                    if (0 <= a < 22.5) or (157.5 <= a <= 180):
                        q = mag[i, j + 1]
                        r = mag[i, j - 1]
                    elif (22.5 <= a < 67.5):
                        q = mag[i + 1, j - 1]
                        r = mag[i - 1, j + 1]
                    elif (67.5 <= a < 112.5):
                        q = mag[i + 1, j]
                        r = mag[i - 1, j]
                    elif (112.5 <= a < 157.5):
                        q = mag[i - 1, j - 1]
                        r = mag[i + 1, j + 1]

                    if (mag[i, j] >= q) and (mag[i, j] >= r):
                        Z[i, j] = mag[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError:
                    pass
        return Z

    nms = NMS(grad_mag, grad_dir)

    # Step 4: Double thresholding
    strong = 255
    weak = 75
    res = np.zeros_like(nms, dtype=np.uint8)
    strong_i, strong_j = np.where(nms >= highThreshold)
    weak_i, weak_j = np.where((nms >= lowThreshold) & (nms < highThreshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # Step 5: Hysteresis edge tracking (Flood fill)
    def hysteresis(img, weak=75, strong=255):
        H, W = img.shape
        strongs = [(i, j) for i in range(1, H - 1) for j in range(1, W - 1) if img[i, j] == strong]
        visited = set(strongs)

        while strongs:
            i, j = strongs.pop()
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < H) and (0 <= nj < W):
                        if img[ni, nj] == weak and (ni, nj) not in visited:
                            img[ni, nj] = strong
                            strongs.append((ni, nj))
                            visited.add((ni, nj))
        img[img != strong] = 0
        return img

    result = hysteresis(res, weak=weak, strong=strong)
    return result