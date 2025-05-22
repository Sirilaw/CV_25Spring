import numpy as np
import cv2
import argparse
import random
import os
import math


def generate_random_seeds(img, num_seeds=5):
    h, w = img.shape
    return [(random.randint(0, w-1), random.randint(0, h-1))
            for _ in range(num_seeds)]


def region_growth_seed(img, seed_points, threshold):
    h, w = img.shape
    segmented = np.full((h, w), -1, dtype=np.int32)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    current_region = 0

    for x0, y0 in seed_points:
        if segmented[y0, x0] != -1:
            continue
        stack = [(x0, y0)]
        segmented[y0, x0] = current_region
        seed_val = int(img[y0, x0])

        while stack:
            x0, y0 = stack.pop()
            for dx, dy in neighbors:
                x, y = x0+dx, y0+dy
                if 0 <= x < w and 0 <= y < h and segmented[y, x] == -1:
                    if abs(int(img[y, x]) - seed_val) <= threshold:
                        segmented[y, x] = current_region
                        stack.append((x, y))
        current_region += 1

    return normalize_segmentation(segmented, current_region)


def region_growth_mean(img, seed_points, threshold):
    h, w = img.shape
    segmented = np.full((h, w), -1, dtype=np.int32)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    current_region = 0

    for x0, y0 in seed_points:
        if segmented[y0, x0] != -1:
            continue
        stack = [(x0, y0)]
        segmented[y0, x0] = current_region
        region_sum = int(img[y0, x0])
        region_count = 1

        while stack:
            x0, y0 = stack.pop()
            region_mean = region_sum / region_count
            for dx, dy in neighbors:
                x, y = x0+dx, y0+dy
                if 0 <= x < w and 0 <= y < h and segmented[y, x] == -1:
                    diff = abs(int(img[y, x]) - region_mean)
                    if diff <= threshold:
                        segmented[y, x] = current_region
                        region_sum += int(img[y, x])
                        region_count += 1
                        stack.append((x, y))
        current_region += 1

    return normalize_segmentation(segmented, current_region)


def region_growth_gradient(img, seed_points, threshold):
    # compute gradient magnitude map once
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)

    h, w = img.shape
    segmented = np.full((h, w), -1, dtype=np.int32)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    current_region = 0

    for x0, y0 in seed_points:
        if segmented[y0, x0] != -1:
            continue
        stack = [(x0, y0)]
        segmented[y0, x0] = current_region

        while stack:
            x0, y0 = stack.pop()
            for dx, dy in neighbors:
                x, y = x0+dx, y0+dy
                if 0 <= x < w and 0 <= y < h and segmented[y, x] == -1:
                    if gradient[y, x] <= threshold:
                        segmented[y, x] = current_region
                        stack.append((x, y))
        current_region += 1

    return normalize_segmentation(segmented, current_region)


def normalize_segmentation(segmented, region_count):
    if region_count <= 0:
        return np.zeros_like(segmented, dtype=np.uint8)
    # bring labels into [0,255]
    norm = segmented.astype(np.float32) / (region_count - 1)
    return np.uint8(255 * norm)


def region_growth(img, seed_points, threshold, criterion='seed'):
    if criterion == 'seed':
        return region_growth_seed(img, seed_points, threshold)
    elif criterion == 'mean':
        return region_growth_mean(img, seed_points, threshold)
    elif criterion == 'gradient':
        return region_growth_gradient(img, seed_points, threshold)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def parse_list(arg_str):
    return list(map(int, arg_str.strip('[]').split(',')))


def build_annotated_grid(labeled_imgs, grid_cols,
                         header_h=30, x_gap=10, y_gap=10,
                         font_scale=0.6, font_thickness=1):
    """
    labeled_imgs: list of (img, label) tuples.
    grid_cols: number of columns.
    header_h: height in pixels for the label strip.
    x_gap, y_gap: spacing between cells.
    """
    # Prepare each cell: header + image
    cells = []
    for img, label in labeled_imgs:
        # ensure BGR 3-channel
        if img.ndim == 2:
            cell_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            cell_img = img.copy()
        h, w = cell_img.shape[:2]
        header = np.full((header_h, w, 3), 255, dtype=np.uint8)
        # put label text
        cv2.putText(header, label, (5, header_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        # stack header above image
        cells.append(cv2.vconcat([header, cell_img]))

    # pad to full grid
    n = len(cells)
    rows = math.ceil(n / grid_cols)
    pad_count = rows * grid_cols - n
    black_cell = np.zeros_like(cells[0])
    for _ in range(pad_count):
        cells.append(black_cell)

    # build each row with x_gap
    row_imgs = []
    for r in range(rows):
        row_cells = cells[r*grid_cols:(r+1)*grid_cols]
        row = row_cells[0]
        for cell in row_cells[1:]:
            gap = np.full((row.shape[0], x_gap, 3), 255, dtype=np.uint8)
            row = cv2.hconcat([row, gap, cell])
        row_imgs.append(row)

    # stack rows with y_gap
    grid = row_imgs[0]
    for r in row_imgs[1:]:
        gap_row = np.full((y_gap, grid.shape[1], 3), 255, dtype=np.uint8)
        grid = cv2.vconcat([grid, gap_row, r])

    return grid

def main():
    parser = argparse.ArgumentParser(description='region-growth with criteria')
    parser.add_argument('--num_seeds', default='10,20,30,40,50', type=str,
                        help='e.g. "[10,20,30,40,50]"')
    parser.add_argument('--threshold', default='10,20,30,40,50', type=str,
                        help='e.g. "[10,20,30,40,50]"')
    parser.add_argument('--criterion', default='seed', type=str,
                        choices=['seed','mean','gradient'],
                        help='growth criterion: seed|mean|gradient')
    parser.add_argument('--input', default='data/planets.jpg', type=str)
    parser.add_argument('--output', default='output/Region-Growth', type=str)
    args = parser.parse_args()

    num_seeds_list = parse_list(args.num_seeds)
    thresh_list    = parse_list(args.threshold)
    crit           = args.criterion

    img_color = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Cannot load image: {args.input}")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    os.makedirs(args.output, exist_ok=True)

    # collect images for grid
    labeled = [(img_color, "Original")]

    for ns in num_seeds_list:
        seeds = generate_random_seeds(img_gray, ns)
        for ts in thresh_list:
            seg = region_growth(img_gray, seeds, ts, criterion=crit)
            # fname = f"seg_{crit}_seed{ns}_th{ts}.png"
            # save_path = os.path.join(args.output, fname)
            # cv2.imwrite(save_path, seg)
            # print(f"Saved: {save_path}")
            labeled.append((seg, f"{crit}, s={ns}, t={ts}"))

    # build annotated grid
    grid = build_annotated_grid(
        labeled, grid_cols=len(thresh_list)+1,
        header_h=30, x_gap=10, y_gap=10,
        font_scale=0.6, font_thickness=1
    )

    base = os.path.splitext(os.path.basename(args.input))[0]
    out_grid = os.path.join(args.output, base + f'_{crit}_grid.png')
    cv2.imwrite(out_grid, grid)
    print(f"Saved grid: {out_grid}")

if __name__ == "__main__":
    main()
