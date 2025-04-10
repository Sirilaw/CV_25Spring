#!/usr/bin/python3

from typing import Tuple

import numpy as np

import numpy.fft as fft

from proj1_code.utils import load_image, save_image, PIL_resize, numpy_arr_to_PIL_image, PIL_image_to_numpy_arr, im2single, single2im


def psnr(img1, img2, max_val: float = 1.0) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)

def low_pass_filter(freq_repr: np.ndarray, ratio: float) -> np.ndarray:
    h, w = freq_repr.shape
    cy, cx = h // 2, w // 2
    radius_y, radius_x = int(cy * ratio), int(cx * ratio)

    mask = np.zeros_like(freq_repr, dtype=bool)
    mask[cy - radius_y:cy + radius_y, cx - radius_x:cx + radius_x] = True

    filtered = np.zeros_like(freq_repr, dtype=complex)
    filtered[mask] = freq_repr[mask]
    return filtered

def compress_image_frequency(image: np.ndarray, ratios: list) -> None:
    original = image
    h, w, c = original.shape

    for r in ratios:
        reconstructed = np.zeros_like(original)

        for ch in range(3):  # 对 R/G/B 每个通道分别处理
            freq = fft.fftshift(fft.fft2(original[:, :, ch]))
            filtered = low_pass_filter(freq, r)
            compressed = fft.ifft2(fft.ifftshift(filtered))
            reconstructed[:, :, ch] = np.real(compressed)

        psnr_val = psnr(original, reconstructed)
        print(f"Ratio: {r}, PSNR: {psnr_val:.2f} dB")

        output_path = f"../results/part4/compressed_{r}.png"
        save_image(output_path, reconstructed)




if __name__ == "__main__":
    image = load_image("../data/1b_cat.bmp")
    ratios = [0.1, 0.3, 0.5, 0.7]
    compress_image_frequency(image, ratios)


