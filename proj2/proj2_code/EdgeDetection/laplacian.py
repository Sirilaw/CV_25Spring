import cv2
import numpy as np

# https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
def StandLaplacian(src, ddepth = cv2.CV_16S, kernel_size=3):
    # laplacian = cv2.Laplacian(image, cv2.CV_8U)

    # Remove noise by blurring with a Gaussian filter
    src = cv2.GaussianBlur(src, (3, 3), 0)

    # Convert the image to grayscale
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Apply Laplace function
    dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)

    # converting back to uint8
    abs_dst = cv2.convertScaleAbs(dst)

    return abs_dst

def CustomLaplacian(src, ddepth = cv2.CV_16S, kernel_size=3):
    # Your Job
    src = cv2.GaussianBlur(src, (3, 3), 0)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    if kernel_size == 1:
        laplacian_filter = np.array(
            [[2, 0, 2],
            [0, -8, 0],
            [2, 0, 2]], dtype=np.float32
        )
    elif kernel_size == 3:
        laplacian_filter = np.array(
            [[2, 0, 2],
            [0, -8, 0],
            [2, 0, 2]], dtype=np.float32
        )
    else:
        ValueError("The value of kernel size if not implemented")

    dst = cv2.filter2D(src_gray, ddepth, laplacian_filter)

    abs_dst = cv2.convertScaleAbs(dst)

    return abs_dst