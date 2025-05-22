import cv2
import numpy as np

def StandSobel(src, ddepth=cv2.CV_16S, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT):
    src = cv2.GaussianBlur(src, (3, 3), 0)
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    sobelCombined = cv2.bitwise_or(abs_grad_x, abs_grad_y)

    return sobelCombined

def CustomSobel(src, ddepth=cv2.CV_16S, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT):
    # Your Job
    src = cv2.GaussianBlur(src, (3, 3), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    sobel_x = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], dtype=np.float32
    ) * scale
    sobel_y = np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]], dtype=np.float32
    ) * scale

    grad_x = cv2.filter2D(gray, ddepth=ddepth, kernel=sobel_x, delta=delta, borderType=borderType)
    grad_y = cv2.filter2D(gray, ddepth=ddepth, kernel=sobel_y, delta=delta, borderType=borderType)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    sobelCombined = cv2.bitwise_or(abs_grad_x, abs_grad_y)

    return sobelCombined