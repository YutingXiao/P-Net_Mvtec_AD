import torch
import cv2 as cv
import numpy as np
import pdb


def image_blurr(image, kernel_size=10, sigma=0, convert='L'):
    # sigma = 0 时，其值由W和H自动决定
    image = np.array(image.cpu())
    n, c, w, h = image.shape

    image_blurred = np.zeros((n, 1, w, h)) if convert == 'L' else np.zeros((n, 3, w, h))
    for i in range(n):
        if convert == 'L':
            image_blurred[i, 0] = np.mean(image[i], axis=0)
        elif convert == 'RGB':
            image_blurred[i] = image[i]

        if sigma != 0:
            for j in range(3):
                image_blurred[i, j] = cv.GaussianBlur(image_blurred[i, j], (kernel_size, kernel_size), sigma)
        else:
            pass

    return torch.FloatTensor(image_blurred)

