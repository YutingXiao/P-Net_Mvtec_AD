import cv2
import numpy as np
import torch


def dilated_eroded(mask):
    n = mask.size(0)
    output = np.zeros_like(np.array(mask))
    for i in range(n):
        mask_np = np.array(mask[i])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(mask_np, kernel)
        eroded = cv2.erode(dilated, kernel)

        output[i] = eroded

    output = torch.FloatTensor(output)
    return output