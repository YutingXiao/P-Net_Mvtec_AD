import numpy as np

from skimage import feature
import torch


def canny(image, sigma):
    n = image.size(0)
    image = np.array(image.cpu())
    image = image * 255

    edge = torch.zeros((n, 3, image.shape[2], image.shape[3]))
    for j in range(n):
        for i in range(3):
            edge[j, i, :, :] = torch.FloatTensor(feature.canny(image[j, i, :, :], sigma=sigma).astype(np.float32))

    edge = (edge > 0).float()
    edge = edge.max(dim=1)[0].unsqueeze(1)
    return edge


