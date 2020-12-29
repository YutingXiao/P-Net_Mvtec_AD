import numpy as np
import pydensecrf.densecrf as dcrf

def dense_crf(img, output_probs):
    """
    CRF for single image
    :param img: HWC
    :param output_probs:CHW
    :return:
    """
    h = output_probs.shape[1]
    w = output_probs.shape[2]
    n_classes = output_probs.shape[0]

    d = dcrf.DenseCRF2D(w, h, n_classes)
    U = -np.log(output_probs)
    U = U.reshape((n_classes, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q
