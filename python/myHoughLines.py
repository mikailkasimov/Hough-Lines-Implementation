import matplotlib.pyplot as plt
import numpy as np
import cv2
from myHoughTransform import myHoughTransform
from numpy.lib.stride_tricks import sliding_window_view

def apply_nms(img, h_size):
    padding = h_size // 2
    padded_img = np.pad(img, pad_width=padding, mode='constant', constant_values=0)
    windows = sliding_window_view(padded_img, (h_size, h_size))
    nms_vect = np.vectorize(
        lambda h: h[len(h) // 2, len(h) // 2] if h[len(h) // 2, len(h) // 2] >= np.max(h) else 0,\
        signature = "(m,m)->()"
    )
    output = nms_vect(windows)
    return output


def myHoughLines(H, nLines):
    h_nms = apply_nms(H, 7)
    rhos, thetas = np.unravel_index(np.argsort(h_nms, axis=None)[-nLines:], h_nms.shape)
    return [rhos,thetas]

