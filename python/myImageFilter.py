import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def myImageFilter(img0, h):
    ker_height, ker_width = h.shape
    img_height, img_width = img0.shape
    padding_x = ker_width // 2
    padding_y = ker_height // 2
    output_height = (img_height+2*padding_y-ker_height)
    output_width = (img_width+2*padding_x-ker_width)
    
    ker = np.flip(h, axis=1)
    ker = np.flip(ker, axis=0)
    padded_img = np.pad(img0, pad_width=((padding_y, padding_y), (padding_x, padding_x)), mode='constant', constant_values=0)
    window = sliding_window_view(padded_img, (ker_height, ker_width))
    
    output = np.tensordot(window, ker, axes=((2, 3), (0, 1)))
    
    return output