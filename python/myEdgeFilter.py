import numpy as np
import math
import scipy
import cv2
import time
import matplotlib.pyplot as plt
from scipy import signal    # For signal.gaussian function
from scipy.signal import windows
from myImageFilter import myImageFilter
from numpy.lib.stride_tricks import sliding_window_view

def myEdgeFilter(img0, sigma):
    #Apply Gaussian Filter
    h_size = 2*math.ceil(3*sigma)+1
    gauss_vector = scipy.signal.windows.gaussian(h_size, std=sigma, sym=True)
    gauss_filter = np.outer(gauss_vector, gauss_vector)
    img0_gauss = myImageFilter(img0, gauss_filter)

    #Sobel X
    sobel_x = np.array([
        [1,0,-1],
        [2,0,-2],
        [1,0,-1]
    ])
    img0_sobel_x = myImageFilter(img0_gauss, sobel_x) 
    
    #Sobel y
    sobel_y = sobel_x.T

    img0_sobel_y = myImageFilter(img0_gauss, sobel_y) 

    #Img gradient magnitude
    img0_gradient_mag = np.sqrt(img0_sobel_x**2 + img0_sobel_y**2) 
    
    #img gradient directions
    round_angle_vect = np.vectorize(round_angle)
    img0_gradient_directions = (np.arctan2(img0_sobel_y,img0_sobel_x) * (180/np.pi)) % 180
    img0_gradient_directions = round_angle_vect(img0_gradient_directions) 

    output = apply_directional_nms(img0_gradient_mag, img0_gradient_directions, 3)
    output = (output / output.max() * 255).astype(np.uint8)
    
    return output

def round_angle(theta):
    angles = [0,45,90,135]
    angle = min(angles, key=lambda x: abs(theta-x))
    return angle

def apply_directional_nms(gradient_mag, gradient_directions, h_size):
    img = gradient_mag
    padding = h_size // 2
    padded_img = padded_img = np.pad(gradient_mag, pad_width=padding, mode='constant', constant_values=0)
    windows = sliding_window_view(padded_img, (h_size, h_size))

    directional_nms_nms_vect = np.vectorize(directional_nms, signature="(m,m),()->()")
    output = directional_nms_nms_vect(windows, gradient_directions)
    
    return output

#Gradient Direction Non-Maximum Supression
def directional_nms(h, theta):
    center_idx = len(h) // 2
    center = h[center_idx, center_idx]
    neighbors = []
    if(theta == 0):
        neighbors = [h[center_idx,0], h[center_idx,-1]]
    elif(theta == 45):
        neighbors = [h[0,0], h[-1,-1]]
    elif(theta == 90):
        neighbors = [h[0, center_idx], h[-1,center_idx]]
    elif(theta == 135):
        neighbors = [h[-1,0], h[0,-1]]

    return center if center >= max(neighbors) else 0