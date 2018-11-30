import os
import cv2
import numpy as np
from skimage import filters
from matplotlib import pyplot as plt 
from skimage import data, transform
import scipy.signal as signal

def cal_distance(pa,pb):
    from math import sqrt
    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
    return dis

def lowPassFilter(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    transfor_matrix = np.zeros(image.shape)
    center_point = tuple(map(lambda x:(x-1)/2,image.shape))
    for i in range(transfor_matrix.shape[0]):
        for j in range(transfor_matrix.shape[1]):
            dis = cal_distance(center_point,(i,j))
            if dis <= d:
                transfor_matrix[i,j]=1
            else:
                transfor_matrix[i,j]=0
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*transfor_matrix)))
    return new_img

def highPassFilter(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)   
    transfor_matrix = np.zeros(image.shape)
    center_point = tuple(map(lambda x:(x-1)/2,image.shape))
    for i in range(transfor_matrix.shape[0]):
        for j in range(transfor_matrix.shape[1]):
            dis = cal_distance(center_point,(i,j))
            if dis <= d:
                transfor_matrix[i,j]=0
            else:
                transfor_matrix[i,j]=1
    return transfor_matrix
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*transfor_matrix)))
    return new_img
    
def img_evaluate(img):
    img=lowPassFilter(img1,25)
    img=img/255.
    smd2=0
    tmp = filters.sobel(img)
    tng=0
    for i in range(int(np.ceil(img.shape[0]*0.25)),int(np.ceil(img.shape[0]*0.75))):
        for j in range(int(np.ceil(img.shape[1]*0.25)),int(np.ceil(img.shape[1]*0.75))):
            smd2+=(img[i+1][j]-img[i][j])*(img[i][j+1]-img[i][j])
            tng+=(tmp[i][j]**2)         
    return (0.5*smd2+1.0*(tng))
    
image=cv2.imread("C:\\Users\\boon\\Desktop\\face_s\\120050740.bmp",0)
plt.imshow(image,"gray")
print(img_evaluate(image))
    
    
