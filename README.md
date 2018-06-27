import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity

def convolve(image, kernel, pad, stride):
   
    (image_Height, image_Width) = image.shape[:2]
    (kernel_Height, kernel_Width) = kernel.shape[:2]
   
    b,g,r = cv.split(img)
    b = cv.copyMakeBorder(b, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    g = cv.copyMakeBorder(g, pad, pad, pad, pad, 0)
    r = cv.copyMakeBorder(r, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    padding = int((kernel_Width - 1) / 2)
    output_b = np.zeros((int((image_Height+pad*2-kernel_Height)/stride+1), int((image_Width+pad*2-kernel_Width)/stride+1)), dtype="float32")
    output_g = np.zeros((int((image_Height+pad*2-kernel_Height)/stride+1), int((image_Width+pad*2-kernel_Width)/stride+1)), dtype="float32")
    output_r = np.zeros((int((image_Height+pad*2-kernel_Height)/stride+1), int((image_Width+pad*2-kernel_Width)/stride+1)), dtype="float32")
   
    for y in np.arange(padding, int((image_Height+pad*2-kernel_Height)/stride+1)):
        for x in np.arange(padding, int((image_Width+pad*2-kernel_Width)/stride+1)):
           
            roi_b = b[int((stride*y) - padding):int((stride*y)+ padding + 1),
                        int((stride*x) - padding):int((stride*x)+ padding + 1)]
            roi_g = g[int((stride*y) - padding):int((stride*y)+ padding + 1),
                        int((stride*x) - padding):int((stride*x)+ padding + 1)] 
            roi_r = r[int((stride*y) - padding):int((stride*y)+ padding + 1),
                        int((stride*x) - padding):int((stride*x)+ padding + 1)] 
           
            k_b = (roi_b * kernel).sum()
            k_g = (roi_g * kernel).sum()
            k_r = (roi_r * kernel).sum()           
          
            output_b[y - padding, x - padding] = k_b
            output_g[y - padding, x - padding] = k_g
            output_r[y - padding, x - padding] = k_r           
           
    output_b = rescale_intensity(output_b, in_range=(0, 255))
    output_b = (output_b * 255).astype("uint8")
    output_g = rescale_intensity(output_g, in_range=(0, 255))
    output_g = (output_g * 255).astype("uint8")
    output_r = rescale_intensity(output_r, in_range=(0, 255))
    output_r = (output_r * 255).astype("uint8")
    output = cv.merge((output_b,output_g,output_r))
 
  
    return output
img = cv.imread('soccer.jpg')
#kernel = np.ones((5,5),np.float32)/25  ->>averaging blurring kernel
kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) # 파이썬에서 np를 써서 배열을 만들면 성능이 더 좋아짐
kernel2 = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
dst = convolve(img,kernel,1,1)
dst2 = convolve(img,kernel2,100,5)
#dst=cv.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])
dst3 = cv.filter2D(img,-1,kernel)
cv.imshow('img',dst)
cv.imshow('img2',img)
cv.imshow('img3', dst2)
cv.imshow('img4', dst3)
cv.waitKey(0)
cv.destroyAllWindows()
