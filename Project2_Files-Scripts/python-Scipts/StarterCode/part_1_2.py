# This project will address the problem of "denoising" images with different methods.
# Besides the particular tasks listed, the student will need to find data that extends
# the datasets provides, quantify noise levels in images (relative to a noiseless
# "ground truth"), produce sets of noisy images with known/documented noise characteristics.
# For all of the methods described the student should experiment with parameters and different data/images
# and comment on the results and how it relates to the methodology

# Build and experiment with several different linear filters (also different sizes) using
# correlation/convolution.  Quantify (e.g. using MSE) their effectiveness (and compare
# quantitatively and qualitatively)with different levels of noise, types of noise, and images.
# Experiment with 3 different nonlinear denoising methods, such as bilateral filtering.  If
# you use a method that was not discussed in class, explain how it works.  As in 1) of
# this assignment experiment, quantify, and show results for different types and levels of
# noise, parameters, etc.

# For libraries, please use only the follow (that's all you need):  skimage, numpy, 
# matplotlib, pandas,  pytorch, random, argparse.  Please do not use open-cv, it makes it 
# harder to grade/evaluate your work.

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import sys
import os
import code
sys.path.append("../../utils")
from utils import NoiseDatsetLoader

import code
import os
import skimage
from skimage import io

import matplotlib.pyplot as plt
import numpy as np
from ip_functions import *
from skimage import data, filters, color, morphology, exposure, img_as_float, img_as_ubyte, util
from skimage.util import img_as_ubyte
from skimage.segmentation import flood, flood_fill
from skimage.morphology import extrema
from skimage.exposure import histogram


# 
# Submission Guidelines:
# 
# Please submit two separate folders - one for parts 1 & 2 and the other for the neural network part.
# For the neural network part of the assignment:
# Do not upload large training and testing datasets.
# Include a folder called "models" and save all the pre-trained models in this folder.
# Include a few images(~10) for inference/testing which demonstrate the different noise types you experimented with
# For parts 1 &2 part of the assignment:
# Include a few images(~5) for testing which demonstrate the different noise types you experimented with

## Files setup

# https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
# Used the above link as a reference on best practices with MSE, particularly on FP considerations
def my_mse(A, B):
    error = np.sum(np.square((A.astype("float") - B.astype("float"))))
    error /= float(A.shape[0] * B.shape[1])
    return error

def prob(CDF, idx):
    idx = int(len(CDF)*idx)
    return CDF[idx]

def cdf1(pdf):
    cdf = np.zeros((len(pdf),))
    for ii in range(len(pdf)):
      cdf[ii] = sum(pdf[0:ii])
    return  cdf

# Self Implemented generation of noise
#######################################################
def add_gamma(img_gray, gshape, gscale):
  y_len = np.shape(img_gray)[0]
  x_len = np.shape(img_gray)[1]
  g_noise = np.random.gamma(gshape, gscale, (y_len, x_len))
  gamma_img = np.add(g_noise, img_gray)
  gamma_img[gamma_img < 0] = 0.0
  return gamma_img

def add_norm(img_gray, mu, sgm):
  y_len = np.shape(img_gray)[0]
  x_len = np.shape(img_gray)[1]
  norm_noise = np.random.normal(mu, sgm, (y_len, x_len))
  norm_img = np.add(norm_noise, img_as_float(img_gray))
  norm_img[norm_img < 0] = 0.0
  return norm_img

def norm_arr(img, mu_arr, sgm_arr):
  result_arr = []
  for ii in range(len(mu_arr)):
    result_arr.append(add_norm(img, mu_arr[ii], sgm_arr[ii]))
  return result_arr

def gamma_arr(img, gshape_arr, gscale_arr):
  result_arr = []
  for ii in range(len(mu_arr)):
    result_arr.append(add_gamma(img, gshape_arr[ii], gscale_arr[ii]))
  return result_arr

def add_hist(img, bins):
  hist = np.histogram(img_as_float(img), bins)
  return hist

def circ_msk(k_size):
    msk = np.ones((k_size, k_size))
    msk[k_size-1][k_size-1] = 0
    msk[0][k_size-1] = 0
    msk[k_size-1][0] = 0    
    msk[0][0] = 0
    return msk

# Build and experiment with several different linear filters (also different sizes) using
# correlation/convolution.

camera = io.imread('../my_images/camera.png')
ducati = io.imread('../my_images/Ducati.png')
camera_gray = color2grey(camera)
ducati_gray = color2grey(ducati)
poke0 = io.imread('../my_images/0.png')
poke1 = io.imread('../my_images/1.png')




#plt.imshow(circ_msk(3), cmap='gray')
#plt.show()
#exit()

# Creating gaussian/normal noise
mu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
var_arr = np.arange(0, 0.2*9, 0.2)
norm_noise_scale = 10
sgm = np.arange(0, norm_noise_scale*9, norm_noise_scale)
cam_noises = norm_arr(camera_gray, mu, sgm)

fig, axis = plt.subplots(3,3)
axis = axis.ravel()
for ii in range(len(cam_noises)):
  axis[ii].imshow(cam_noises[ii], cmap='gray')
  axis[ii].set_title('Sigma: ' +  str(sgm[ii]))
plt.suptitle("Levels of Gaussian Noise")
plt.show()
plt.savefig('Norm_noise_levels')

###################### Compare Kernel Sizes in filtering Gaussian Noise

print("Mean ---------------------------------------------------------")

mu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
norm_noise_scale = 10
sgm = np.arange(0, norm_noise_scale*9, norm_noise_scale)
cam_noises = norm_arr(camera_gray, mu, sgm)
#cam_noises = gamma_arr(camera_gray, mu, sgm)
hist_arr = []
for jj in range(len(cam_noises)):
  hist_arr.append(cam_noises[jj])
  hist_arr.append(np.histogram(cam_noises[jj], 512)[0])

fig1, axis = plt.subplots(2*3,3)
axis = axis.ravel()
for ii in range(0, len(hist_arr), 2):
  axis[ii].imshow(hist_arr[ii], cmap='gray')
  axis[ii+1].plot(hist_arr[ii+1])
  axis[ii+1].set_title('hist')
plt.show()
plt.savefig('Norm_noise_with_hist')


cNoise = cam_noises[2]/cam_noises[2].max()  
c3  = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((3 ,3 )))  
c5  = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((5 ,5 )))  
c7  = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((7 ,7 )))
c9  = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((9 ,9 )))  
c11 = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((11,11)))  
c13 = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((13,13)))  
c15 = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), np.ones((15,15)))  


mse_original = my_mse(cam_noises[0], cam_noises[0])
mse_noise = my_mse(cam_noises[0], cNoise)
mse_c3  =  my_mse(cam_noises[0], c3)
mse_c5  =  my_mse(cam_noises[0], c5)
mse_c7  =  my_mse(cam_noises[0], c7)
mse_c9  =  my_mse(cam_noises[0], c9)
mse_c11 =  my_mse(cam_noises[0], c11)
mse_c13 =  my_mse(cam_noises[0], c13)
mse_c15 =  my_mse(cam_noises[0], c15)

print('mse_original ',mse_original )
print('mse_noise ',mse_noise )
print('mse_c3 ',mse_c3 )
print('mse_c5 ',mse_c5 )
print('mse_c7 ',mse_c7 )
print('mse_c9 ',mse_c9 )
print('mse_c11',mse_c11)
print('mse_c13',mse_c13)
print('mse_c15',mse_c15)


plt.subplot(331)
plt.suptitle("Mean Filtering of Various Kernel Sizes")
plt.title('Original')
plt.imshow(cam_noises[0], cmap='gray')
plt.subplot(332)
plt.title('Original with noise (No Filtering)')
plt.imshow(cam_noises[2]/cam_noises[2].max(), cmap='gray')
plt.subplot(333)
plt.imshow(c3, cmap='gray')
plt.title('3x3, MSE: %d' % (mse_c3))
plt.subplot(334)
plt.imshow(c5, cmap='gray')
plt.title('5x5, MSE: %d' % (mse_c5))
plt.subplot(335)
plt.imshow(c7, cmap='gray')
plt.title('7x7, MSE: %d' % (mse_c7))
plt.subplot(336)
plt.imshow(c9, cmap='gray')
plt.title('9x9, MSE: %d' % (mse_c9))
plt.subplot(337)
plt.imshow(c11, cmap='gray')
plt.title('11x11, MSE: %d' % (mse_c11))
plt.subplot(338)
plt.imshow(c13, cmap='gray')
plt.title('13x13, MSE: %d' % (mse_c13))
plt.subplot(339)
plt.imshow(c15, cmap='gray')
plt.title('15x15, MSE: %d' % (mse_c15))
plt.show()
plt.savefig('Comparison_Mean_Filter_Kernel_Sizes_Gaussian_Noise')


###################### Compare Kernel Sizes in filtering Gaussian Noise

print("Mean Circ ---------------------------------------------------------")

mu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
norm_noise_scale = 10
sgm = np.arange(0, norm_noise_scale*9, norm_noise_scale)
cam_noises = norm_arr(camera_gray, mu, sgm)
#hist_arr = []
#for jj in range(len(cam_noises)):
#  hist_arr.append(cam_noises[jj])
#  hist_arr.append(np.histogram(cam_noises[jj], 512)[0])
#
#fig1, axis = plt.subplots(2*3,3)
#axis = axis.ravel()
#for ii in range(0, len(hist_arr), 2):
#  axis[ii].imshow(hist_arr[ii], cmap='gray')
#  axis[ii+1].plot(hist_arr[ii+1])
#  axis[ii+1].set_title('hist')
#
#plt.show()
#plt.savefig('Norm_noise_with_hist')
  

cNoise = cam_noises[2]/cam_noises[2].max()  
c3  = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), circ_msk(3))  
c5  = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), circ_msk(5))  
c7  = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), circ_msk(7))
c9  = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), circ_msk(9))  
c11 = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), circ_msk(11))  
c13 = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), circ_msk(13))  
c15 = filters.rank.mean(cam_noises[2]/cam_noises[2].max(), circ_msk(15))  


mse_original = my_mse(cam_noises[0], cam_noises[0])
mse_noise = my_mse(cam_noises[0], cNoise)
mse_c3  =  my_mse(cam_noises[0], c3)
mse_c5  =  my_mse(cam_noises[0], c5)
mse_c7  =  my_mse(cam_noises[0], c7)
mse_c9  =  my_mse(cam_noises[0], c9)
mse_c11 =  my_mse(cam_noises[0], c11)
mse_c13 =  my_mse(cam_noises[0], c13)
mse_c15 =  my_mse(cam_noises[0], c15)

print('mse_original ',mse_original )
print('mse_noise ',mse_noise )
print('mse_c3 ',mse_c3 )
print('mse_c5 ',mse_c5 )
print('mse_c7 ',mse_c7 )
print('mse_c9 ',mse_c9 )
print('mse_c11',mse_c11)
print('mse_c13',mse_c13)
print('mse_c15',mse_c15)

plt.subplot(331)
plt.suptitle("Mean Circular Filtering of Various Kernel Sizes")
plt.title('Original')
plt.imshow(cam_noises[0], cmap='gray')
plt.subplot(332)
plt.title('Original with noise (No Filtering)')
plt.imshow(cam_noises[2]/cam_noises[2].max(), cmap='gray')
plt.subplot(333)
plt.imshow(c3, cmap='gray')
plt.title('3x3, MSE: %d' % (mse_c3))
plt.subplot(334)
plt.imshow(c5, cmap='gray')
plt.title('5x5, MSE: %d' % (mse_c5))
plt.subplot(335)
plt.imshow(c7, cmap='gray')
plt.title('7x7, MSE: %d' % (mse_c7))
plt.subplot(336)
plt.imshow(c9, cmap='gray')
plt.title('9x9, MSE: %d' % (mse_c9))
plt.subplot(337)
plt.imshow(c11, cmap='gray')
plt.title('11x11, MSE: %d' % (mse_c11))
plt.subplot(338)
plt.imshow(c13, cmap='gray')
plt.title('13x13, MSE: %d' % (mse_c13))
plt.subplot(339)
plt.imshow(c15, cmap='gray')
plt.title('15x15, MSE: %d' % (mse_c15))
plt.show()
plt.savefig('Comparison_Mean_Circular_Filter_Kernel_Sizes_Gaussian_Noise')


###################### Gaussian

print("Gaussian ---------------------------------------------------------")
mu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
norm_noise_scale = 10
sgm = np.arange(0, norm_noise_scale*9, norm_noise_scale)
cam_noises = norm_arr(camera_gray, mu, sgm)
hist_arr = []
#for jj in range(len(cam_noises)):
#  hist_arr.append(cam_noises[jj])
#  hist_arr.append(np.histogram(cam_noises[jj], 512)[0])
#
#fig1, axis = plt.subplots(2*3,3)
#axis = axis.ravel()
#for ii in range(0, len(hist_arr), 2):
#  axis[ii].imshow(hist_arr[ii], cmap='gray')
#  axis[ii+1].plot(hist_arr[ii+1])
#  axis[ii+1].set_title('hist')
#
#plt.show()
#plt.savefig('Norm_noise_with_hist')

cNoise = cam_noises[2]/cam_noises[2].max()  
c3  = filters.gaussian(cam_noises[2]/cam_noises[2].max(), sigma=1)
c5  = filters.gaussian(cam_noises[2]/cam_noises[2].max(), sigma=0.1)
c7  = filters.gaussian(cam_noises[2]/cam_noises[2].max(), sigma=0.4)
c9  = filters.gaussian(cam_noises[2]/cam_noises[2].max(), sigma=0.8)
c11 = filters.gaussian(cam_noises[2]/cam_noises[2].max(), sigma=1.6)
c13 = filters.gaussian(cam_noises[2]/cam_noises[2].max(), sigma=3.2)
c15 = filters.gaussian(cam_noises[2]/cam_noises[2].max(), sigma=6.4)


mse_original = my_mse(cam_noises[0], cam_noises[0])
mse_noise = my_mse(cam_noises[0], cNoise)
mse_c3  =  my_mse(cam_noises[0], c3)
mse_c5  =  my_mse(cam_noises[0], c5)
mse_c7  =  my_mse(cam_noises[0], c7)
mse_c9  =  my_mse(cam_noises[0], c9)
mse_c11 =  my_mse(cam_noises[0], c11)
mse_c13 =  my_mse(cam_noises[0], c13)
mse_c15 =  my_mse(cam_noises[0], c15)

print('mse_original ',mse_original )
print('mse_noise ',mse_noise )
print('mse_c3 ',mse_c3 )
print('mse_c5 ',mse_c5 )
print('mse_c7 ',mse_c7 )
print('mse_c9 ',mse_c9 )
print('mse_c11',mse_c11)
print('mse_c13',mse_c13)
print('mse_c15',mse_c15)

plt.figure()
plt.title('Comparison of Gaussian Filter Kernel Sizes')
plt.subplot(331)
plt.title('Original')
plt.imshow(cam_noises[0], cmap='gray')
plt.subplot(332)
plt.title('Original with noise (No Filtering)')
plt.imshow(cam_noises[2]/cam_noises[2].max(), cmap='gray')
plt.subplot(333)
plt.imshow(c3, cmap='gray')
plt.title('Sigma 1.0, MSE: %d' %(mse_c3))
plt.subplot(334)
plt.imshow(c5, cmap='gray')
plt.title('Sigma 0.1, MSE: %d' %(mse_c5))
plt.subplot(335)
plt.imshow(c7, cmap='gray')
plt.title('Sigma 0.4, MSE: %d' % (mse_c7))
plt.subplot(336)
plt.imshow(c9, cmap='gray')
plt.title('Sigma 0.8, MSE: %d' % (mse_c9))
plt.subplot(337)
plt.imshow(c11, cmap='gray')
plt.title('Sigma 1.6, MSE: %d' % (mse_c11))
plt.subplot(338)
plt.imshow(c13, cmap='gray')
plt.title('Sigma 3.2, MSE: %d' % (mse_c13))
plt.subplot(339)
plt.imshow(c15, cmap='gray')
plt.title('Sigma 6.4, MSE: %d' % (mse_c15))
plt.show()
plt.savefig('Comparison_Gaussian_Filter_Kernel_Sizes_Gaussian_Noise')

####################### Compare Kernel Sizes in filtering ___________ Noise

print("Median ---------------------------------------------------------")

mu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
norm_noise_scale = 10
sgm = np.arange(0, norm_noise_scale*9, norm_noise_scale)
cam_noises = norm_arr(camera_gray, mu, sgm)
hist_arr = []
for jj in range(len(cam_noises)):
  hist_arr.append(cam_noises[jj])
  hist_arr.append(np.histogram(cam_noises[jj], 512)[0])

fig2, axis = plt.subplots(2*3,3)
axis = axis.ravel()
for ii in range(0, len(hist_arr), 2):
  axis[ii].imshow(hist_arr[ii], cmap='gray')
  axis[ii+1].plot(hist_arr[ii+1])
  axis[ii+1].set_title('hist')

plt.show()
plt.savefig('Norm_noise_with_hist')

cNoise = cam_noises[2]/cam_noises[2].max()  
c3  = filters.rank.median(cam_noises[2]/cam_noises[2].max(), np.ones((3 ,3 )))  
c5  = filters.rank.median(cam_noises[2]/cam_noises[2].max(), np.ones((5 ,5 )))  
c7  = filters.rank.median(cam_noises[2]/cam_noises[2].max(), np.ones((7 ,7 )))
c9  = filters.rank.median(cam_noises[2]/cam_noises[2].max(), np.ones((9 ,9 )))  
c11 = filters.rank.median(cam_noises[2]/cam_noises[2].max(), np.ones((11,11)))  
c13 = filters.rank.median(cam_noises[2]/cam_noises[2].max(), np.ones((13,13)))  
c15 = filters.rank.median(cam_noises[2]/cam_noises[2].max(), np.ones((15,15)))  

mse_original = my_mse(cam_noises[0], cam_noises[0])
mse_noise = my_mse(cam_noises[0], cNoise)
mse_c3  =  my_mse(cam_noises[0], c3)
mse_c5  =  my_mse(cam_noises[0], c5)
mse_c7  =  my_mse(cam_noises[0], c7)
mse_c9  =  my_mse(cam_noises[0], c9)
mse_c11 =  my_mse(cam_noises[0], c11)
mse_c13 =  my_mse(cam_noises[0], c13)
mse_c15 =  my_mse(cam_noises[0], c15)

print('mse_original ',mse_original )
print('mse_noise ',mse_noise )
print('mse_c3 ',mse_c3 )
print('mse_c5 ',mse_c5 )
print('mse_c7 ',mse_c7 )
print('mse_c9 ',mse_c9 )
print('mse_c11',mse_c11)
print('mse_c13',mse_c13)
print('mse_c15',mse_c15)

plt.subplot(331)
plt.suptitle("Median Filtering of Various Kernel Sizes")
plt.title('Original')
plt.imshow(cam_noises[0], cmap='gray')
plt.subplot(332)
plt.title('Original with noise (No Filtering)')
plt.imshow(cam_noises[2]/cam_noises[2].max(), cmap='gray')
plt.subplot(333)
plt.imshow(c3, cmap='gray')
plt.title('3x3, MSE: %d' % (mse_c3))
plt.subplot(334)
plt.imshow(c5, cmap='gray')
plt.title('5x5, MSE: %d' % (mse_c5))
plt.subplot(335)
plt.imshow(c7, cmap='gray')
plt.title('7x7, MSE: %d' % (mse_c7))
plt.subplot(336)
plt.imshow(c9, cmap='gray')
plt.title('9x9, MSE: %d' % (mse_c9))
plt.subplot(337)
plt.imshow(c11, cmap='gray')
plt.title('11x11, MSE: %d' % (mse_c11))
plt.subplot(338)
plt.imshow(c13, cmap='gray')
plt.title('13x13, MSE: %d' % (mse_c13))
plt.subplot(339)
plt.imshow(c15, cmap='gray')
plt.title('15x15, MSE: %d' % (mse_c15))
plt.show()
plt.savefig('Comparison_Median_Filter_Kernel_Sizes_Gaussian_Noise')


####################### subtract mean Noise

print("Mean Bilateral ---------------------------------------------------------")

mu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
norm_noise_scale = 10
sgm = np.arange(0, norm_noise_scale*9, norm_noise_scale)
cam_noises = norm_arr(camera_gray, mu, sgm)
#hist_arr = []
#for jj in range(len(cam_noises)):
#  hist_arr.append(cam_noises[jj])
#  hist_arr.append(np.histogram(cam_noises[jj], 512)[0])
#
#fig2, axis = plt.subplots(2*3,3)
#axis = axis.ravel()
#for ii in range(0, len(hist_arr), 2):
#  axis[ii].imshow(hist_arr[ii], cmap='gray')
#  axis[ii+1].plot(hist_arr[ii+1])
#  axis[ii+1].set_title('hist')
#
#plt.show()
#plt.savefig('Norm_noise_with_hist')

cNoise = cam_noises[2]/cam_noises[2].max()  
c3  = filters.rank.mean_bilateral(cam_noises[2]/cam_noises[2].max(), np.ones((3 ,3 )))  
c5  = filters.rank.mean_bilateral(cam_noises[2]/cam_noises[2].max(), np.ones((5 ,5 )))  
c7  = filters.rank.mean_bilateral(cam_noises[2]/cam_noises[2].max(), np.ones((7 ,7 )))
c9  = filters.rank.mean_bilateral(cam_noises[2]/cam_noises[2].max(), np.ones((9 ,9 )))  
c11 = filters.rank.mean_bilateral(cam_noises[2]/cam_noises[2].max(), np.ones((11,11)))  
c13 = filters.rank.mean_bilateral(cam_noises[2]/cam_noises[2].max(), np.ones((13,13)))  
c15 = filters.rank.mean_bilateral(cam_noises[2]/cam_noises[2].max(), np.ones((15,15)))  

mse_original = my_mse(cam_noises[0], cam_noises[0])
mse_noise = my_mse(cam_noises[0], cNoise)
mse_c3  =  my_mse(cam_noises[0], c3)
mse_c5  =  my_mse(cam_noises[0], c5)
mse_c7  =  my_mse(cam_noises[0], c7)
mse_c9  =  my_mse(cam_noises[0], c9)
mse_c11 =  my_mse(cam_noises[0], c11)
mse_c13 =  my_mse(cam_noises[0], c13)
mse_c15 =  my_mse(cam_noises[0], c15)

print('mse_original ',mse_original )
print('mse_noise ',mse_noise )
print('mse_c3 ',mse_c3 )
print('mse_c5 ',mse_c5 )
print('mse_c7 ',mse_c7 )
print('mse_c9 ',mse_c9 )
print('mse_c11',mse_c11)
print('mse_c13',mse_c13)
print('mse_c15',mse_c15)

plt.subplot(331)
plt.suptitle("Mean Bilateral Filtering of Various Kernel Sizes")
plt.title('Original')
plt.imshow(cam_noises[0], cmap='gray')
plt.subplot(332)
plt.title('Original with noise (No Filtering)')
plt.imshow(cam_noises[2]/cam_noises[2].max(), cmap='gray')
plt.subplot(333)
plt.imshow(c3, cmap='gray')
plt.title('3x3, MSE: %d' % (mse_c3))
plt.subplot(334)
plt.imshow(c5, cmap='gray')
plt.title('5x5, MSE: %d' % (mse_c5))
plt.subplot(335)
plt.imshow(c7, cmap='gray')
plt.title('7x7, MSE: %d' % (mse_c7))
plt.subplot(336)
plt.imshow(c9, cmap='gray')
plt.title('9x9, MSE: %d' % (mse_c9))
plt.subplot(337)
plt.imshow(c11, cmap='gray')
plt.title('11x11, MSE: %d' % (mse_c11))
plt.subplot(338)
plt.imshow(c13, cmap='gray')
plt.title('13x13, MSE: %d' % (mse_c13))
plt.subplot(339)
plt.imshow(c15, cmap='gray')
plt.title('15x15, MSE: %d' % (mse_c15))
plt.show()
plt.savefig('Comparison_Mean_Bilateral_Kernel_Sizes_Gaussian_Noise')


# Quantify (e.g. using MSE) their effectiveness (and compare
# quantitatively and qualitatively)with different levels of noise, types of noise, and images.
# Experiment with 3 different nonlinear denoising methods, such as bilateral filtering.  If
# you use a method that was not discussed in class, explain how it works.  As in 1) of
# this assignment experiment, quantify, and show results for different types and levels of
# noise, parameters, etc.

# For libraries, please use only the follow (that's all you need):  skimage, numpy, 
# matplotlib, pandas,  pytorch, random, argparse.  Please do not use open-cv, it makes it 
# harder to grade/evaluate your work.




