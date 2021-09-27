# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:16:29 2019

@author: Andy
"""

import matplotlib.pyplot as plt
import numpy as np

im_path = '..\\..\\EXPERIMENTS\\Italy\\data\\adsa\\20190701_1k2f_60c_volume_ref.bmp'
area = 5.19/3.6 * 7.58/3.6  # mm^2

im = plt.imread(im_path)
im_bw = im[:,:,0] == 0
plt.imshow(im_bw*255)

vol_pix = 0
for r in range(im_bw.shape[0]):
    w_pix = sum(im_bw[r,:])
    vol_pix += np.pi*(w_pix/2)**2
w_im_mm = 7.58/3.6 # width of the image in mm measured in powerpoint
w_im_pix = im.shape[1] # width of image in pixels
pix2mm = w_im_mm / w_im_pix
vol_uL = vol_pix*pix2mm**3

print('volume of drop is %f uL.' % vol_uL)

# compute drop area
num_white = np.sum(im_bw)
num_tot = im_bw.shape[0] * im_bw.shape[1]
num_black = (num_tot - num_white)

drop_area = area*num_black/num_tot

print("drop area = %f mm^2" % drop_area)