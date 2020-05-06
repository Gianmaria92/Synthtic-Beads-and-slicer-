# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:49:34 2020

@author: Gianmaria
"""

import image_slicer
import glob 
import numpy as np
from PIL import Image
from tqdm import tqdm 
from scipy import signal 
from skimage.util import random_noise
import matplotlib.pyplot as plt

def gaussian_kernel(size=10, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    ker = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return ker/ker.sum()
   
        
path = "C:/Users/Gianmaria/Desktop/test/beads/"
save_dir = path + "Fictious_beads/" 
tile_save_dir = save_dir + "Sliced_beads/"    

nx = 2048
ny = 2048
n_images = 1640
beads_density = 0.0001
n_beads = np.int(nx * ny * beads_density)
SNR = 30
n_tiles = 256
psf_lat = gaussian_kernel(size=5)
max_scale = 2**8



for c in tqdm(range(n_images)):
    name = save_dir + 'beads_' + str(c) + '.png'
    img = np.zeros((nx, ny))
    bins = [np.arange(nx+1), np.arange(ny+1)]
    
#    for i in range(n_beads):
    rand_x = np.random.randint(0, nx-1, size=n_beads)
    rand_y = np.clip(np.random.normal(loc=np.int(ny//2), scale = np.int(ny/8), size=n_beads), 0, ny)
    img, _, _ = np.histogram2d(rand_x, rand_y, bins=bins, weights=np.ones((n_beads,)))
           
    img = signal.fftconvolve(img, psf_lat, mode='same')
    img = img + np.abs(np.min(img))
    noise_level = np.max(img) / SNR
    noise = np.random.normal(0, noise_level, (nx, ny))
    noise_img = np.multiply((img + noise) * max_scale, 10)

    im = Image.fromarray(noise_img).convert('L')  
    im.save(name)
#    tiles = image_slicer.slice(name, n_tiles, save=False)
#    prefix = str(c)+'_slice'
#    image_slicer.save_tiles(tiles, directory=tile_save_dir, prefix=prefix, format='png')                
        


        
        
        
        
        
        