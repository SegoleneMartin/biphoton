#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:51:27 2021

@author: segolenemartin
"""
import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft, ifft, fftn, ifftn, fftshift, ifftshift


def gaussian_kernel(KERNEL_SIZE, KERNEL_SIGMA):
    x, y, z = np.mgrid[- KERNEL_SIZE : KERNEL_SIZE + 1, -KERNEL_SIZE : KERNEL_SIZE + 1, - KERNEL_SIZE : KERNEL_SIZE + 1]
    normal = 1 / (2.0 * np.pi * KERNEL_SIGMA**2)
    kernel =  np.exp(-((x**2 + y**2 + z**2) / (2.0 * KERNEL_SIGMA**2))) * normal
    kernel  = kernel / np.sum(kernel)
    return(kernel)


def blur(x, shape, KERNEL_SIZE=4, sigma=2):
    n1, n2, n3 = shape
    #KERNEL_SIZE = 4
    h = gaussian_kernel(KERNEL_SIZE, sigma)
    kernel_h = np.pad(h, ((n1//2, n1//2), (n2//2, n2//2), (n3//2, n3//2)),  mode="constant")
    fft_h = fftn(ifftshift(kernel_h))
    
    if n1 % 2 == 0:
        s1 = (KERNEL_SIZE, KERNEL_SIZE+1)
    else :
        s1 = (KERNEL_SIZE, KERNEL_SIZE)
    if n2 % 2 == 0:
        s2 = (KERNEL_SIZE, KERNEL_SIZE+1)
    else :
        s2 = (KERNEL_SIZE, KERNEL_SIZE)
    if n3 % 2 == 0:
        s3 = (KERNEL_SIZE, KERNEL_SIZE+1)
    else :
        s3 = (KERNEL_SIZE, KERNEL_SIZE)
        
    x = np.pad(x, (s1, s2, s3), mode="edge")
    fft_x = fftn(x)
    x_new = np.real(ifftn(fft_x * fft_h))
    x_new = x_new[s1[0]:-s1[1], s2[0]:-s2[1], s3[0]:-s3[1]]
    
    return(x_new)

def MSE(x, y):
    return(np.mean((x-y)**2))

def estimation_level(d, x):
    r = np.zeros(len(d)-1)
    for n in range(len(d)-1):
        if n < len(d)-2:
            S = (x >= d[n]) * (x < d[n+1])
        else:
            S = (x >= d[n]) * (x <= d[n+1])
        r[n] = np.sum(S*x)/np.sum(S)
    return r

def estimation_interval(r, x_max, x_min):
    d = np.zeros(len(r)+1)
    d[0] = x_min
    for n in range(1, len(r)):
        d[n] = (r[n-1] + r[n]) / 2
    d[-1] = x_max
    return d


def LM_quantifier(x, J):
    xq = np.copy(x)
    MSE_old = 1
    MSE_new = 0
    x_max = np.max(x)
    x_min = np.min(x)
    q = x_max / J
    r = np.zeros(J)
    d = np.arange(np.min(x), J*q, q)
    
    while abs(MSE_old - MSE_new) > 1e-7:
        print("err = ", abs(MSE_old - MSE_new))
        xq_old = np.copy(xq)
        MSE_old = MSE(x, xq_old)
        r = estimation_level(d, x)
        d = estimation_interval(r, x_max, x_min)
        list_S = []
        for n in range(len(r)):
            if n < len(r)-2:
                S = (x >= d[n]) * (x < d[n+1])
            else:
                S = (x >= d[n]) * (x <= d[n+1])
            list_S.append(S)
            xq = xq * (1-S) + r[n]*S
        MSE_new = MSE(x, xq)
    return(xq, list_S, d, r)



