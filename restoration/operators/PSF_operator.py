#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:42:30 2021

@author: segolenemartin
"""

import numpy as np
from scipy.fft import fftn, ifftn, ifftshift
from scipy.sparse.linalg import LinearOperator

def generate_blur(h, shape):
    
    n1, n2, n3 = shape
    N = n1*n2*n3
    KERNEL_SIZE = h.shape[0]//2
    kernel_h = np.pad(h, ((n1//2, n1//2), (n2//2, n2//2), (n3//2, n3//2)),  mode="constant")
    fft_h = fftn(ifftshift(kernel_h))


    def H(x):
        x = x.reshape(n1, n2, n3)
        x = np.pad(x, ((KERNEL_SIZE, KERNEL_SIZE+1), (KERNEL_SIZE, KERNEL_SIZE+1), (KERNEL_SIZE, KERNEL_SIZE+1)), mode="edge")
        fft_x = fftn(x)
        x_new = np.real(ifftn(fft_x * fft_h))
        x_new = x_new[KERNEL_SIZE:-KERNEL_SIZE-1, KERNEL_SIZE:-KERNEL_SIZE-1, KERNEL_SIZE:-KERNEL_SIZE-1]
        return(x_new.reshape(N))
    
    def H_T(x):
        x = x.reshape(n1, n2, n3)
        x = np.pad(x, ((KERNEL_SIZE, KERNEL_SIZE+1), (KERNEL_SIZE, KERNEL_SIZE+1), (KERNEL_SIZE, KERNEL_SIZE+1)), mode="edge")
        fft_x = fftn(x)
        x_new = np.real(ifftn(fft_x * np.conj(fft_h)))
        x_new = x_new[KERNEL_SIZE:-KERNEL_SIZE-1, KERNEL_SIZE:-KERNEL_SIZE-1, KERNEL_SIZE:-KERNEL_SIZE-1]
        return(x_new.reshape(N))
    
    
    H = LinearOperator((N, N), matvec = H, rmatvec = H_T)
    
    return(H)