#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:00:00 2021

@author: segolenemartin
"""
import numpy as np

def PSNR(x_bar, x, x_max=1) :
    mse = np.mean((x - x_bar) ** 2 )
    if mse == 0:
        return 100
    return 20 * np.log10(x_max / np.sqrt(mse))