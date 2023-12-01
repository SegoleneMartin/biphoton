#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:39:05 2021

@author: segolenemartin
"""
import numpy as np

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
    print("d = ", d)
    
    while abs(MSE_old - MSE_new) > 1e-12:
        print("err = ", abs(MSE_old - MSE_new))
        xq_old = np.copy(xq)
        MSE_old = MSE(x, xq_old)
        print("1")
        r = estimation_level(d, x)
        print("r = ", r)
        print("2")
        d = estimation_interval(r, x_max, x_min)
        print("d = ", d)
        print("3")
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
