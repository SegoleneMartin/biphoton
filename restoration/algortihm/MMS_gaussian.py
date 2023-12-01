#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
========================================================================
Created on Tue Aug 31 2021

@author: segolene.martin@centralesupelec.fr
Inspired form the Matlab pipeline of Emilie Chouzenoux.

PENALIZED LOCAL MAJORATION-MINIMIZATION SUBSPACE ALGORITHM

========================================================================  
"""
import os
import numpy as np
from scipy.linalg import norm
import time
from scipy.linalg import pinv
from scipy.sparse.linalg import LinearOperator
from pathlib import Path
import matplotlib.pyplot as plt

DIRECTORY = Path(__file__).parents[2]
os.chdir(DIRECTORY)
print(DIRECTORY)
from Deblurring.Operators.gradient_operator import generate_G
from Tools.PSNR import PSNR

def P_MMS_gaussian(y, H, delta, lambd, x_0, x_bar, x_min, x_max, res, noise_params, D_scale, max_iter, tolerance):
    """

    Parameters
    ----------
    y : numpy.ndarray of shape (n1, n2, n3)
        Degraded image
    H : LinearOperator of shape (N, N), with N = (n1, n2, n3)
        Blur operator
    G : LinearOperator of shape (N, 3*N)
        Gradient operator
    delta : float
        Regularization parameter
    lambd : float
        Bound on the data-fit constraint
    phi : the regularization function flag as indicated below
        (1) phi(u) = (1-exp(-u.^2./(2*delta^2))); 
        (2) phi(u) = (u.^2)./(2*delta^2 + u.^2); 
        (3) phi(u) = log(1 + (u.^2)./(delta^2)); 
        (4) phi(u) = sqrt(1 + u^2/delta^2)-1; 
        (5) phi(u) = 1/2 u^2; 
        (6) phi(u) = 1-exp(-((1 + (u.^2)./(delta.^2)).^(1/2)-1));
    x_0 : numpy.ndarray of shape (nx, ny, nz)
        Initial point
    x_bar : numpy.ndarray of shape (nx, ny, nz)
        True image (to compute the error)
    x_min : float
        Minimum pixel value.
    x_max : float
        Maximum pixel value.
    res : tuple of 3 floats
        Resolution
    max_iter : int
        Maximum number of iteration
    tolerance : float
        Precision criterion 

    Returns
    -------
    x : numpy.ndarray 
        Last iterate generated by the algorithm

    Minimization of
    ---------------
    F(x) =  sum(phi(sqrt(Gv(x)^2+Gh(x)^2+Gt(x)^2))), 
    subject to:
    dist_[x_min;x_max](x) = 0,
    || Hx - y ||^2 <= lambd.

    """
    #### Define gradient operator G given the resolution ####
    n1, n2, n3 = x_0.shape
    N = n1 * n2 * n3
    G = generate_G((n1, n2, n3), res)
    
    #### Flatten vectors to 1D arrays ####
    y = y.reshape(N)
    x_0 = x_0.reshape(N)
    x_bar = x_bar.reshape(N)
    
    #### Initialize variables ####
   
    criterion = 10000000
    counter = 1
    a1 = 1.1
    a2 = 1.1
    e1 = 1.2
    e2 = 1.2
    GAMMA_1 = (a1)**(e1)     # penalty parameter on pixel-range constraint
    GAMMA_2 = (a2)**(e2)       # penalty parameter on data-fit constraint
    GAMMA_2_old= GAMMA_2 / 2
    GAMMA_2_old_old = GAMMA_2 / 4
    
    GAMMA = np.array([GAMMA_1, GAMMA_2])
    #EPSILON_0 = 10000
    EPSILON_0 = 1
    EPSILON = EPSILON_0 / (GAMMA_2)**(1.5)
    start_time = time.time()

    #### First iteration : MM gradient descent ####
    d = grad_F(x_0, y, lambd, H, G, delta, x_min, x_max, GAMMA, D_scale)
    Ad = lambd * G.T @ (A(G @ x_0, delta, N) @ (G @ d)) + 2 * H.T @ (H @ d) + GAMMA[0] * 2 * d  # curvature of the majorant of F_GAMMA at x_0, applied to d
    Bd = - d.T @ Ad     # curvature of the majorant of f_GAMMA at 0
    u = - 1 / Bd * (d.T @ d)
    step = - u * d
    x = x_0 + step
    x_old = np.copy(x_0)
    Phi_x_old = Phi(G @ x, G, delta)
    Psi_x_old = np.sum((H @ x - y)**2)
    
    #### Solve sequence of subproblems P_GAMMA, with precision EPSILON ####
    while counter < max_iter and criterion > tolerance and time.time()-start_time < 7600:

        if counter >=3 :
            x_0 = x + (x - x_old) * (1/ GAMMA_2 - 1/ GAMMA_2_old) / (1/ GAMMA_2_old - 1/ GAMMA_2_old_old)
        else:
            x_0 = np.copy(x)
            
        # Get last two iterates of MMS applied to P_GAMMA
        print(EPSILON, GAMMA)
        x_inner_new, x_inner, criterion = MMS(y, lambd, H, G, delta, x_0, x_old, x_bar, x_min, x_max, D_scale, 1000, EPSILON, GAMMA, (n1, n2, n3))
      
        # Update variables
        if counter < 2:
            x_old = x_inner
            x = x_inner_new
        else:
            x_old = np.copy(x)
            x = x_inner_new
            
        Phi_x = Phi(G @ x, G, delta)
        Psi_x = np.sum((H @ x - y)**2)
        criterion = abs((Phi_x - Phi_x_old)/Phi_x) + R1(x, x_min, x_max) + abs((Psi_x - Psi_x_old)/Psi_x) 
        Phi_x_old = Phi_x
        Psi_x_old = Psi_x
        
        print('ITER = {}, GAMMA = {}, EPS = {:5f}, criterion = {}'.format(counter, GAMMA, EPSILON, criterion))
        print('--------------------\n')
            
        counter += 1
        
        GAMMA_2_old_old = GAMMA_2_old
        GAMMA_2_old = GAMMA_2
        GAMMA_1 = min((a1*counter)**(e1), 30000000000)
        GAMMA_2 = min((a2*counter)**(e2), 50000000000)
        GAMMA = np.array([GAMMA_1, GAMMA_2])
        if GAMMA_2 == 500000:
            EPSILON = EPSILON /1.25
        else:
            EPSILON = EPSILON_0 / (GAMMA_2)**(0.25)

    return(x.reshape((n1, n2, n3)))


def MMS(y, lambd, H, G, delta, x, x_old, x_bar, x_min, x_max, D_scale, max_iter, EPSILON, GAMMA, shape):
    """
    Parameters
    ----------
    y : numpy.ndarray of shape (N, 1)
        Degraded image
    H : LinearOperator of shape (N, N)
        Blur operator
    G : LinearOperator of shape (N, 3*N)
        Gradient operator
    delta : float
        Regularization parameter
    lambd : float
        Bound on the data-fit constraint
    x : numpy.ndarray of shape (N, 1)
        Initial point
    x_old : numpy.ndarray of shape (N, 1)
        Previous point
    x_bar : numpy.ndarray of shape (N, 1)
        True image (to compute the error)
    x_min : float
        Minimum pixel value.
    x_max : float
        Maximum pixel value.
    max_iter : int
        Maximum number of iteration
    EPSILON : float
        Precision criterion on norm grad_F
    GAMMA : numpy.ndarray of shape (2,)
        Penalty parameters

    Returns
    -------
    x : numpy.ndarray 
        Last iterate generated by the algorithm

    Minimization of
    ---------------
    F_GAMMA(x) =  sum(phi(sqrt(Gv(x)^2+Gh(x)^2+Gt(x)^2))) + GAMMA_1 * R1(x) + GAMMA_2 * R2(x), 
    where:
    R1(x) = dist^2_[x_min;x_max](x),
    R2(x) = dist^2_B(Hx), and B = Ball(y, sqrt(lambd)).
    """
    N = x.shape[0]
    step = x - x_old
    criterion = 10000000 # to enter the loop
    counter = 0

    #### MMS algorithm on F_GAMMA ####
    while counter < max_iter and criterion > EPSILON :
        grad_F_x, Hx, Gx = grad_F(x, y, lambd, H, G, delta, x_min, x_max, GAMMA, D_scale, return_products = True)
        criterion = norm(grad_F_x)
        D = np.vstack((-grad_F_x, step)).T

        if counter == 0 :
            grad_F_x_old = grad_F(x_old, y, lambd, H, G, delta, x_min, x_max, GAMMA, D_scale)
            m_H = - (H @ grad_F_x_old)
            o_H =  (H @ step)
            m_G = - G @ grad_F_x_old
            o_G = G @ step
            m_I = - grad_F_x_old
            o_I = step

        else :
            o_H = m_H * u[0] + o_H * u[1]
            o_G = m_G * u[0] + o_G * u[1]
            o_I = m_I * u[0] + o_I * u[1]

        m_H_new = - (H @ grad_F_x)
        m_G_new = - G @ grad_F_x
        m_I_new = - grad_F_x

        tmp_H = np.vstack((m_H_new, o_H)).T
        tmp_G = np.vstack((m_G_new, o_G)).T
        tmp_I = np.vstack((m_I_new, o_I)).T
        B_tmp = lambd * tmp_G.T @ (A(Gx, delta, N) @ tmp_G) + 2 * tmp_H.T @ tmp_H
        B = B_tmp + 2 * GAMMA[0] * tmp_I.T @ tmp_I
        Dgrad = D.T @ grad_F_x
        u = -pinv(B) @ Dgrad
        step = D @ u
    
        x = x + step
        m_H, m_G, m_I = m_H_new, m_G_new, m_I_new

        if (counter ) % 20 == 0 :
            print("iter = {}, error = {}, PSNR = {:4.2f}, Psi={}, R1 = {}".format(counter, criterion, PSNR(x_bar, x), Phi(Gx, G, delta), R1(x, x_min, x_max)))

        counter += 1

    return(x, x - step, criterion)

def F(x, y, lambd, H, G, delta, x_min, x_max, GAMMA, D_scale):
    Hx = H @ x
    Gx = G @ x
    return(lambd * Phi(Gx, G, delta) + np.sum((Hx -y)**2) + GAMMA[0]*R1(x, x_min, x_max))

def grad_F(x, y, lambd, H, G, delta, x_min, x_max, GAMMA, D_scale, return_products=False):
    """
    Gradient of function F_GAMMA = regularization + penalties

    Parameters
    ----------
    x : numpy.ndarray of shape (N, 1)
        Point at which the gradient is computed
    y : numpy.ndarray of shape (N, 1)
        Degraded image
    H : LinearOperator of shape (N, N)
        Blur operator
    G : LinearOperator of shape (N, 3*N)
        Gradient operator
    delta : float
        Regularization parameter
    lambd : float
        Bound on the data-fit constraint
    x_min : float
        Minimum pixel value.
    x_max : float
        Maximum pixel value.
    GAMMA : numpy.ndarray of shape (2,)
        Penalty parametes
    return_products (optional): bool
        If True, return the products H @ x and G @ x to avoid new calculations later
    """
    Hx = H @ x
    Gx = G @ x
        
    if return_products == False:
        return(lambd * grad_Phi(Gx, G, delta) + grad_Psi(Hx, H, y) + GAMMA[0] * grad_R1(x, x_min, x_max))
    else:
        return(lambd * grad_Phi(Gx, G, delta) + grad_Psi(Hx, H, y) + GAMMA[0] * grad_R1(x, x_min, x_max), Hx, Gx)
    
def Phi(Gx, G, delta):
    N = Gx.shape[0] // 3
    Gx = Gx.reshape(N, 3)
    return(np.sum(np.sqrt(delta + np.sum(Gx**2, axis = 1))))

def grad_Phi(Gx, G, delta):
    N = Gx.shape[0] // 3
    Gx = Gx.reshape(N, 3)
    z = (Gx / np.sqrt(delta + np.sum(Gx**2, axis = 1))[:, np.newaxis]).reshape(3*N)
    return(G.T @ z)

def proj_box(x, x_min, x_max):
    return(x * (x_min <= x) * (x <= x_max) + x_max * (x > x_max) + x_min * (x < x_min))

def R1(x, x_min, x_max):
    return(norm(x - proj_box(x, x_min, x_max))**2)

def grad_R1(x, x_min, x_max):
    return(2 * (x - proj_box(x, x_min, x_max)))

def proj_ball(x, y, r):
    # projection on B(y, r)
    diff = x - y
    n_diff = np.sqrt(diff @ diff)
    if n_diff > r:
        return(y + diff * r / n_diff)
    else:
        return(x)

def grad_Psi(Hx, H, y):
    return 2 * (H.T @ (Hx - y))

def test_in_box(x, x_min, x_max):
        eps = 0.0000
        return(1 - 1 * (x <= x_max +eps) * (x >= x_min - eps))

def A(Gx, delta, N):
    Gx = Gx.reshape(N, 3)
    def Ax_u(u):
        C = 1 / np.sqrt(delta + np.sum(Gx**2, axis = 1))
        curvature_Phi = (C[:, np.newaxis] * u.reshape(N,3)).reshape(3*N,)

        return(curvature_Phi)

    matvec = Ax_u
    return LinearOperator((3*N, 3*N), matvec = matvec, rmatvec = matvec)