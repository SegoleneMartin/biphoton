import numpy as np
from PSF_estimation.utils import mymgrid
import matplotlib.pyplot as plt

def Rx(theta):
    return np.array([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
  
def Ry(theta):
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(theta):
    return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def genC(angles, var):
    phi, theta, psi = angles
    R = Rx(psi) @ Ry(theta) @ Rz(phi)
    return R.T @ np.diag(1 / var) @ R


def gaussian_kernel(C, kernel_size, args):
    d1, d2, d3, _ = mymgrid(kernel_size, args.resolution)
    value = np.einsum('hjkl, hi, ijkl-> jkl', np.array([d1, d2, d3]), C, np.array([d1, d2, d3]))
    kernel = np.exp(-(value / 2.0)) 
    kernel = kernel / np.sum(kernel)
    return kernel


def gaussian_kernel_true( C, kernel_size, args):
    d1, d2, d3, _ = mymgrid(kernel_size, args.resolution)
    beta = 2.0
    value = (np.einsum('hjkl, hi, ijkl-> jkl', np.array([d1, d2, d3]), C, np.array([d1, d2, d3])))**(beta / 2)
    kernel = np.exp(-(value / 2.0)) 
    kernel = kernel / np.sum(kernel)
    return kernel


def gaussian_function( C, kernel_size):
    d1, d2, d3, _  = mymgrid(kernel_size)
    value = np.einsum('hjkl, hi, ijkl-> jkl', np.array([d1, d2, d3]), C, np.array([d1, d2, d3]))
    value = np.exp(-(value / 2.0)) * np.sqrt(np.linalg.det(C) / (2 * np.pi)**3) 
    return value

