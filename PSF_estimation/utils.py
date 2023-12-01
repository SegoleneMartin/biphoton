import numpy as np
from scipy.signal import fftconvolve
import PSF_estimation.global_variables as gv
from scipy.fft import fftn, ifftn, ifftshift, fftshift



def coef_c( D, Omega, eps):
    Q = 3
    return Q / 2 * np.log(2 * np.pi) + 1 / 2 * miniphi(D, eps) + 1 / 2 * np.einsum('hijk, kl,hijl-> hij', Omega,
                                                                                   (D + eps * np.eye(3)), Omega)


def miniphi(D, eps):
    return np.log(np.linalg.det(D + eps * np.eye(3)))


def get_barycentre(im):
    xi, yi, zi = np.mgrid[0:im.shape[0],
                          0:im.shape[1],
                          0:im.shape[2]]
    xg = np.sum(np.multiply(xi, im)) / np.sum(im)
    yg = np.sum(np.multiply(yi, im)) / np.sum(im)
    zg = np.sum(np.multiply(zi, im)) / np.sum(im)
    return xg, yg, zg


def get_a(im):
    size = 2
    kernel = np.ones((size, size, size)) / size ** 3
    im_flou = fftconvolve(im, kernel, "same")
    return np.max(im_flou)

def mymgrid(shape, res):
    half_size = np.array(shape) // 2
    r1, r2, r3 = res
    d1 = np.linspace(- r1 * half_size[0], half_size[0] * r1, shape[0])
    d2 = np.linspace(- r2 * half_size[1], half_size[1] * r2, shape[1])
    d3 = np.linspace(- r3 * half_size[2], half_size[2] * r3, shape[2])
    d1, d2, d3 = np.meshgrid(d1, d2, d3, indexing='ij')
    return d1, d2, d3, np.stack((d1, d2, d3), axis=3)

def myprint(*args):
    if gv.debug:
        for i in args:
            print(i, end=" ")
        print()

def compute_angles(V):
    if V[2] < 0:
        V = - V
    u1, u2, u3 = V    
    phi = -np.arctan2(u2, u1) 
    theta = np.arctan2(u3, np.sqrt(u1**2 + u2**2)) + np.pi / 2
    return phi, theta

def compute_variances(L, U):
    var = [0, 0, 0]
    var[2] = 1 / L[0]
    permutation = [2, 0, 1]
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    U_bis = U[:, idx]

    if abs(np.linalg.det(U_bis) - 1) <= 1e-04:
        var[0] = 1 / L[1]
        var[1] = 1 / L[2]
    else:
        var[0] = 1 / L[2]
        var[1] = 1 / L[1]
    return var

def fft(array):
    fft = fftn(ifftshift(array, axes=(0, 1, 2)), axes=(0, 1, 2))
    return fft

def ifft(array):
    ifft = fftshift(ifftn(array, axes=(0, 1, 2)), axes=(0, 1, 2))
    return ifft

def generate_X(x):
    n1, n2, n3 = x.shape
    fft_x = fft(x)    

    def X(u):
        u = u.reshape(n1, n2, n3)
        fft_u = fft(u)
        u_new = np.real(ifft(fft_u * fft_x))
        return u_new
    
    def X_T(u):
        u = u.reshape(n1, n2, n3)
        fft_u = fft(u)
        u_new = np.real(ifft(fft_u * np.conj(fft_x)))
        return u_new
    
    return X, X_T
