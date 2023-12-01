import numpy as np
from scipy.signal import fftconvolve
#import make_sphere, kernel
import PSF_estimation.global_variables as gv
from PSF_estimation.make_sphere import make_sphere
from PSF_estimation.kernel import gaussian_kernel_true


def gen_observation(sphere_size, D,  kernel_size, window_size, args):
    my_sphere = make_sphere(sphere_size, window_size).astype('float64')
    generated_h = gaussian_kernel_true( D, kernel_size, args)
    im = gv.a_true + gv.b_true * fftconvolve(my_sphere, generated_h, 'same')
    # im = my_sphere
    return im, generated_h, my_sphere
    
