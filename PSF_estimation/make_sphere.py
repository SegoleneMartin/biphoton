import numpy as np
import PSF_estimation.global_variables as gv
from PSF_estimation.utils import mymgrid
#import kernel


def make_sphere(sphere_size, window_size, args, center_y=None):

    rayon = sphere_size / 2
    d1, d2, d3, __ = mymgrid(window_size, args.resolution)

    if center_y is not None:
        center = (center_y - window_size // 2) * args.resolution
    else:
        center = [0, 0, 0]
    sphere = np.where(np.sqrt((d1 - center[0]) ** 2 + (d2 - center[1]) ** 2 + 
                (d3 - center[2]) ** 2) <= rayon, 1, 0)
    return sphere


def linear_attenuation(dist_to_center, rayon, limit_val=0.0):
    """
    :param Omega: 3D array of shape (window_size, window_size, window_size, 3)
    :param sphere_size: size of the sphere in mm
    """
    return (limit_val - 1 / rayon) * dist_to_center + 1

    