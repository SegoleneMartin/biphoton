import numpy as np
from .kernel import genC, gaussian_kernel, gaussian_kernel_true, gaussian_function

big_sphere_size = 1  # micrometers

# # 2 case possible : 'real', 'simulation"
# case = 'real'
# #case = 'real'

# if case == 'real':
#     real = True
    
# else:  # case == 'simulation':
#     real = False
# simulation = not real

# if real:
#     kernel_size = None
    # resolution = np.array((0.049, 0.049, 0.1))
    # a_min = 0.0
    # a_max = 0.1
    # b_min = 0.00005
    # b_max = 5.0
    # gam_h = 1e-7
    # gamma = 1e1
    # gam_D = gamma * 100000
    # gam_a = gamma
    # gam_b = gamma
    # gam_s = gamma * 10000
    # gam_p = gamma


# else:
beta = 2.0
resolution = np.array([0.049, 0.049, 0.1])
n_billes = 1
sigma_noise = 0.01
FWHM_true = np.array([0.2, 0.2, 1.3])
variances_true = (FWHM_true / 2.355) ** 2
angles_true = np.array([-np.pi / 4, np.pi - np.pi/4, 0.0])
window_size = [101, 101, 101]
kernel_size = window_size
a_true = 0
b_true = 1
C = genC(angles_true, variances_true)
D_true = genC(angles_true, variances_true)

a_min = 0.0
a_max = 0.1
b_min = 0.00005
b_max = 3.0
lam = 1e-8
gam_h = 1e-7
gamma = 1e1
gam_mu = gamma
gam_D = gamma * 1
gam_a = gamma
gam_b = gamma
gam_s = gamma * 10000
gam_p = gamma


reussi = False

