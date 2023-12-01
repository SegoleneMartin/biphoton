import numpy as np
import random
from gen_observation import gen_observation
import global_variables as gv
#import observation3D
import matplotlib.pyplot as plt
from skimage import io as skio
from scipy.io import savemat
import kernel
import os

taille_image = gv.kernel_size
ws = gv.window_size

def SNR(u, v) :
    return 10*np.log10(np.mean(u)/(np.std(v-u)))

def noise_level_for_SNR(x, desired_SNR_dB=10):
    """
    Calculate the noise level (standard deviation) for a desired SNR.
    
    Parameters:
        x (numpy array): Ground truth image.
        desired_SNR_dB (float): Desired SNR in dB.
        
    Returns:
        float: Noise level (standard deviation).
    """
    signal_power = np.var(x)
    
    # Calculate the desired noise power
    noise_power = signal_power / (10 ** (desired_SNR_dB / 10))
    
    # Return the standard deviation (square root of the variance)
    return np.sqrt(noise_power)

def compute_SNR(x, y):
    """
    Compute the Signal-to-Noise Ratio (SNR) in dB.
    
    Parameters:
        x (numpy array): Ground truth image.
        y (numpy array): Noisy image.
        
    Returns:
        float: SNR in dB.
    """
    # Calculate the signal power and the noise power
    signal_power = np.var(x)
    noise_power = np.var(y - x)

    # Avoid division by zero in the case of a perfect signal (no noise)
    if noise_power == 0:
        return float('inf')  # Return positive infinity for perfect signals
    
    return 10 * np.log10(signal_power / noise_power)

SNR = 20
vary_beta = False
if vary_beta :
    list_beta = np.arange(1.0, 4.2, 0.25)
else :
    list_beta = [gv.beta]

for beta in list_beta :
    try :
        #os.mkdir("images/simu/{}/".format(beta))
        os.mkdir("images/simu/")
    except :
        pass

    gv.beta = beta
    #gv.sigma_noise = sigma_noise

    for i in range(gv.n_billes):
        y_big = np.zeros(taille_image)
        y_small = np.zeros(taille_image)

        margin = taille_image[0] // 2
        centre = [np.random.randint(margin, taille_image[j] - margin) for j in range(3)]
        # centre = [margin , margin , margin ]

        y_big[centre[0] - ws[0] // 2 : centre[0] + ws[0] // 2 + 1,
        centre[1] - ws[1] // 2: centre[1] + ws[1] // 2 + 1,
        centre[2] - ws[2] // 2: centre[2] + ws[2] // 2 + 1] += gen_observation(sphere_size=gv.big_sphere_size, D=gv.D_true, mu=gv. window_size=ws, kernel_size=gv.kernel_size)[0]

        y_big_clean = np.copy(y_big)
        gv.sigma_noise = noise_level_for_SNR(y_big_clean, desired_SNR_dB=SNR)
        y_big = y_big_clean + gv.sigma_noise * np.random.randn(taille_image[0], taille_image[1], taille_image[2])

        #skio.imsave("images/simu/y_big.tif", y_big)
        print(compute_SNR(y_big, y_big_clean))
        #np.save("images/simu/{}/y_big_{}.npy".format(gv.beta, i), y_big)
        np.save("images/simu/y_big_{}.npy".format(i), y_big)
        #np.save("images/simu/{}/y_big_clean_{}.npy".format(gv.sigma_noise, i), y_big_clean)
        # skio.imsave("images/simu/{}/y_big_{}.tif".format(beta, i), y_big)
