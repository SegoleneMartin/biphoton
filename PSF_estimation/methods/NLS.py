import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.fft import fftn
from scipy.ndimage import center_of_mass
import numpy as np
from ..make_sphere import make_sphere
from ..observation3D import observ_slice, observ_profile_convolution, observ_profile_kernel
from PSF_estimation.utils import compute_angles, compute_variances, generate_X, mymgrid, coef_c
import time
from PSF_estimation import global_variables as gv
from ..kernel import gaussian_kernel, gaussian_kernel_true, genC
import os
import datetime
from scipy.optimize import least_squares



# Create class NLS which has to be intialized with args, path_crops and name_crop
class NLS():

    def __init__(self, args):
        self.args = args

    def pipeline(self, path_crops, name_crop):
        self.path_crops = path_crops
        self.name_crop = name_crop
        self.args.resolution = np.array(self.args.resolution)

        # Load bead crop
        y = np.load(path_crops + name_crop, allow_pickle=True)
        center_y = np.round(np.array(center_of_mass(y)))
        self.args.kernel_size = np.array(y.shape)

        # Generate theoritical sphere
        x = make_sphere(sphere_size = self.args.big_sphere_size, window_size = self.args.kernel_size, args=self.args, center_y=center_y)

        # Run PSF estimation for all regularization parameters lambdas
        self.args.plots = []
        self.args.plot_names = []
        observ_slice(x, 0, 'x', self.args)
        observ_slice(y, 0, 'y', self.args)

        save_path_results =  'results/' + self.args.image_folder_name + '/' + self.args.method_PSF + '/' + name_crop[0] + '/'
        try:
            os.makedirs(save_path_results)
        except:
            pass
        
        t = time.time()
        a_est, b_est, D_est, h_args_est = self.PSFEstimation(x, y)
        temps_exec = time.time() - t

        self.record_results(save_path_results, x, y, a_est, b_est, D_est, h_args_est, temps_exec)

    def record_results(self, save_path_results, x, y, a_est, b_est, D_est, h_args_est, temps_exec):
        # Record results in a txt file
        L, U = np.linalg.eigh(D_est)
        angles_est = compute_angles(U[:, 0])
        variances_est = compute_variances(L, U)
        FWHM_est = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(variances_est)

        txt = f"date = {datetime.datetime.now()}" \
                f"\npath crops = {self.path_crops}" \
                f"\nname crop = {self.name_crop}" \
                f"\nresolution = {self.args.resolution} " \
                f"\nkernel_size = {self.args.kernel_size})" \
                f"\n\nESTIMATED VALUES" \
                f"\na_est = {a_est}" \
                f"\nb_est = {b_est}" \
                f"\nD_est = {D_est}" \
                f"\nangles_phi_est = {angles_est[0]}" \
                f"\nangles_theta_est = {angles_est[1]}" \
                f"\nvariances_est = {variances_est}" \
                f"\nFWHM_est = {FWHM_est} " \
                f"\n\nUSED PARAMETERS" \
                f"\ngam_D = {self.args.gam_D}" \
                f"\n\nExecution time: = {temps_exec}"

        results_file = open(save_path_results + "results.txt", "w")
        results_file.write(txt)
        results_file.close()

        # Save plots
        try:
            os.mkdir(save_path_results + "plots")
        except:
            pass
        for i in range(len(self.args.plots)):
            fig = self.args.plots[i]
            fig.savefig(save_path_results + "plots/" + self.args.plot_names[i] + ".png")
        plt.close('all')

        # Save values
        try:
            os.mkdir(save_path_results + "values")
        except:
            pass
        np.save(save_path_results + "values/" + "h_args_est" + ".npy", h_args_est)
        np.save(save_path_results + "values/" + "D_est" + ".npy", D_est)
        np.save(save_path_results + "values/" + "x" + ".npy", x)
        np.save(save_path_results + "values/" + "y" + ".npy", y)

    def PSFEstimation(self, x, y):

        self.args.plots = []
        self.args.plot_names = []

        X, X_T = generate_X(x)
        lb = np.array([0.0, 0.0, -np.pi, 0.0, 1e-6, 1e-6, 1e-6])
        ub = np.array([2.0, 4.0, np.pi, np.pi, 2, 2, 3])    

        # Initialize variables
        theta = np.pi - np.pi / 6
        phi = 0.0
        sigma2_1 = 0.01
        sigma2_2 = 0.01
        sigma2_3 = 0.5
        a = 0.0
        b = 0.5
        u0 = np.array([a, b, phi, theta, sigma2_1, sigma2_2, sigma2_3])

        # Least squares minimization
        res_lsq = least_squares(self.residual_function, u0, bounds=(lb, ub), args=(y, X))
        a, b, phi, theta, sigma2_1, sigma2_2, sigma2_3 = res_lsq.x

        # Compute D_est, h_args_est
        angles_est = np.array([phi, theta, 0.0])
        variances_est = np.array([sigma2_1, sigma2_2, sigma2_3])
        D_est = genC(angles_est, variances_est)
        h_args_est = gaussian_kernel(D_est, self.args.kernel_size, args=self.args)
        observ_slice(h_args_est, 0, "h_args_est", self.args)
        if self.args.simulation_PSF == True:
            h = gaussian_kernel_true(gv.D_true, self.args.kernel_size, args=self.args)
            observ_slice(h, 0, "h_true", self.args)
        observ_slice(fftconvolve(h_args_est, x, "same"), 0, "Convolution h_args_est with x", self.args)
        observ_profile_convolution(a + b * fftconvolve(h_args_est, x, "same"), y, "Convolution h_args_est with x", self.args)
        observ_profile_kernel(h_args_est, "h_args_est", self.args)

        return a, b, D_est, h_args_est


    def residual_function(self, u, y, X):
        a, b, phi, theta, sigma2_1, sigma2_2, sigma2_3 = u
        angles = np.array([phi, theta, 0.0])
        var = np.array([sigma2_1, sigma2_2, sigma2_3])
        D = genC(angles, var)
        h = gaussian_kernel(D, self.args.kernel_size, args=self.args)
        convo = X(h)
        return (y - a - b * convo).flatten()

