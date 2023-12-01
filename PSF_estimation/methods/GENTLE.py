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
from ..kernel import gaussian_kernel, gaussian_kernel_true
import os
import datetime
import math


# Create class GENTLE which has to be intialized with args, path_crops and name_crop
class GENTLE():

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
        for i, lambd in enumerate(self.args.lambdas):
            self.args.plots = []
            self.args.plot_names = []
            observ_slice(x, 0, 'x', self.args)
            observ_slice(y, 0, 'y', self.args)
            save_path_results =  'results/' + self.args.image_folder_name + '/' + self.args.method_PSF + '/' + name_crop[0] + '/' + 'lambd_{}/'.format(lambd) 
            try:
                os.makedirs(save_path_results)
            except:
                pass
            print("\n======> Execution for lambda = ", lambd)
            t = time.time()
            a_est, b_est, h_est, D_est, h_args_est = self.PSFEstimation(x, y, lambd)
            temps_exec = time.time() - t

            self.record_results(save_path_results, x, y, lambd, a_est, b_est, h_est, D_est, h_args_est, temps_exec)

    def record_results(self, save_path_results, x, y, lambd, a_est, b_est, h_est, D_est, h_args_est, temps_exec):
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
                f"\nlambda = {lambd}" \
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
        np.save(save_path_results + "values/" + "h_est" + ".npy", h_est)
        np.save(save_path_results + "values/" + "h_args_est" + ".npy", h_args_est)
        np.save(save_path_results + "values/" + "D_est" + ".npy", D_est)
        np.save(save_path_results + "values/" + "x" + ".npy", x)
        np.save(save_path_results + "values/" + "y" + ".npy", y)

    def PSFEstimation(self, x, y, lambd):

        self.args.plots = []
        self.args.plot_names = []

        # Initialize variables
        a = 0
        b = 1
        h = np.abs(y) / np.sum(y)
        D = np.diag(1 / self.args.resolution)

        X, X_T = generate_X(x)
        fftn2 = np.max(np.abs(fftn(x)))**2
        gam_h = 2 / (self.args.b_max**2 * fftn2)
        _, _, _, Omega = mymgrid(self.args.kernel_size, self.args.resolution)
        
        s = self.args.resolution[0] * self.args.resolution[1] * self.args.resolution[2]
        list_iter = []
        list_errors = []

        for i in range(self.args.n_iter):
            convo = X(h)
            new_a = self.update_a(y, b, convo)
            new_b = self.update_b(y, new_a, convo)
            new_h = self.update_h(y, X_T, new_a, new_b, s, h, D, convo, Omega, self.args.epsD, gam_h, lambd)
            new_D = self.update_D(new_h, D, Omega, self.args.epsD, lambd) 
            stop_new = np.linalg.norm(h - new_h) \
                    + np.linalg.norm(D - new_D) \
                    + abs(a - new_a) \
                    + abs(b - new_b)
            stop = stop_new

            if i % self.args.print_n_iter == 0:
                print("\niteration : ", i, 
                    "\n lambda  ", lambd,
                    "\n stop : ", stop,
                    "\n a-new_a", abs(a - new_a),
                    "\n b-new_b", abs(b - new_b),
                    "\n h-new_h", np.linalg.norm(h - new_h),
                    "\n D-new_D", np.linalg.norm(D - new_D),
                    "\n a est", new_a,
                    "\n b est", new_b)
                
            list_iter.append(i)
            list_errors.append(np.linalg.norm(D - new_D))
            if stop < self.args.stop_criteria :
                gv.reussi = True
                print("last iteration :", i) 
                break

            a, b, h, D = new_a, new_b, new_h, new_D

        fig1 = plt.figure()
        plt.plot(list_iter, list_errors)
        plt.xlabel("Itérations")
        plt.ylabel("norm (D-new_D)")
        plt.yscale('log')
        plt.title('Convergence of D')
        self.args.plots.append(fig1)
        self.args.plot_names.append("Convergence")

        h_args = gaussian_kernel(D + self.args.epsD * np.eye(3), self.args.kernel_size, args=self.args)
        observ_slice(h_args, 0, "h_args_est", self.args)
        observ_slice(h, 0, "h_est", self.args)
        if self.args.simulation_PSF == True:
            h = gaussian_kernel_true([0.0, 0.0, 0.0], gv.D_true, self.args.kernel_size, args=self.args)
            observ_slice(h, 0, "h_true", self.args)
        observ_slice(fftconvolve(h_args, x, "same"), 0, "Convolution h_args_est with x", self.args)
        observ_slice(fftconvolve(h, x, "same"), 0, "Convolution h_est with x", self.args)
        observ_profile_convolution(a + b * fftconvolve(h_args, x, "same"), y, "Convolution h_args_est with x", self.args)
        observ_profile_kernel(h_args, "h_args_est", self.args)
        observ_profile_convolution(a + b * fftconvolve(h, x, "same"), y, "Convolution h_est with x", self.args)
        observ_profile_kernel(h, "h_est", self.args)

        return a, b, h, D, h_args

    def proj_box(self, x, val_min, val_max):
        return (val_min <= x <= val_max) * x + (x < val_min) * val_min + (x > val_max) * val_max

    def update_a(self, y, b, convo):
        N = convo.shape[0] * convo.shape[1] * convo.shape[2]
        return self.proj_box(1 / N * np.sum(y - b * convo), self.args.a_min, self.args.a_max)
    
    def update_b(self, y, a, convo):
        return self.proj_box(np.sum(np.multiply(y, convo) - a * convo) / (np.sum(convo ** 2)), self.args.b_min, self.args.b_max)
    
    def update_D(self, h,  D, Omega, eps, lambd):
        gam = self.args.gam_D 
        Omega = Omega.reshape(Omega.shape[0] * Omega.shape[1] * Omega.shape[2], 3, 1)
        hflat = h.flatten()
        m = 1 / 2 * lambd * gam * np.sum(hflat)
        S = 1 / 2 * lambd * gam * np.einsum('a, aij, akj -> ik', hflat, Omega, Omega)
        w, V = np.linalg.eigh(D - S)
        d = np.maximum(w - eps + np.sqrt((w + eps) ** 2 + 4 * m), 0)
        return 1 / 2 * V @ (d.reshape(d.shape[0], 1) * V.T)

    def update_h(self, y, X_T, a, b, s, h,  D, convo, Omega, eps, gam_h, lambd):
        alph = gam_h
        gam = gam_h
        rho = 1 / (gam * lambd)
        grad = X_T(b ** 2 * convo + b * a - b * y) * (a + b * convo)
        forward = h - alph * grad
        c = coef_c( D, Omega, eps)
        return self.proxg(c, forward, lambd, gam, rho, s)

    def w(self, nu, c, h, rho, s):
        return -1 - c + rho * (h - nu) + np.log(s)

    def proxg(self, c, h, lambd, gam, rho, s):
        nu = self.nu_hat(c, h, lambd, gam, rho, s)
        nu = nu[-1]
        W = self.compute_W_exp(nu, c, h, lambd, gam, rho, s)
        return (1 / rho) * W

    def nu_hat(self, c, h, lambd, gam, rho, s):
        """
        Newton to get the value nu_hat
        """
        epsilon = 1e-6
        maxIter = 10000
        nIter = 0
        nu = [0]
        mycnt = 0
        while True:
            mycnt += 1
            nIter += 1
            fi, dfi = self.phi_and_dphi(nu[-1], c, h, lambd, gam, rho, s)
            if math.isinf(dfi):
                print("isinf !!!!!")
                break

            new_nu = nu[-1] - fi / dfi
            nu.append(new_nu)
            if np.abs(new_nu) > 1e50 or np.isnan(nu[-1]) or new_nu < -1e50:
                new_nu = (np.random.rand(1) - 0.5) * 10
                nu[-1] = new_nu
                print("max k, min k", np.max(h), np.min(h))
                print("max c, min c", np.max(c), np.min(c))
                raise Exception('nu is nan :(')
            #print("Newton : ", np.abs(nu[-1] - nu[-2]))
            if nIter > maxIter or np.abs(nu[-1] - nu[-2]) < epsilon:
                break
        return nu

    def phi_and_dphi(self, nu, c, h, lambd, gam, rho, s):
        W = self.compute_W_exp(nu, c, h, lambd, gam, rho, s)
        delta = self.args.resolution[0] * self.args.resolution[1] * self.args.resolution[2]
        return (1 / rho) * np.sum(W) - 1, - np.sum(1 - 1 / (1 + W))

    def Lambert_W(self, v):
        w_matrix = np.ones(v.shape)
        u = np.inf * w_matrix
        n_iter = 0
        while np.sum(np.abs((w_matrix - u) / np.maximum(w_matrix, 1e-5)) > 1e-07) > 0 and n_iter < 100:
            u = np.copy(w_matrix)
            e = np.exp(w_matrix)
            f = w_matrix * e - v
            w_matrix = w_matrix - f / (e * (w_matrix + 1) - f * (w_matrix + 2) / (2 * w_matrix + 2))
            n_iter += 1
        return w_matrix

    def compute_W_exp(self, nu, c, h, lambd, gam, rho, s):
        limit = 200
        w_hat = self.w(nu, c, h, rho, s)
        tau = w_hat + np.log(rho)
        W = tau * (1 - np.log(tau + 1e-6) / (1 + tau))
        W[tau < limit] = self.Lambert_W(np.exp(tau[tau < limit]))
        return W
    

