# For each method (GENTLE and NLS):
# - compute the mean of the estimated values over all the beads crops
# - save the mean values in the folder results/image_folder_name/method/mean/values/
# - save the plots in the folder results/image_folder_name/method/mean/plots/
# - save the results.txt file in the folder results/image_folder_name/method/mean/

import numpy as np
import os
from PSF_estimation.utils import compute_angles, compute_variances
from .kernel import gaussian_kernel, gaussian_kernel_true, genC
from .observation3D import observ_slice, observ_profile_kernel
import datetime


def compute_mean_over_bead(args):

    path_to_results = 'results/' + args.image_folder_name + '/' + args.method_PSF + '/'

    if args.method_PSF == 'NLS':

        # For each folder in results/image_folder_name/method/ load the values in values/
        folders = os.listdir(path_to_results)
        list_angles_phi = []
        list_angles_theta = []
        list_variances = []

        for folder in [f for f in folders if not f.startswith('.')]:
            # Load the values of D
            D = np.load(path_to_results + folder +
                        '/values/D_est.npy', allow_pickle=True)
            L, U = np.linalg.eigh(D)
            angles = compute_angles(U[:, 0])
            variances = compute_variances(L, U)
            list_angles_phi.append(angles[0])
            list_angles_theta.append(angles[1])
            list_variances.append(variances)

        # Turn to array and compute the mean
        list_angles_phi = np.array(list_angles_phi)
        list_angles_theta = np.array(list_angles_theta)
        list_variances = np.array(list_variances)
        mean_angles_phi = np.mean(list_angles_phi)
        mean_angles_theta = np.mean(list_angles_theta)
        mean_variances = np.mean(list_variances, axis=0)

        # Given the mean values, compute the mean D and the associated mean h_args
        mean_angles = np.array([mean_angles_phi, mean_angles_theta, 0.0])
        mean_D = genC(mean_angles, mean_variances)
        mean_h_args = gaussian_kernel(mean_D, args.kernel_size, args=args)

        # Create the folders results/image_folder_name/method/mean/ and its subfolders
        try:
            os.makedirs(path_to_results + 'mean/' + args.method_PSF + '/')
        except:
            pass
        try:
            os.makedirs(path_to_results + 'mean/' + args.method_PSF + '/')
        except:
            pass
        try:
            os.makedirs(path_to_results + 'mean/' + args.method_PSF + '/')
        except:
            pass

        # Save the mean values in results/image_folder_name/method/mean/values/
        np.save(path_to_results + 'mean/' + args.method_PSF +
                '/' + 'values/' + 'D_est' + '.npy', mean_D)
        np.save(path_to_results + 'mean/' + args.method_PSF + '/' +
                'values/' + 'h_args_est' + '.npy', mean_h_args)

        # Save the plots in results/image_folder_name/method/mean/plots/
        args.plots = []
        args.plot_names = []
        observ_slice(mean_h_args, 0, "h_args_est", args)
        observ_profile_kernel(mean_h_args, "h_args_est", args)
        for i in range(len(args.plots)):
            fig = args.plots[i]
            fig.savefig(path_to_results + 'mean/' + args.method_PSF +
                        '/' + 'plots/' + args.plot_names[i] + ".png")

        # Save the results.txt file in results/image_folder_name/method/mean/
        txt = f"date = {datetime.datetime.now()}" \
            f"\nimage folder name = {args.image_folder_name}" \
            f"\nresolution = {args.resolution} " \
            f"\n\nESTIMATED VALUES" \
            f"\nmean_angle_phi_est = {mean_angles_phi}" \
            f"\nmean_angle_theta_est = {mean_angles_theta}" \
            f"\nmean_variances_est = {mean_variances}"
        results_file = open(path_to_results + 'mean/' +
                            args.method_PSF + '/' + "results.txt", "w")
        results_file.write(txt)
        results_file.close()

    elif args.method_PSF == 'GENTLE':  # do the same but for each value of lambda
        folders = os.listdir(path_to_results)

        for lambd in args.lambdas:

            # For each folder in results/image_folder_name/method/ load the values in values/
            list_angles_phi = []
            list_angles_theta = []
            list_variances = []
            list_h = []

            for folder in folders:
                # Load the values of D
                D = np.load(path_to_results + folder + '/' + 'lambd_' +
                            str(lambd) + '/values/D_est.npy', allow_pickle=True)
                h = np.load(path_to_results + folder + '/' + 'lambd_' +
                            str(lambd) + '/values/h_est.npy', allow_pickle=True)
                L, U = np.linalg.eigh(D)
                angles = compute_angles(U[:, 0])
                variances = compute_variances(L, U)
                list_angles_phi.append(angles[0])
                list_angles_theta.append(angles[1])
                list_variances.append(variances)
                list_h.append(h)

            # Turn to array and compute the mean
            d1 = max([h.shape[0] for h in list_h])
            d2 = max([h.shape[1] for h in list_h])
            d3 = max([h.shape[2] for h in list_h])
            mean_h = np.zeros((d1, d2, d3))
            for h in list_h:  # center h so that its center coincides with the center of mean_h
                n1 = h.shape[0]
                n2 = h.shape[1]
                n3 = h.shape[2]
                mean_h[int((d1-n1)/2):int((d1+n1)/2), int((d2-n2)/2)
                           :int((d2+n2)/2), int((d3-n3)/2):int((d3+n3)/2)] += h
            mean_h /= len(list_h)

            list_angles_phi = np.array(list_angles_phi)
            list_angles_theta = np.array(list_angles_theta)
            list_variances = np.array(list_variances)
            mean_angles_phi = np.mean(list_angles_phi)
            mean_angles_theta = np.mean(list_angles_theta)
            mean_variances = np.mean(list_variances, axis=0)

            # Given the mean values, compute the mean D and the associated mean h_args
            mean_angles = np.array([mean_angles_phi, mean_angles_theta, 0.0])
            mean_D = genC(mean_angles, mean_variances)
            mean_h_args = gaussian_kernel(mean_D, args.kernel_size, args=args)

            # Create the folders results/image_folder_name/method/mean/ and its subfolders
            try:
                os.makedirs(path_to_results + 'mean/' +
                            'lambd_' + str(lambd) + '/')
            except:
                pass
            try:
                os.makedirs(path_to_results + 'mean/' +
                            'lambd_' + str(lambd) + '/' + 'values/')
            except:
                pass
            try:
                os.makedirs(path_to_results + 'mean/' +
                            'lambd_' + str(lambd) + '/' + 'plots/')
            except:
                pass

            # Save the mean values in results/image_folder_name/method/mean/values/
            np.save(path_to_results + 'mean/' + 'lambd_' +
                    str(lambd) + '/' + 'values/' + 'D_est' + '.npy', mean_D)
            np.save(path_to_results + 'mean/' + 'lambd_' + str(lambd) +
                    '/' + 'values/' + 'h_args_est' + '.npy', mean_h_args)
            np.save(path_to_results + 'mean/' + 'lambd_' +
                    str(lambd) + '/' + 'values/' + 'h_est' + '.npy', mean_h)

            # Save the plots in results/image_folder_name/method/mean/plots/
            args.plots = []
            args.plot_names = []
            observ_slice(mean_h_args, 0, "h_args_est", args)
            observ_profile_kernel(mean_h_args, "h_args_est", args)
            observ_slice(mean_h, 0, "h_est", args)
            observ_profile_kernel(mean_h, "h_est", args)
            for i in range(len(args.plots)):
                fig = args.plots[i]
                fig.savefig(path_to_results + 'mean/' + 'lambd_' +
                            str(lambd) + '/' + 'plots/' + args.plot_names[i] + ".png")

            # Save the results.txt file in results/image_folder_name/method/mean/
            txt = f"date = {datetime.datetime.now()}" \
                f"\nimage folder name = {args.image_folder_name}" \
                f"\nresolution = {args.resolution} " \
                f"\n\nESTIMATED VALUES" \
                f"\nmean_angle_phi_est = {mean_angles_phi}" \
                f"\nmean_angle_theta_est = {mean_angles_theta}" \
                f"\nmean_variances_est = {mean_variances}"
            results_file = open(path_to_results + 'mean/' +
                                'lambd_' + str(lambd) + '/' + "results.txt", "w")
            results_file.write(txt)
            results_file.close()
