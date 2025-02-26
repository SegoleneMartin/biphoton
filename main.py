# File to execute to run the pipeline
import argparse
import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator
from mayavi import mlab

from utils import load_cfg_from_cfg_file, merge_cfg_from_list
from PSF_estimation.methods.GENTLE import GENTLE
from PSF_estimation.methods.NLS import NLS
from PSF_estimation.make_crops import make_crops
from PSF_estimation.mean_over_beads import compute_mean_over_bead
from restoration.methods.Penalized_MMS_heteroscedastic import P_MMS
from restoration.tools.noise_estimation import estimate_params
from restoration.operators.PSF_operator import generate_blur


def parse_args():
    # Get the config keys from config files in ./config
    parser = argparse.ArgumentParser(description='Main')
    cfg = load_cfg_from_cfg_file('config/main_config.yaml')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    config_PSF = 'config/config_PSF.yaml'
    config_restoration = 'config/config_restoration.yaml'
    cfg.update(load_cfg_from_cfg_file(config_PSF))
    cfg.update(load_cfg_from_cfg_file(config_restoration))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main():
    args = parse_args()

    # Execute the PSF estimation pipeline if estimate_PSF is True
    if args.estimate_PSF:
        print('Starting PSF estimation')

        # If crop folder does not exist, create it
        try:
            os.mkdir('crops/')
        except:
            pass
        # If crops are not already created, create them
        try:
            os.mkdir('crops/' + args.image_folder_name + '/')
        except:
            pass
        # If the folder is not empty, skip the crop creation
        if len(os.listdir('crops/' + args.image_folder_name + '/')) == 0:
            make_crops(args.image_folder_name)

        # Import the PSF estimation method
        if args.method_PSF == 'GENTLE':
            algo = GENTLE(args)
        elif args.method_PSF == 'NLS':
            algo = NLS(args)
        else:
            raise ValueError('Unknown PSF estimation method')

        path_crops = 'crops/' + args.image_folder_name + '/'
        # create result folder
        try:
            os.makedirs("results/" + args.image_folder_name +
                        '/' + args.method_PSF + '/')
        except:
            pass
        # Enumerate the files in folder path_crops
        files = os.listdir(path_crops)
        # Remove the files that are not .npy files
        files = [file for file in files if file.endswith('.npy')]
        for name_crop in files:
            print('======> Processing crop ' +
                  name_crop[0] + ' with method ' + args.method_PSF + '...')
            algo.pipeline(path_crops, name_crop)
            print('======> Done processing crop ' + name_crop[0] + ' !\n')

        # Compute the mean over all the beads crops
        compute_mean_over_bead(args)

    else:
        print('Skipping PSF estimation')

    # Execute the restoration pipeline if restore is True
    if args.restore_image:

        # Load the degraded image y and normalize it
        print('Loading image')
        y_bar = io.imread(os.path.join(
            'images', args.image_folder_name, args.image_name))[:, 0, :, :].T
        y_max = y_bar.max()
        y_bar = y_bar / y_max
        y = y_bar[args.n1_min:args.n1_max,
                  args.n2_min:args.n2_max, args.n3_min:args.n3_max]
        plt.figure()
        plt.imshow(y[:, :, 10], cmap='gray')
        plt.title('Selected crop to restore (2D slice)')
        plt.show(block=False)

        mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))
        src = mlab.pipeline.scalar_field(y)
        mlab.pipeline.volume(src, vmin=y.min(), vmax=y.max())
        mlab.title('Selected crop to restore (3D view)',
                   size=0.2, color=(0, 0, 0))

        mlab.show()

        continue_ = input(
            'Do you want to continue and restore this crop (size = {} x {} x {})? ([y]/n)'.format(y.shape[0], y.shape[1], y.shape[2]))
        if continue_ != 'y':
            return
        else:
            plt.close()

        # Estimate noise parameters
        print("Estimating noise parameters")
        if args.estimate_noise:
            a, b, _, _ = estimate_params(y_bar, plot_regression=True)
            print('Estimated noise parameters: a = {}, b = {}'.format(a, b))
            print('If the linear fit is not satisfactory, the noise model might be wrong. Please adjust the parameters in the config file and set estimate_noise to False.')
        else:
            a = args.a
            b = args.b
            print('Using pre-set noise parameters: a = {}, b = {}'.format(a, b))

        # Load the PSF estimation results and define operator H
        print('Loading PSF')
        # if a path is specified in the config file, load the PSF from the specified path
        if args.path_to_psf is not None:
            h = np.load(args.path_to_psf)
        # otherwise, load the mean PSF from the results folder
        else:
            try:
                path_to_psf = os.path.join('results', args.image_folder_name,
                                           args.method_PSF, 'mean', 'lambd_' + str(args.selected_lambda), 'values', 'h_args_est.npy')
                h = np.load(path_to_psf)[:, :, :-1]
            except:
                raise ValueError(
                    'No PSF found at location {}. Please run the PSF estimation pipeline first.'.format(path_to_psf))
        N = y.shape[0] * y.shape[1] * y.shape[2]
        H, H_T = generate_blur(h)
        H = LinearOperator((N, N), matvec=H, rmatvec=H_T)

        # Import the restoration method
        print('Starting restoration')
        if args.method_restoration == 'P_MMS':

            P_MMS(y=y,
                  H=H,
                  delta=args.delta,
                  eta=args.eta,
                  x_0=np.copy(y),
                  x_bar=np.copy(y),
                  x_min=0,
                  x_max=1,
                  res=args.res,
                  noise_params=(a, b),
                  max_iter=1000,
                  tolerance=2*1e-05)

        else:
            raise ValueError('Unknown restoration method')


if __name__ == "__main__":
    main()
