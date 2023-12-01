# File to execute to run the pipeline
import argparse
import os
from utils import load_cfg_from_cfg_file, merge_cfg_from_list
from PSF_estimation.methods.GENTLE import GENTLE
from PSF_estimation.methods.NLS import NLS
from PSF_estimation.make_crops import make_crops
from PSF_estimation.mean_over_beads import compute_mean_over_bead

def parse_args():
    # Get the config keys from config files in ./config
    parser = argparse.ArgumentParser(description='Main')
    cfg = load_cfg_from_cfg_file('config/main_config.yaml')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    config_PSF = 'config/config_PSF.yaml'
    config_restoration= 'config/config_restoration.yaml'
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
            os.makedirs("results/" + args.image_folder_name + '/' + args.method_PSF + '/')
        except:
            pass
        # Enumerate the files in folder path_crops
        files = os.listdir(path_crops)
        # Remove the files that are not .npy files
        files = [file for file in files if file.endswith('.npy')]
        for name_crop in files:
            print('======> Processing crop ' + name_crop[0] + ' with method ' + args.method_PSF + '...')
            algo.pipeline(path_crops, name_crop)
            print('======> Done processing crop ' + name_crop[0] + ' !\n')

        # Compute the mean over all the beads crops
        compute_mean_over_bead(args)



        
    else:
        print('Skipping PSF estimation')

if __name__ == "__main__":
    main()