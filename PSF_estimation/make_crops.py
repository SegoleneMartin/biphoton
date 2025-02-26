from skimage import io as skio
from skimage import measure
from scipy.signal import fftconvolve
import numpy as np
import os
# import module global_variables
import PSF_estimation.global_variables as gv
import os
import matplotlib.pyplot as plt


def make_crops(image_folder_name):

    path_ims = 'images/' + image_folder_name + '/'
    path_crops = 'crops/' + image_folder_name + '/'
    dirs = os.listdir(path_ims)
    sizes = []

    for imname in dirs:

        print("opening image : ", imname)
        im = skio.imread(path_ims + imname)
        im = im[:, 1, :, :].T
        im = im / np.max(im) * 0.3

        print("Image shape", im.shape)
        filter = np.ones((3, 3, 3))
        filter = filter / np.sum(filter)
        imfiltered = fftconvolve(im, filter, "same")
        seuil = 0.03
        imfiltered[imfiltered < seuil] = 0
        imfiltered[imfiltered > seuil] = 1
        labels, n_regions = measure.label(imfiltered, return_num=True)

        regions = []
        regions_size = []
        print("Nombre de pré-régions : ", n_regions)
        size_min = 1000

        for i in range(1, n_regions + 1):
            size = np.sum(np.where(labels == i, 1, 0))
            print(i, size)
            if size > size_min:
                regions.append(i)
                regions_size.append(size)

        print("Nombre de régions sélectionnées :", len(regions))
        for i in range(1, len(regions) + 1):
            selected_region = regions[i - 1]
            locs = np.argwhere(labels == selected_region)
            xmin, ymin, zmin = np.min(locs, axis=0)
            xmax, ymax, zmax = np.max(locs, axis=0)

            padding = 20
            if (xmax - xmin) % 2 == 0:
                xmax += padding + 1
                xmin -= padding
            else:
                xmax += padding
                xmin -= padding
            if (ymax - ymin) % 2 == 0:
                ymax += padding + 1
                ymin -= padding
            else:
                ymax += padding
                ymin -= padding
            if (zmax - zmin) % 2 == 0:
                zmax += padding + 1
                zmin -= padding
            else:
                zmax += padding
                zmin -= padding

            if zmax > im.shape[2]:
                zmax = im.shape[2] - 1

            im_croped = im[max(xmin, 0):xmax, max(
                ymin, 0):ymax, max(zmin, 0):zmax]
            print(im_croped.shape)

            # plot the croped image and ask the user whether he wants to save it
            fig = plt.subplots(1, 2, figsize=(15, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(im_croped[:, :, im_croped.shape[2] // 2], cmap='gray')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.subplot(1, 2, 2)
            plt.imshow(im_croped[:, im_croped.shape[1] // 2, :], cmap='gray')
            plt.xlabel("X")
            plt.ylabel("Z")
            plt.show(block=False)
            save = input("Do you want to save this croped image? (y/n)")
            if save == 'y':
                print("Image saved")
            else:
                print("Image not saved")
                plt.close()
                continue
            plt.close()
            size = im_croped.flatten().shape[0]
            sizes.append(size)

            try:
                print(path_crops + str(i) + ".npy")
                np.save(path_crops + str(i) + ".npy", im_croped)
                skio.imsave(path_crops + str(i) + ".tif", im_croped)
            except:
                pass
