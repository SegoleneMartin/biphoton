import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "sans-serif",
    "contour.linewidth":6.0,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'ytick.labelsize': 17,
    'xtick.labelsize': 17,
    'figure.max_open_warning': 0
})

def observ_slice(values, center, title, args):
    fig1 = plt.figure()
    plt.imshow(values[:, :, values.shape[2] // 2 + center].T, 
                extent=[0, values.shape[0]*args.resolution[0], 0, values.shape[1]*args.resolution[1]])
    plt.xlabel("X ($\mu m$)")
    plt.ylabel("Y ($\mu m$)")
    plt.colorbar()
    args.plots.append(fig1)
    args.plot_names.append(title + " - plan XY")

    fig2 = plt.figure()
    plt.imshow(values[:, values.shape[1] // 2 + center, :].T, 
                extent=[0, values.shape[0]*args.resolution[0], 0, values.shape[2]*args.resolution[2]])
    plt.xlabel("X ($\mu m$)")
    plt.ylabel("Z ($\mu m$)")
    args.plots.append(fig2)
    args.plot_names.append(title + " - plan XZ")

    fig3 = plt.figure()
    plt.imshow(values[values.shape[0] // 2 + center, :, :].T, 
                extent=[0, values.shape[1]*args.resolution[1], 0, values.shape[2]*args.resolution[2]])
    plt.xlabel("Y ($\mu m$)")
    plt.ylabel("Z ($\mu m$)")
    plt.colorbar()
    args.plots.append(fig3)
    args.plot_names.append(title + " - plan YZ")


def observ_profile_convolution(convo, y, title, args):
    res = args.resolution
    # Compute profiles of y
    profile_y_X = np.sum(y, axis=(1, 2)) * res[1] * res[2]
    profile_y_Y = np.sum(y, axis=(0, 2)) * res[0] * res[2]
    profile_y_Z = np.sum(y, axis=(0, 1)) * res[0] * res[1]

    # Compute profiles of convo
    profile_convo_X = np.sum(convo, axis=(1, 2)) * res[1] * res[2]
    profile_convo_Y = np.sum(convo, axis=(0, 2)) * res[0] * res[2]
    profile_convo_Z = np.sum(convo, axis=(0, 1)) * res[0] * res[1]

    # Plot profiles
    n_pixel_X = y.shape[0]
    n_pixel_Y = y.shape[1]
    n_pixel_Z = y.shape[2]

    grid_X = np.arange(n_pixel_X) * res[0]
    grid_Y = np.arange(n_pixel_Y) * res[1]
    grid_Z = np.arange(n_pixel_Z) * res[2]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(grid_X, profile_y_X, label=r"$\boldsymbol{y}$", color='red', linewidth=3.5)
    ax[0].plot(grid_X, profile_convo_X, label=r"$\hat{\alpha} + \hat{\beta}\hat{\boldsymbol{h}}\ast \boldsymbol{x}  $", color='blue', linewidth=3.5, linestyle='--')
    ax[0].set_xlabel(r"Width ($\mu m$)", fontsize=20)

    ax[1].plot(grid_Y, profile_y_Y, label=r"$\boldsymbol{y}$", color='red', linewidth=3.5)
    ax[1].plot(grid_Y, profile_convo_Y, label=r"$\hat{\alpha} + \hat{\beta}\hat{\boldsymbol{h}}\ast\boldsymbol{x}  $",  color='blue', linewidth=3.5, linestyle='--')
    ax[1].set_xlabel(r"Width ($\mu m$)", fontsize=20)

    ax[2].plot(grid_Z, profile_y_Z, label=r"$\boldsymbol{y}$", color='red', linewidth=3.5)
    ax[2].plot(grid_Z, profile_convo_Z, label=r"$\hat{\alpha} + \hat{\beta}\hat{\boldsymbol{h}}\ast \boldsymbol{x} $",  color='blue', linewidth=3.5, linestyle='--')
    ax[2].set_xlabel(r"Width ($\mu m$)", fontsize=20)

    for j in range(3):
        if j == 2:
            major_ticks = np.arange(-0, 13, 1)
            minor_ticks = np.arange(-0, 13, 0.5)
        else:
            major_ticks = np.arange(-0, 4.5, 0.5)
            minor_ticks = np.arange(-0, 4.5, 0.25)

        ax[j].set_xticks(major_ticks)
        ax[j].set_xticks(minor_ticks, minor=True)
        ax[j].ticklabel_format(axis='both', style='sci', scilimits=(-5, 6) )
        ax[j].grid(which='minor', alpha=0.2)
        ax[j].grid(which='major', alpha=0.5)

        if j == 2:
            ax[j].set_xlim(0, 12)
        else:
            ax[j].set_xlim(0, 4)

    labels = [r"$\boldsymbol{y}$", r"$\hat{\alpha} + \hat{\beta}\hat{\boldsymbol{h}}\ast \boldsymbol{x}$"]
    #fig.tight_layout() 
    fig.subplots_adjust(bottom=0.30)  
    fig.legend(labels=labels, loc="lower center", ncol=2, fontsize=24)

    args.plots.append(fig)
    args.plot_names.append(title + " - profiles")


def observ_profile_kernel(h, title, args):
    res = args.resolution

    # Compute profiles of convo
    profile_convo_X = np.sum(h, axis=(1, 2)) * res[1] * res[2]
    profile_convo_Y = np.sum(h, axis=(0, 2)) * res[0] * res[2]
    profile_convo_Z = np.sum(h, axis=(0, 1)) * res[0] * res[1]

    # Plot profiles
    n_pixel_X = h.shape[0]
    n_pixel_Y = h.shape[1]
    n_pixel_Z = h.shape[2]

    grid_X = np.arange(n_pixel_X) * res[0]
    grid_Y = np.arange(n_pixel_Y) * res[1]
    grid_Z = np.arange(n_pixel_Z) * res[2]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(grid_X, profile_convo_X, label=r"$\hat{\boldsymbol{h}} $", color='blue', linewidth=3.5, linestyle='--')
    ax[0].set_xlabel(r"Width ($\mu m$)", fontsize=20)

    ax[1].plot(grid_Y, profile_convo_Y, label=r"$\hat{\boldsymbol{h}} $",  color='blue', linewidth=3.5, linestyle='--')
    ax[1].set_xlabel(r"Width ($\mu m$)", fontsize=20)

    ax[2].plot(grid_Z, profile_convo_Z, label=r"$\hat{\boldsymbol{h}} $",  color='blue', linewidth=3.5, linestyle='--')
    ax[2].set_xlabel(r"Width ($\mu m$)", fontsize=20)

    for j in range(3):
        if j == 2:
            major_ticks = np.arange(-0, 13, 1)
            minor_ticks = np.arange(-0, 13, 0.5)
        else:
            major_ticks = np.arange(-0, 4.5, 0.5)
            minor_ticks = np.arange(-0, 4.5, 0.25)

        ax[j].set_xticks(major_ticks)
        ax[j].set_xticks(minor_ticks, minor=True)
        ax[j].ticklabel_format(axis='both', style='sci', scilimits=(-5, 6) )
        ax[j].grid(which='minor', alpha=0.2)
        ax[j].grid(which='major', alpha=0.5)

        if j == 2:
            ax[j].set_xlim(0, 12)
        else:
            ax[j].set_xlim(0, 4)

    labels = [r"$\hat{\boldsymbol{h}} $"]
    #fig.tight_layout() 
    fig.subplots_adjust(bottom=0.30)  
    fig.legend(labels=labels, loc="lower center", ncol=1, fontsize=24)

    args.plots.append(fig)
    args.plot_names.append(title + " - profiles")