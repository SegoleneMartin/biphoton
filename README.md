# PSF Estimation and Image Restoration for biphoton microscopy

## Overview
This project provides tools for Point Spread Function (PSF) estimation and image restoration. The biphoton images are assumed to be composed of 2 channels: a first channel with the image of the sample (e.g., muscle of size $n_x \times n_y \times n_z$), and a second channel with the image of $1 \mu m$-diameter micro-beads (of size $n_x \times n_y \times n_z$).

The codebase includes:
- a **PSF restoration** method named GENTLE which estimates the PSF given the bead image. The estimated PSF corresponds to a blur kernel with Gaussian prior shape.
- a **noise parameters estimation method**, to estimate the parameters of the underlying heteroscedastic noise.
- a **restoration method** named P-MMS, witch deconvolves the sample with the estimated PSF and denoises given the estimated noise levels

## Project Structure
```
|-- main.py                # Main script to execute the project
|-- utils.py               # Utility functions
|-- config/                # Configuration files for PSF estimation, restoration
|   |-- config_PSF.yaml
|   |-- config_restoration.yaml
|   |-- main_config.yaml
|-- PSF_estimation/        # PSF estimation module
|-- restoration/           # Image restoration module
|-- images/mouse_muscle    # Raw 2-channels image, one with beads, the other one with the sample
|-- crops/mouse_muscle     # Extracted bead crops saved as images
|-- results/mouse_muscle   # PSF and restoration results
```

## Usage
To run the main script:
```bash
python main.py
```
By default, the script will use configurations specified in `config/main_config.yaml`. Modify these files as needed to customize the parameters.

## Configuration
The YAML files in the `config/` directory store various settings:
- `config_PSF.yaml`: Settings related to PSF estimation.
- `config_restoration.yaml`: Parameters for image restoration methods.
- `main_config.yaml`: General configurations for the pipeline.

Modify these files before running the main script to adjust behavior.

## Dependencies
Ensure the following Python libraries are installed:
- numpy
- scipy
- matplotlib
- yaml

If additional dependencies are required, install them using pip.