#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
========================================================================
Created on Tue Aug 31 2021

@author: segolene.martin@centralesupelec.fr
Inspired from the Matlab pipeline of Emilie Chouzenoux.

PENALIZED LOCAL MAJORATION-MINIMIZATION SUBSPACE ALGORITHM

This module implements a penalized local majoration-minimization subspace algorithm 
for image restoration. The algorithm minimizes a cost function combining a 
regularization term (computed from image gradients) and penalty terms enforcing 
pixel range and data-fit constraints.
========================================================================
"""

import time
import numpy as np
import os
from scipy.linalg import norm, pinv
from scipy.sparse.linalg import LinearOperator
from scipy.fft import fftn, ifftn, fftshift, ifftshift
import tifffile

from restoration.tools.PSNR import PSNR
from restoration.operators.gradient_operator import generate_G


def P_MMS(y, H, delta, eta, x_0, x_bar, x_min, x_max, res, noise_params, max_iter, tolerance, args):
    """
    Performs the Penalized Local Majoration-Minimization Subspace (P-MMS) algorithm.

    The algorithm minimizes the functional:

        F(x) = sum( phi( sqrt( Gv(x)^2 + Gh(x)^2 + Gt(x)^2 ) ) )
               subject to:
                   R1(x) = ||x - proj_box(x, x_min, x_max)||^2 = 0   (pixel range constraint)
                   R2(x) = ||H x - y||^2 <= alpha                     (data-fit constraint)

    Parameters
    ----------
    y : numpy.ndarray, shape (n1, n2, n3)
        Degraded (observed) image.
    H : LinearOperator, shape (N, N) with N = n1*n2*n3
        Blur (forward) operator.
    delta : float
        Regularization parameter added inside the square root for stability.
    eta : float
        Scaling parameter used to compute the data-fit bound (alpha = sqrt(eta * N)).
    x_0 : numpy.ndarray, shape (n1, n2, n3)
        Initial estimate of the image.
    x_bar : numpy.ndarray, shape (n1, n2, n3)
        Ground truth image (used for computing the SNR during iterations).
    x_min : float
        Minimum allowed pixel value.
    x_max : float
        Maximum allowed pixel value.
    res : tuple of 3 floats
        Spatial resolution of the image.
    noise_params : tuple of two floats
        Parameters (a, b) characterizing the noise model (not directly used in the code below).
    max_iter : int
        Maximum number of iterations for the outer loop.
    tolerance : float
        Convergence tolerance for the stopping criterion.

    Returns
    -------
    x : numpy.ndarray, shape (n1, n2, n3)
        Restored image (last iterate produced by the algorithm).
    list_SNR : list of floats
        List of SNR values computed at each iteration.
    """
    # save raw image y as a .tif file
    path_to_save = os.path.join('results', args.image_folder_name,
                                args.method_restoration, 'crop_x{}-{}_y{}-{}'.format(args.n1_min, args.n1_max, args.n2_min, args.n2_max), 'delta_{}'.format(delta), 'eta_{}'.format(eta))
    try:
        os.makedirs(path_to_save)
    except:
        pass
    tifffile.imsave(os.path.join(
        path_to_save, 'degraded_image.tif'), y.transpose(2, 0, 1))

    # Define the gradient operator given the resolution.
    n1, n2, n3 = x_0.shape
    N = n1 * n2 * n3
    G = generate_G((n1, n2, n3), res)

    # Define D_scale as a diagonal matrix (Scaling factor applied to the data (used in the data-fit penalty).)
    s = 3
    kernel = np.ones((s, s, s))
    kernel = kernel / kernel.sum()
    a1, a2, a3 = (n1 - s) // 2, (n2 - s) // 2, (n3 - s) // 2
    kernel_full = np.pad(
        kernel, ((a1, a1), (a2, a2), (a3, a3)), mode="constant")

    fft_kernel = fftn(ifftshift(kernel_full, axes=(0, 1, 2)), axes=(0, 1, 2))
    fft_y = fftn(ifftshift(y, axes=(0, 1, 2)), axes=(0, 1, 2))
    y_conv = np.real(
        fftshift(ifftn(fft_y * fft_kernel, axes=(0, 1, 2)), axes=(0, 1, 2)))
    D_scale = np.ones(N) / np.sqrt(a * abs(y_conv.reshape(N)) + b)

    N = x.shape[0]
    step = x - x_old
    criterion = 1e7  # Initial high value to enter the loop.
    counter = 0

    # Flatten images to 1D arrays.
    y = y.reshape(N)
    x_0 = x_0.reshape(N)
    x_bar = x_bar.reshape(N)

    # Define noise parameters (a, b) and compute the data-fit bound.
    a, b = noise_params
    alpha = np.sqrt(eta * N)

    # Initialize algorithm variables.
    criterion = 1e7
    counter = 1
    a1 = 2
    a2 = 2
    e1 = 2.0
    e2 = 2.0
    GAMMA_1 = a1 ** e1      # Penalty parameter for the pixel-range constraint.
    GAMMA_2 = a2 ** e2      # Penalty parameter for the data-fit constraint.
    GAMMA_2_old = GAMMA_2 / 2
    GAMMA_2_old_old = GAMMA_2 / 4

    GAMMA = np.array([GAMMA_1, GAMMA_2])
    EPSILON_0 = 100000
    EPSILON = EPSILON_0 / (GAMMA_2 ** 0.75)
    start_time = time.time()
    list_SNR = [compute_SNR(x_bar, x_0)]

    # First iteration: MM gradient descent step.
    d = grad_F(x_0, y, alpha, H, G, delta, x_min, x_max, GAMMA, D_scale)
    Ad = (G.T @ (A(G @ x_0, delta, N) @ (G @ d))
          + GAMMA[1] * 2 * H.T @ (D_scale * (D_scale * (H @ d)))
          + GAMMA[0] * 2 * d)  # Curvature of the majorant of F_GAMMA at x_0, applied to d.
    Bd = - d.T @ Ad    # Curvature of the majorant at 0.
    u = - (d.T @ d) / Bd
    step = - u * d
    x = x_0 + step
    x_old = np.copy(x_0)
    Phi_x_old = Phi(G @ x, G, delta)

    # Outer loop: solve a sequence of subproblems P_GAMMA with precision EPSILON.
    while counter < max_iter and criterion > tolerance and (time.time() - start_time) < 7600:
        if counter >= 3:
            x_0 = x + (x - x_old) * (1 / GAMMA_2 - 1 / GAMMA_2_old) / \
                (1 / GAMMA_2_old - 1 / GAMMA_2_old_old)
        else:
            x_0 = np.copy(x)

        print(EPSILON, GAMMA)
        x_inner_new, x_inner, criterion, list_SNR = MMS(
            y, alpha, H, G, delta, x_0, x_old, x_bar, x_min, x_max, D_scale,
            max_iter=1000, EPSILON=EPSILON, GAMMA=GAMMA, shape=(n1, n2, n3), list_SNR=list_SNR)

        # Update iterates.
        if counter < 2:
            x_old = x_inner
            x = x_inner_new
        else:
            x_old = np.copy(x)
            x = x_inner_new

        list_SNR.append(compute_SNR(x_bar, x))
        Phi_x = Phi(G @ x, G, delta)
        criterion = abs((Phi_x - Phi_x_old) / Phi_x) + \
            R1(x, x_min, x_max) + R2(H @ x, y, alpha, D_scale)
        Phi_x_old = Phi_x

        print('ITER = {}, GAMMA = {}, EPS = {:5f}, criterion = {}'.format(
            counter, GAMMA, EPSILON, criterion))
        print('--------------------\n')

        counter += 1

        # Update penalty parameters.
        GAMMA_2_old_old = GAMMA_2_old
        GAMMA_2_old = GAMMA_2
        GAMMA_1 = min((a1 * counter) ** e1, 3e10)
        GAMMA_2 = min((a2 * counter) ** e2, 5e10)
        GAMMA = np.array([GAMMA_1, GAMMA_2])
        if GAMMA_2 == 500000:
            EPSILON = EPSILON / 1.25
        else:
            EPSILON = EPSILON_0 / (GAMMA_2 ** 0.75)

        # save the restored image as a numpy array and .tif files every 100 iterations
        if counter % 100 == 0:
            np.save(os.path.join(path_to_save, 'restored_image_iter{}.npy'.format(
                counter)), x.reshape((n1, n2, n3)))
            tifffile.imsave(os.path.join(path_to_save, 'restored_image_iter{}.tif'.format(
                counter)), x.reshape((n1, n2, n3)).transpose(2, 0, 1))
    return (x.reshape((n1, n2, n3)), list_SNR)


def MMS(y, alpha, H, G, delta, x, x_old, x_bar, x_min, x_max, D_scale, max_iter, EPSILON, GAMMA, shape, list_SNR=None):
    """
    Solves the subproblem P_GAMMA via a Majoration-Minimization Subspace (MMS) algorithm.

    The subproblem minimizes the penalized functional:

        F_GAMMA(x) = Phi(Gx, delta) + GAMMA[0] * R1(x) + GAMMA[1] * R2(Hx, y, alpha, D_scale)

    where
        Phi(Gx, delta) = sum( sqrt(delta + ||gradient||^2) ),
        R1(x) = ||x - proj_box(x, x_min, x_max)||^2,
        R2(Hx, y, alpha, D_scale) = ||D_scale * Hx - proj_ball(D_scale * Hx, D_scale * y, alpha)||^2.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        Flattened degraded image.
    alpha : float
        Data-fit bound (computed as sqrt(eta * N) in P_MMS).
    H : LinearOperator, shape (N, N)
        Blur operator.
    G : LinearOperator, shape (N, 3*N)
        Gradient operator.
    delta : float
        Regularization parameter.
    x : numpy.ndarray, shape (N,)
        Current iterate (initial point for this subproblem).
    x_old : numpy.ndarray, shape (N,)
        Previous iterate.
    x_bar : numpy.ndarray, shape (N,)
        Ground truth image (for SNR computation).
    x_min : float
        Minimum pixel value.
    x_max : float
        Maximum pixel value.
    D_scale : float
        Scaling factor for the data-fit term.
    max_iter : int
        Maximum number of iterations for the MMS loop.
    EPSILON : float
        Tolerance on the norm of the gradient for convergence.
    GAMMA : numpy.ndarray, shape (2,)
        Penalty parameters [GAMMA_1, GAMMA_2].
    shape : tuple
        Original shape of the image (n1, n2, n3) for reshaping if needed.
    list_SNR : list, optional
        List of SNR values (updated during iterations).

    Returns
    -------
    x : numpy.ndarray, shape (N,)
        Last iterate generated by the MMS algorithm.
    x_prev_step : numpy.ndarray, shape (N,)
        The update from the previous step (used for extrapolation).
    criterion : float
        Final value of the convergence criterion (norm of gradient).
    list_SNR : list
        Updated list of SNR values.
    """
    N = x.shape[0]
    step = x - x_old
    criterion = 1e7  # Initial high value to enter the loop.
    counter = 0

    # MMS algorithm on F_GAMMA.
    while counter < max_iter and criterion > EPSILON:
        grad_F_x, Hx, Gx = grad_F(
            x, y, alpha, H, G, delta, x_min, x_max, GAMMA, D_scale, return_products=True)
        criterion = norm(grad_F_x)
        D = np.vstack((-grad_F_x, step)).T

        if counter == 0:
            grad_F_x_old = grad_F(x_old, y, alpha, H, G,
                                  delta, x_min, x_max, GAMMA, D_scale)
            m_H = - D_scale * (H @ grad_F_x_old)
            o_H = D_scale * (H @ step)
            m_G = - G @ grad_F_x_old
            o_G = G @ step
            m_I = - grad_F_x_old
            o_I = step
        else:
            o_H = m_H * u[0] + o_H * u[1]
            o_G = m_G * u[0] + o_G * u[1]
            o_I = m_I * u[0] + o_I * u[1]

        m_H_new = - D_scale * (H @ grad_F_x)
        m_G_new = - G @ grad_F_x
        m_I_new = - grad_F_x

        tmp_H = np.vstack((m_H_new, o_H)).T
        tmp_G = np.vstack((m_G_new, o_G)).T
        tmp_I = np.vstack((m_I_new, o_I)).T

        B_tmp = tmp_G.T @ (A(Gx, delta, N) @ tmp_G) + \
            2 * GAMMA[1] * tmp_H.T @ tmp_H
        B = B_tmp + 2 * GAMMA[0] * tmp_I.T @ tmp_I
        Dgrad = D.T @ grad_F_x
        u = - pinv(B) @ Dgrad
        step = D @ u

        x = x + step
        m_H, m_G, m_I = m_H_new, m_G_new, m_I_new

        print("iter = {}, error = {}, PSNR = {:4.2f}, Psi = {}, R1 = {}, R2 = {}"
              .format(counter, criterion, PSNR(x_bar, x), Phi(Gx, G, delta),
                      R1(x, x_min, x_max), R2(Hx, y, alpha, D_scale)))
        counter += 1

    return (x, x - step, criterion, list_SNR)


def F(x, y, alpha, H, G, delta, x_min, x_max, GAMMA, D_scale):
    """
    Computes the total objective function F_GAMMA(x).

    F_GAMMA(x) = Phi(Gx, delta) + GAMMA[0]*R1(x) + GAMMA[1]*R2(Hx, y, alpha, D_scale)

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        Current image (flattened).
    y : numpy.ndarray, shape (N,)
        Degraded image.
    alpha : float
        Data-fit bound.
    H : LinearOperator, shape (N, N)
        Blur operator.
    G : LinearOperator, shape (N, 3*N)
        Gradient operator.
    delta : float
        Regularization parameter.
    x_min : float
        Minimum pixel value.
    x_max : float
        Maximum pixel value.
    GAMMA : numpy.ndarray, shape (2,)
        Penalty parameters.
    D_scale : float
        Scaling factor for the data-fit term.

    Returns
    -------
    float
        The value of the objective function at x.
    """
    Hx = H @ x
    Gx = G @ x
    return (Phi(Gx, G, delta) + GAMMA[0]*R1(x, x_min, x_max) + GAMMA[1]*R2(Hx, y, alpha, D_scale))


def grad_F(x, y, alpha, H, G, delta, x_min, x_max, GAMMA, D_scale, return_products=False):
    """
    Computes the gradient of the penalized objective function F_GAMMA(x).

    The function is defined as:
        F_GAMMA(x) = Phi(Gx, delta) + GAMMA[0]*R1(x) + GAMMA[1]*R2(Hx, y, alpha, D_scale)

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        Point at which the gradient is computed.
    y : numpy.ndarray, shape (N,)
        Degraded image.
    alpha : float
        Data-fit bound.
    H : LinearOperator, shape (N, N)
        Blur operator.
    G : LinearOperator, shape (N, 3*N)
        Gradient operator.
    delta : float
        Regularization parameter.
    x_min : float
        Minimum pixel value.
    x_max : float
        Maximum pixel value.
    GAMMA : numpy.ndarray, shape (2,)
        Penalty parameters.
    D_scale : float
        Scaling factor for the data-fit term.
    return_products : bool, optional
        If True, also return H @ x and G @ x to avoid redundant computations.

    Returns
    -------
    grad : numpy.ndarray, shape (N,)
        Gradient of F_GAMMA at x.
    Hx : numpy.ndarray, shape (N,), optional
        The product H @ x (if return_products is True).
    Gx : numpy.ndarray, shape (N,), optional
        The product G @ x (if return_products is True).
    """
    Hx = H @ x
    Gx = G @ x

    grad = (grad_Phi(Gx, G, delta) +
            GAMMA[0] * grad_R1(x, x_min, x_max) +
            GAMMA[1] * grad_R2(Hx, y, alpha, H, D_scale))
    if not return_products:
        return grad
    else:
        return (grad, Hx, Gx)


def Phi(Gx, G, delta):
    """
    Computes the regularization term Phi based on the image gradient.

    Phi(Gx, delta) = sum( sqrt(delta + ||gradient||^2) ),
    where the gradient is computed from Gx.

    Parameters
    ----------
    Gx : numpy.ndarray, shape (3*N,)
        Flattened array containing gradient components.
    G : LinearOperator, shape (N, 3*N)
        Gradient operator.
    delta : float
        Regularization parameter for numerical stability.

    Returns
    -------
    float
        The value of the regularization term.
    """
    N = Gx.shape[0] // 3
    Gx = Gx.reshape(N, 3)
    return np.sum(np.sqrt(delta + np.sum(Gx**2, axis=1)))


def grad_Phi(Gx, G, delta):
    """
    Computes the gradient of Phi with respect to x.

    Parameters
    ----------
    Gx : numpy.ndarray, shape (3*N,)
        Flattened array containing gradient components.
    G : LinearOperator, shape (N, 3*N)
        Gradient operator.
    delta : float
        Regularization parameter.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The gradient of Phi at x.
    """
    N = Gx.shape[0] // 3
    Gx = Gx.reshape(N, 3)
    z = (Gx / np.sqrt(delta + np.sum(Gx**2, axis=1))
         [:, np.newaxis]).reshape(3 * N)
    return G.T @ z


def proj_box(x, x_min, x_max):
    """
    Projects the vector x onto the box [x_min, x_max] element-wise.

    Parameters
    ----------
    x : numpy.ndarray
        Input vector.
    x_min : float
        Minimum allowed value.
    x_max : float
        Maximum allowed value.

    Returns
    -------
    numpy.ndarray
        The projection of x onto [x_min, x_max].
    """
    return (x * (x_min <= x) * (x <= x_max) +
            x_max * (x > x_max) +
            x_min * (x < x_min))


def R1(x, x_min, x_max):
    """
    Computes the squared distance of x to the box [x_min, x_max].

    Parameters
    ----------
    x : numpy.ndarray
        Input vector.
    x_min : float
        Minimum allowed value.
    x_max : float
        Maximum allowed value.

    Returns
    -------
    float
        The squared Euclidean distance from x to its projection onto [x_min, x_max].
    """
    return norm(x - proj_box(x, x_min, x_max)) ** 2


def grad_R1(x, x_min, x_max):
    """
    Computes the gradient of R1 with respect to x.

    Parameters
    ----------
    x : numpy.ndarray
        Input vector.
    x_min : float
        Minimum allowed value.
    x_max : float
        Maximum allowed value.

    Returns
    -------
    numpy.ndarray
        The gradient of R1 at x.
    """
    return 2 * (x - proj_box(x, x_min, x_max))


def proj_ball(x, y, r):
    """
    Projects x onto the Euclidean ball centered at y with radius r.

    Parameters
    ----------
    x : numpy.ndarray
        Point to be projected.
    y : numpy.ndarray
        Center of the ball.
    r : float
        Radius of the ball.

    Returns
    -------
    numpy.ndarray
        The projection of x onto the ball B(y, r).
    """
    diff = x - y
    n_diff = np.linalg.norm(diff)
    if n_diff > r:
        return y + diff * r / n_diff
    else:
        return x


def R2(Hx, y, alpha, D_scale):
    """
    Computes the squared distance of the scaled blurred image to a ball.

    Specifically, R2(Hx, y, alpha, D_scale) = ||D_scale * Hx - proj_ball(D_scale * Hx, D_scale * y, alpha)||^2

    Parameters
    ----------
    Hx : numpy.ndarray
        The blurred image (H @ x).
    y : numpy.ndarray
        The observed image.
    alpha : float
        Radius of the ball (data-fit bound).
    D_scale : float
        Scaling factor applied to Hx and y.

    Returns
    -------
    float
        The squared distance used in the data-fit penalty.
    """
    proj = proj_ball(D_scale * Hx, D_scale * y, alpha)
    return norm(D_scale * Hx - proj) ** 2


def grad_R2(Hx, y, alpha, H, D_scale):
    """
    Computes the gradient of R2 with respect to x.

    Parameters
    ----------
    Hx : numpy.ndarray
        The blurred image (H @ x).
    y : numpy.ndarray
        The observed image.
    alpha : float
        Data-fit bound (radius of the ball).
    H : LinearOperator, shape (N, N)
        Blur operator.
    D_scale : float
        Scaling factor applied to the data.

    Returns
    -------
    numpy.ndarray
        The gradient of R2 with respect to x.
    """
    proj = proj_ball(D_scale * Hx, D_scale * y, alpha)
    return 2 * (H.T @ (D_scale * (D_scale * Hx - proj)))


def test_in_box(x, x_min, x_max):
    """
    Tests whether each element of x lies within the interval [x_min, x_max].

    Parameters
    ----------
    x : numpy.ndarray
        Input vector.
    x_min : float
        Minimum allowed value.
    x_max : float
        Maximum allowed value.

    Returns
    -------
    numpy.ndarray
        An array of 0s and 1s indicating whether each element of x is within the box.
    """
    eps = 0.0000
    return 1 - 1 * (x <= x_max + eps) * (x >= x_min - eps)


def A(Gx, delta, N):
    """
    Returns a LinearOperator representing the local curvature (Hessian) 
    of the regularization term Phi at the current point.

    The operator is defined via its action on a vector u as:
        A(Gx, delta, N)(u) = curvature_Phi * u,
    where the curvature is computed element-wise as 1/sqrt(delta + ||gradient||^2).

    Parameters
    ----------
    Gx : numpy.ndarray, shape (3*N,)
        The product G @ x.
    delta : float
        Regularization parameter.
    N : int
        Number of pixels in the original image (n1*n2*n3).

    Returns
    -------
    LinearOperator
        A linear operator of shape (3*N, 3*N) that approximates the Hessian of Phi.
    """
    Gx = Gx.reshape(N, 3)

    def Ax_u(u):
        C = 1 / np.sqrt(delta + np.sum(Gx**2, axis=1))
        curvature_Phi = (C[:, np.newaxis] * u.reshape(N, 3)).reshape(3 * N)
        return curvature_Phi

    return LinearOperator((3 * N, 3 * N), matvec=Ax_u, rmatvec=Ax_u)


def compute_SNR(x, y):
    """
    Computes the Signal-to-Noise Ratio (SNR) in decibels.

    Parameters
    ----------
    x : numpy.ndarray
        Ground truth image.
    y : numpy.ndarray
        Noisy (or restored) image.

    Returns
    -------
    float
        SNR in dB. Returns infinity if the noise power is zero.
    """
    # Calculate the signal and noise power.
    signal_power = np.var(x)
    noise_power = np.var(y - x)

    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)
