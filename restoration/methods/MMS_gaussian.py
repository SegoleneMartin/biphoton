#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
========================================================================
Created on Tue Aug 31 2021

@author: segolene.martin@centralesupelec.fr
Inspired from the Matlab pipeline of Emilie Chouzenoux.

PENALIZED LOCAL MAJORATION-MINIMIZATION SUBSPACE ALGORITHM (Gaussian case)

This module implements a penalized local majoration-minimization subspace
algorithm for image restoration under Gaussian noise. The algorithm minimizes
a cost function combining a regularization term (computed from image gradients)
and penalty terms enforcing pixel range and data-fit constraints.
========================================================================
"""

import time
import numpy as np
from scipy.linalg import norm, pinv
from scipy.sparse.linalg import LinearOperator

from tools.PSNR import PSNR
from operators.gradient_operator import generate_G


def P_MMS_gaussian(y, H, delta, lambd, x_0, x_bar, x_min, x_max, res, noise_params, D_scale, max_iter, tolerance):
    """
    Performs the Penalized Local Majoration-Minimization Subspace (P-MMS) algorithm
    under a Gaussian noise model.

    The algorithm minimizes the functional

        F(x) = lambd * sum(phi(sqrt(Gv(x)^2 + Gh(x)^2 + Gt(x)^2)))
               + ||H x - y||^2
               subject to:  R1(x) = ||x - proj_box(x, x_min, x_max)||^2 = 0

    where phi is a regularization function chosen among:
        (1) phi(u) = 1 - exp(-u^2/(2*delta^2))
        (2) phi(u) = u^2/(2*delta^2 + u^2)
        (3) phi(u) = log(1 + u^2/(delta^2))
        (4) phi(u) = sqrt(1 + u^2/delta^2) - 1
        (5) phi(u) = 1/2 * u^2
        (6) phi(u) = 1 - exp(-((sqrt(1 + u^2/delta^2)-1)))

    Parameters
    ----------
    y : numpy.ndarray, shape (n1, n2, n3)
        Degraded (observed) image.
    H : LinearOperator, shape (N, N) with N = n1*n2*n3
        Blur (forward) operator.
    delta : float
        Regularization parameter for numerical stability.
    lambd : float
        Bound on the data-fit constraint; used to weight the regularization term.
    x_0 : numpy.ndarray, shape (n1, n2, n3)
        Initial estimate of the image.
    x_bar : numpy.ndarray, shape (n1, n2, n3)
        Ground truth image (used for computing the error/PSNR).
    x_min : float
        Minimum allowed pixel value.
    x_max : float
        Maximum allowed pixel value.
    res : tuple of 3 floats
        Spatial resolution of the image.
    noise_params : tuple
        Tuple containing parameters for the noise model (not explicitly used).
    D_scale : float
        Scaling factor applied to the data-fit term.
    max_iter : int
        Maximum number of iterations for the outer loop.
    tolerance : float
        Convergence tolerance for the stopping criterion.

    Returns
    -------
    x : numpy.ndarray, shape (n1, n2, n3)
        Restored image (last iterate produced by the algorithm).
    """
    # Define the gradient operator based on the image resolution.
    n1, n2, n3 = x_0.shape
    N = n1 * n2 * n3
    G = generate_G((n1, n2, n3), res)

    # Flatten images to 1D arrays.
    y = y.reshape(N)
    x_0 = x_0.reshape(N)
    x_bar = x_bar.reshape(N)

    # Initialize algorithm variables.
    criterion = 1e7
    counter = 1
    a1 = 1.1
    a2 = 1.1
    e1 = 1.2
    e2 = 1.2
    GAMMA_1 = a1 ** e1     # Penalty parameter for the pixel-range constraint.
    GAMMA_2 = a2 ** e2     # Penalty parameter for the data-fit constraint.
    GAMMA_2_old = GAMMA_2 / 2
    GAMMA_2_old_old = GAMMA_2 / 4

    GAMMA = np.array([GAMMA_1, GAMMA_2])
    EPSILON_0 = 1
    EPSILON = EPSILON_0 / (GAMMA_2 ** 1.5)
    start_time = time.time()

    # First iteration: MM gradient descent step.
    d = grad_F(x_0, y, lambd, H, G, delta, x_min, x_max, GAMMA, D_scale)
    Ad = (lambd * G.T @ (A(G @ x_0, delta, N) @ (G @ d)) +
          2 * H.T @ (H @ d) +
          GAMMA[0] * 2 * d)  # Curvature of the majorant of F_GAMMA at x_0, applied to d.
    Bd = - d.T @ Ad   # Curvature of the majorant at 0.
    u = - (d.T @ d) / Bd
    step = - u * d
    x = x_0 + step
    x_old = np.copy(x_0)
    Phi_x_old = Phi(G @ x, G, delta)
    Psi_x_old = np.sum((H @ x - y) ** 2)

    # Outer loop: solve a sequence of subproblems P_GAMMA with precision EPSILON.
    while counter < max_iter and criterion > tolerance and (time.time() - start_time) < 7600:
        if counter >= 3:
            x_0 = x + (x - x_old) * (1 / GAMMA_2 - 1 / GAMMA_2_old) / \
                (1 / GAMMA_2_old - 1 / GAMMA_2_old_old)
        else:
            x_0 = np.copy(x)

        print(EPSILON, GAMMA)
        x_inner_new, x_inner, criterion = MMS(
            y, lambd, H, G, delta, x_0, x_old, x_bar, x_min, x_max, D_scale,
            max_iter=1000, EPSILON=EPSILON, GAMMA=GAMMA, shape=(n1, n2, n3))

        # Update iterates.
        if counter < 2:
            x_old = x_inner
            x = x_inner_new
        else:
            x_old = np.copy(x)
            x = x_inner_new

        Phi_x = Phi(G @ x, G, delta)
        Psi_x = np.sum((H @ x - y) ** 2)
        criterion = abs((Phi_x - Phi_x_old) / Phi_x) + \
            R1(x, x_min, x_max) + abs((Psi_x - Psi_x_old) / Psi_x)
        Phi_x_old = Phi_x
        Psi_x_old = Psi_x

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
            EPSILON = EPSILON_0 / (GAMMA_2 ** 0.25)

    return x.reshape((n1, n2, n3))


def MMS(y, lambd, H, G, delta, x, x_old, x_bar, x_min, x_max, D_scale, max_iter, EPSILON, GAMMA, shape):
    """
    Solves the subproblem P_GAMMA via a Majoration-Minimization Subspace (MMS) algorithm.

    The subproblem minimizes the penalized functional

        F_GAMMA(x) = lambd * Phi(Gx, delta) + GAMMA[0] * R1(x) + ||H x - y||^2

    where:
        - Phi(Gx, delta) = sum( sqrt(delta + ||gradient||^2) ),
        - R1(x) = ||x - proj_box(x, x_min, x_max)||^2,
        - The data-fit term is ||H x - y||^2 (with bound lambd used in the penalty curvature).

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        Flattened degraded image.
    lambd : float
        Data-fit bound (weight for the regularization term).
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
        Ground truth image (for PSNR computation).
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
        Original shape of the image (n1, n2, n3).

    Returns
    -------
    x : numpy.ndarray, shape (N,)
        Last iterate produced by the MMS algorithm.
    x_prev_step : numpy.ndarray, shape (N,)
        The update from the previous step (used for extrapolation).
    criterion : float
        Final value of the convergence criterion.
    """
    N = x.shape[0]
    step = x - x_old
    criterion = 1e7  # Initial high value to enter the loop.
    counter = 0

    # MMS algorithm on F_GAMMA.
    while counter < max_iter and criterion > EPSILON:
        grad_F_x, Hx, Gx = grad_F(
            x, y, lambd, H, G, delta, x_min, x_max, GAMMA, D_scale, return_products=True)
        criterion = norm(grad_F_x)
        D = np.vstack((-grad_F_x, step)).T

        if counter == 0:
            grad_F_x_old = grad_F(x_old, y, lambd, H, G,
                                  delta, x_min, x_max, GAMMA, D_scale)
            m_H = - (H @ grad_F_x_old)
            o_H = (H @ step)
            m_G = - G @ grad_F_x_old
            o_G = G @ step
            m_I = - grad_F_x_old
            o_I = step
        else:
            o_H = m_H * u[0] + o_H * u[1]
            o_G = m_G * u[0] + o_G * u[1]
            o_I = m_I * u[0] + o_I * u[1]

        m_H_new = - (H @ grad_F_x)
        m_G_new = - G @ grad_F_x
        m_I_new = - grad_F_x

        tmp_H = np.vstack((m_H_new, o_H)).T
        tmp_G = np.vstack((m_G_new, o_G)).T
        tmp_I = np.vstack((m_I_new, o_I)).T
        B_tmp = lambd * tmp_G.T @ (A(Gx, delta, N) @
                                   tmp_G) + 2 * tmp_H.T @ tmp_H
        B = B_tmp + 2 * GAMMA[0] * tmp_I.T @ tmp_I
        Dgrad = D.T @ grad_F_x
        u = - pinv(B) @ Dgrad
        step = D @ u

        x = x + step
        m_H, m_G, m_I = m_H_new, m_G_new, m_I_new

        if counter % 20 == 0:
            print("iter = {}, error = {}, PSNR = {:4.2f}, Psi = {}, R1 = {}"
                  .format(counter, criterion, PSNR(x_bar, x), Phi(Gx, G, delta), R1(x, x_min, x_max)))

        counter += 1

    return (x, x - step, criterion)


def F(x, y, lambd, H, G, delta, x_min, x_max, GAMMA, D_scale):
    """
    Computes the total objective function

        F(x) = lambd * Phi(Gx, delta) + ||H x - y||^2 + GAMMA[0] * R1(x)

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        Current image (flattened).
    y : numpy.ndarray, shape (N,)
        Degraded image.
    lambd : float
        Weight for the regularization term.
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
        The value of the objective function.
    """
    Hx = H @ x
    Gx = G @ x
    return lambd * Phi(Gx, G, delta) + np.sum((Hx - y) ** 2) + GAMMA[0] * R1(x, x_min, x_max)


def grad_F(x, y, lambd, H, G, delta, x_min, x_max, GAMMA, D_scale, return_products=False):
    """
    Computes the gradient of the penalized objective function

        F_GAMMA(x) = lambd * Phi(Gx, delta) + ||H x - y||^2 + GAMMA[0] * R1(x)

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        Point at which the gradient is computed.
    y : numpy.ndarray, shape (N,)
        Degraded image.
    lambd : float
        Weight for the regularization term.
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
        The gradient of F_GAMMA at x.
    Hx : numpy.ndarray, shape (N,), optional
        The product H @ x (if return_products is True).
    Gx : numpy.ndarray, shape (N,), optional
        The product G @ x (if return_products is True).
    """
    Hx = H @ x
    Gx = G @ x

    grad = (lambd * grad_Phi(Gx, G, delta) +
            grad_Psi(Hx, H, y) +
            GAMMA[0] * grad_R1(x, x_min, x_max))
    if not return_products:
        return grad
    else:
        return (grad, Hx, Gx)


def Phi(Gx, G, delta):
    """
    Computes the regularization term Phi based on the image gradient.

    Phi(Gx, delta) = sum( sqrt(delta + ||gradient||^2) )

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
        Squared Euclidean distance from x to its projection onto [x_min, x_max].
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
        Projection of x onto the ball B(y, r).
    """
    diff = x - y
    n_diff = np.sqrt(diff @ diff)
    if n_diff > r:
        return y + diff * r / n_diff
    else:
        return x


def grad_Psi(Hx, H, y):
    """
    Computes the gradient of the data-fit term Psi.

    Psi(Hx) = ||H x - y||^2, so that

        grad_Psi(Hx, H, y) = 2 * H.T @ (H x - y)

    Parameters
    ----------
    Hx : numpy.ndarray
        The product H @ x.
    H : LinearOperator, shape (N, N)
        Blur operator.
    y : numpy.ndarray
        Observed image.

    Returns
    -------
    numpy.ndarray
        The gradient of the data-fit term.
    """
    return 2 * (H.T @ (Hx - y))


def test_in_box(x, x_min, x_max):
    """
    Tests whether each element of x is within the interval [x_min, x_max].

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
        Array of values indicating if each element is in the box.
    """
    eps = 0.0000
    return 1 - 1 * (x <= x_max + eps) * (x >= x_min - eps)


def A(Gx, delta, N):
    """
    Returns a LinearOperator that approximates the local curvature (Hessian)
    of the regularization term Phi at the current point.

    The operator acts on a vector u as:
        A(Gx, delta, N)(u) = 1/sqrt(delta + ||gradient||^2) * u,
    computed element-wise.

    Parameters
    ----------
    Gx : numpy.ndarray, shape (3*N,)
        The product G @ x (flattened gradient).
    delta : float
        Regularization parameter.
    N : int
        Number of pixels in the original image (n1*n2*n3).

    Returns
    -------
    LinearOperator
        A linear operator of shape (3*N, 3*N) approximating the Hessian of Phi.
    """
    Gx = Gx.reshape(N, 3)

    def Ax_u(u):
        C = 1 / np.sqrt(delta + np.sum(Gx**2, axis=1))
        curvature_Phi = (C[:, np.newaxis] * u.reshape(N, 3)).reshape(3 * N)
        return curvature_Phi

    return LinearOperator((3 * N, 3 * N), matvec=Ax_u, rmatvec=Ax_u)
