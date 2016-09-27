# Copyright (c) 2015-2016 by Stermedia Sp. z o.o.
"""
SDWFCM.py : Spatial Distance Weighted Fuzzy C-Means clustering algorithm.
 Adopted from https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/skfuzzy/cluster/_cmeans.py
 Copyright (c) 2012, the scikit-fuzzy team. All rights reserved.
 Modified according to: Guo, Liu, Wu, Hong and Zhang. A New Spatial Fuzzy C-Means for Spatial
  Clustering. WSEAS Transactions on Computers 14:369-381, 2015.

Please cite:
[1] J.Czakon, P.Giedziun, M.Blach, G.Zurek, P.Krajewski, W.Dyrka. An ensemble algorithm for the nuclei segmentation
in the histological images. MICCAI 2015 - Computational Brain Tumor Cluster of Events, Munich, 9.10.2015

Part of this work was conducted under the Support Programme of the Partnership between Higher Education
and Science and Business Activity Sector financed by City of Wroclaw.

Functions:
    neighbours - Return list of neighbouring pixels.
    sdfcm_cmeans - Spatial Distance Weighted Fuzzy C-Means clustering algorithm.
"""

import itertools

import numpy as np
from scipy.spatial.distance import cdist


def neighbours(im, x, y, d=1):
    """Return list of neighbouring pixels.

    :param im: image
    :param x: horizontal coordinate
    :param y: vertical coordinate
    :param d: neighbourhood
    :return: a tuple (list of values of neighbouring pixels, list of coordinates of neighbouring pixels)
    """
    w, h = tuple(im.shape)

    x_min = x - d if x > d - 1 else x
    y_min = y - d if y > d - 1 else y
    x_max = x + d if x < w - d else x
    y_max = y + d if y < h - d else y

    n = []
    nlist = list(itertools.product(range(x_min, x_max + 1), range(y_min, y_max + 1)))
    nlist.remove((x, y))

    for (ix, iy) in nlist:
        n.append(im[ix, iy])

    return (n, nlist)


def _sdfcm_cmeans0(data, u_old, c, m, w, h,
                   l, number_neighbours):
    """
    Adopted from scikit-fuzzy. Copyright (c) 2012, the scikit-fuzzy team. All rights reserved.
    Modified according to Guo, Liu, Wu, Hong and Zhang. WSEAS Transactions on Computers 14:369-381, 2015.
    """
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / (np.ones((data.shape[1],
                                    1)).dot(np.atleast_2d(um.sum(axis=1))).T)

    d = _distance(data, cntr)
    d = np.fmax(d, np.finfo(np.float64).eps)

    d_reshaped = np.ndarray.reshape(d, (c, w, h))

    f = np.zeros((c, w, h))

    for (ix, iy) in itertools.product(range(0, w), range(0, h)):

        d_neighbours_sum_by_clust = np.zeros((c, 1))

        for i in range(c):
            d_neighbours = neighbours(d_reshaped[i, :, :], ix, iy, d=number_neighbours)[0]
            d_neighb_sum = np.sum(d_neighbours)
            d_neighbours_sum_by_clust[i] = d_neighb_sum
            f[i, ix, iy] = d_neighb_sum

        d_neighbours_sum_min = np.min(d_neighbours_sum_by_clust)

        f[:, ix, iy] /= d_neighbours_sum_min

    f = np.reshape(f, (c, w * h))

    d = (1 - l) * np.multiply(d, f) + l * d

    jm = (um * d ** 2).sum()

    u = d ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return cntr, u, jm, d


def _distance(data, centers):
    """
    Calcuate Euclidean distance from each point to each cluster center,
    returning results in matrix form.
    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.
    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.
    See Also
    --------
    scipy.spatial.distance.cdist

    Adopted from scikit-fuzzy. Copyright (c) 2012, the scikit-fuzzy team. All rights reserved.
    """
    return cdist(data, centers).T


def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix u. Measures 'fuzziness' in partitioned clustering.
    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.
    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.

    Adopted from scikit-fuzzy. Copyright (c) 2012, the scikit-fuzzy team. All rights reserved.
    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)


def sdfcm_cmeans(data, c, m, w, h, l, number_neighbours,
                 error, maxiter, init=None, seed=None):
    """
    Fuzzy c-means clustering algorithm [1]_.
    Parameters
    ----------
    data : 2d array, size (S, N)
        Data to be clustered.  N is the number of data sets; S is the number
        of features within each sample vector.
    c : int
        Desired number of clusters or classes.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    w : int
        Image width.
    h : int
        Image height.
    l : float
        Lamba weight coefficient - see Guo et al.2015.
    number_neighbours : int
        Size of the spatial neighbourhood (see neighbours()).
    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter : int
        Maximum number of iterations allowed.
    init : 2d array, size (S, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.
    Returns
    -------
    cntr : 2d array, size (S, c)
        Cluster centers.  Data for each center along each feature provided
        for every cluster (of the `c` requested clusters).
    u : 2d array, (S, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (S, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (S, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.
    Notes
    -----
    The algorithm implemented is from Ross et al. [1]_.
    References
    ----------
    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.
           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.

    Adopted from scikit-fuzzy. Copyright (c) 2012, the scikit-fuzzy team. All rights reserved.
    Modified according to Guo, Liu, Wu, Hong and Zhang. WSEAS Transactions on Computers 14:369-381, 2015.
    """

    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 /= np.ones(
            (c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.empty(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [cntr, u, Jjm, d] = _sdfcm_cmeans0(data, u2, c, m, w, h,
                                           l, number_neighbours)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return cntr, u, u0, d, jm, p, fpc
