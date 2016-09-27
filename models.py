# Copyright (c) 2016 by Stermedia Sp. z o.o.
"""
This module provides functions for running 3d KMeans, GMM, SDWFCM and DICT on PET images.

Please cite:
[1] J.Czakon, F.Drapejkowski, G.Zurek, P.Giedziun, J.Zebrowski, W.Dyrka. Machine learning methods for accurate
delineation of tumors in PET images. MICCAI 2016 - PETSEG challenge, Athens, 21.10.2016

Part of this work was conducted under the Support Programme of the Partnership between Higher Education
and Science and Business Activity Sector financed by City of Wroclaw.

Functions:
    desharp_pre - Blur a little bit a 3d PET image if appears too sharp.
    desharp_post - Remove small objects in segmented image if original PET image appeared too sharp.
    select_tumor - Select labels which correspond to a tumor.
    kmeans3d - Run the 3d KMeans algorithm over the 3d PET image.
    gmm3d - Run the 3d Gaussian Mixture Model over the 3d PET image.
    fcm3d - Run the 3d Spatial Distance Weighted Fuzzy C-Means algorithm over the 3d PET image.
    dict3d - Run the 3d Dictionary-based labeling algorithm over the 3d PET image.
"""

import pickle

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_opening, binary_closing
from sklearn.cluster import KMeans
from sklearn.mixture import GMM

from backend.SDWFCM3 import sdfcm_cmeans3
from backend.dict_model3 import dict_label_image3

# 3d Laplaian of Gaussian for assessing image sharpness
L = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
              [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
              [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])


def desharp_pre(pet):
    """Blur a little bit a 3d PET image if appears too sharp.

    The goal is to get rid of grainy noise if present.

    :param pet: 3d PET image.
    :return the 3d PET image, slightly blurred if necessary.
    """
    result = pet.copy()
    prc = np.percentile(convolve(result, L, mode='constant', cval=0.0), 99.9)
    if prc > 1.0:
        result = gaussian_filter(result, 1)

    return result


def select_tumor(pet, image_clustered, k, first):
    """Select labels which correspond to a tumor.

    The job is done based on maximum intensity in the original image.

    :param pet: 3d PET image.
    :param image_clustered: clustered 3d PET image (labels).
    :param k: total number of labels.
    :param first: number of the most intensive labels assumed to correspond to the tumor.
    :return 3d binary mask of the tumor.
    """

    tumori = np.ones(first) * -1
    tumora = np.zeros(first)
    for i in range(k):
        for j in range(first):
            if tumora[j] < np.mean(pet[image_clustered == i]):
                tumori = np.insert(tumori, j, i)
                tumora = np.insert(tumora, j, np.mean(pet[image_clustered == i]))
                break

    tumor_image = np.zeros(image_clustered.shape)
    for j in range(first):
        tumor_image += (image_clustered == tumori[j]) * 1

    return tumor_image


def desharp_post(pet, tumor_image):
    """Remove small objects in segmented image if original PET image appeared too sharp.

    The goal is to get rid of grainy noise if present.

    :param pet: 3d PET image.
    :param tumor_image: 3d binary mask of the tumor.
    :return 3d binary mask of the tumor, slightly cleaned if necessary.
    """

    result = tumor_image.copy()
    prc = np.percentile(convolve(pet, L, mode='constant', cval=0.0), 99.9)
    if prc > 1.0:
        result = binary_opening(result, iterations=2)
        result = binary_closing(result, iterations=2)

    return result


def kmeans3d(pet, **kwargs):
    """Run the 3d KMeans algorithm over the 3d PET image.

    This algorithm is based on image intensity only. If the input image appears to sharp, grainy noisy is assumed
     and some blurring is applied, followed by morphological opening and closing applied to the output binary mask.

    :param pet: 3d PET image.

    Kwargs:
        'k': number of clusters.
        'first': number of selected clusters.

    :return 3d binary mask of the tumor.
    """

    k = kwargs.get('k', 2)
    first = kwargs.get('first', 1)

    pet = desharp_pre(pet)

    z, w, h = tuple(pet.shape)
    image_array = np.reshape(pet, (z * w * h, 1))

    km = KMeans(n_clusters=k, max_iter=100).fit(image_array)
    labels_k = km.predict(image_array)

    image_clustered = np.reshape(labels_k, (z, w, h))

    tumor_image = select_tumor(pet, image_clustered, k, first)
    tumor_image = desharp_post(pet, tumor_image)

    return tumor_image


def gmm3d(pet, **kwargs):
    """Run the 3d Gaussian Mixture Model over the 3d PET image.

    This algorithm is based on image intensity only. If the input image appears to sharp, grainy noisy is assumed
     and some morphological opening and closing is applied to the output binary mask.

    :param pet: 3d PET image.

    Kwargs:
        'n': number of clusters.
        'first': number of selected clusters.

    :return 3d binary mask of the tumor.
    """
    n = kwargs.get('n', 4)
    first = kwargs.get('first', 1)
    covar = kwargs.get('covar', 'diag')
    param = kwargs.get('param', 'm')
    init_param = kwargs.get('init_param', 'm')
    init_num = kwargs.get('init_num', 4)
    iter_num = kwargs.get('iter_num', 100)

    if 'param' in kwargs:
        init_param = param  # !

    z, w, h = tuple(pet.shape)
    image_array = np.reshape(pet, (z * w * h, 1))

    g = GMM(n_components=n,
            n_init=init_num,
            n_iter=iter_num,
            covariance_type=covar,
            params=param,
            init_params=init_param).fit(image_array)

    labels_g = g.predict(image_array)

    image_clustered = np.reshape(labels_g, (z, w, h))

    tumor_image = select_tumor(pet, image_clustered, n, first)
    tumor_image = desharp_post(pet, tumor_image)

    return tumor_image


def fcm3d(pet, **kwargs):
    """Run the 3d Spatial Distance Weighted Fuzzy C-Means algorithm over the 3d PET image.

    This algorithm is based on image intensity only. If the input image appears to sharp, grainy noisy is assumed
     and some blurring is applied, followed by morphological opening and closing applied to the output binary mask.

    :param pet: 3d PET image.

    Kwargs:
        'n': number of clusters (int).
        'first': number of selected clusters (int).
        'm': degree of fuzzy classification (float).
        'l': weight of spatial features (float).
        'num_nei': number of neighbors (int).

    :return 3d binary mask of the tumor.
    """
    n = kwargs.get('n', 2)
    first = kwargs.get('first', 1)
    m = kwargs.get('m', 2)
    l = kwargs.get('l', 0.5)
    num_nei = kwargs.get('num_nei', 1)
    error = kwargs.get('error', 1e-3)
    maxiter = kwargs.get('maxiter', 15)

    pet = desharp_pre(pet)

    z, w, h = tuple(pet.shape)
    image_array = np.reshape(pet, (1, z * w * h))  # a small difference from kmeans3d and gmm3d!

    sdwfcm = sdfcm_cmeans3(image_array, n, m, z=z, w=w, h=h, l=l,
                           number_neighbours=num_nei,
                           error=error, maxiter=maxiter, init=None)

    labels_s = np.argmax(sdwfcm[1], axis=0)

    image_clustered = np.reshape(labels_s, (z, w, h))

    tumor_image = select_tumor(pet, image_clustered, n, first)
    tumor_image = desharp_post(pet, tumor_image)

    return tumor_image


def dict3d(pet, **kwargs):
    """Run the 3d Dictionary-based labeling algorithm over the 3d PET image.

        The model has been trained on a limited set of training data from PETSEG'16 competition.
        If the input image appears to sharp, grainy noisy is assumed and some blurring is applied,
        followed by morphological opening and closing applied to the output binary mask.

        :param pet: 3d PET image.

        Kwargs:
            'th': threshold value for labeling (float).

        :return 3d binary mask of the tumor.
        """

    parms = open("dict3d-model.pkl", "rb")
    maxp, dictext, _, dictidl = pickle.load(parms)

    th = kwargs.get('th', 0.5)

    pet = desharp_pre(pet)

    image_label = dict_label_image3(pet, maxp, dictext, dictidl)
    tumor_image = 1.0 * (image_label > th)

    tumor_image = desharp_post(pet, tumor_image)

    return tumor_image
