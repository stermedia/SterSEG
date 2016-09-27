# Copyright (c) 2015-2016 by Stermedia Sp. z o.o.
"""
This module provides functions for learning and using the Dictionary model
for image segmentation, based on Dahl and Larsen, Learning Dictionaries of
Discriminative Image Patches, in: Proc. British Machine Vision Conference, 
p.77, 2011, and adapted to 3d images.

Please cite:
[1] J.Czakon, P.Giedziun, M.Blach, G.Zurek, P.Krajewski, W.Dyrka. An ensemble algorithm for the nuclei segmentation
in the histological images. MICCAI 2015 - Computational Brain Tumor Cluster of Events, Munich, 9.10.2015
[2] J.Czakon, F.Drapejkowski, G.Zurek, P.Giedziun, J.Zebrowski, W.Dyrka. Machine learning methods for accurate
delineation of tumors in PET images. MICCAI 2016 - PETSEG challenge, Athens, 21.10.2016

Part of this work was conducted under the Support Programme of the Partnership between Higher Education
and Science and Business Activity Sector financed by City of Wroclaw.

Functions:
    samples2patches3: Extract patches and labels from a dataset of 3d images.
    dict_init3: Initialize dictionary for a set of patches.
    dict_train3: Train the dictionary model for a list of patches using the vector quantization method.
    dict_label_patch3: Label a 3d patch.
    dict_label_image3: Label a 3d image.
"""

import random

import numpy as np

random.seed()


def _label_cmp3(patch1, patch2):
    diff = 0.0
    xmax = len(patch1)
    ymax = len(patch1[0])
    zmax = len(patch1[0][0])
    for x in range(xmax):
        for y in range(ymax):
            for z in range(zmax):
                diff += 2.0 * abs(patch1[x][y][z] - patch2[x][y][z])
    diff = 1.0 - (diff / (xmax * ymax * zmax))
    return diff


def _patch_cmp3(patch1, patch2):
    diff = 0.0
    xmax = len(patch1)
    ymax = len(patch1[0])
    zmax = len(patch1[0][0])
    for x in range(xmax):
        for y in range(ymax):
            for z in range(zmax):
                diff += abs(patch1[x][y][z] - patch2[x][y][z]) ** 2
    diff = (diff ** 0.5) / (xmax * ymax * zmax)
    return diff


def _dict_mean(dicto, p, dsize):
    dict_sum = dicto[p, 0].copy()
    for r in range(1, dsize):
        dict_sum += dicto[p, r]  # copy not needed
    return dict_sum / dsize


def samples2patches3(sample_images, sample_labels, leave_out, patch_size):
    """Extract patches and labels from a dataset of 3d images.

    :param sample_images: list of images
    :param sample_labels: list of corresponding labels
    :param leave_out: list of samples to be left out in training.
    :param patch_size: patch size in pixels.
    :return: a list of patches and a list of corresponding labels.
    """

    patchs = []
    labels = []

    for i in [i for i in range(len(sample_labels)) if i not in leave_out]:
        xmax = len(sample_labels[i])
        ymax = len(sample_labels[i][0])
        zmax = len(sample_labels[i][0][0])

        for x in range(xmax - patch_size + 1):
            for y in range(ymax - patch_size + 1):
                for z in range(zmax - patch_size + 1):
                    patchs.append(sample_images[i][x:x + patch_size, y:y + patch_size, z:z + patch_size])
                    labels.append(sample_labels[i][x:x + patch_size, y:y + patch_size, z:z + patch_size])

    return patchs, labels


def dict_init3(patchs, labels, sample_freq_init, lab_sim_thresh):
    """Initialize dictionary for a set of patches.

        :param patchs: list of patches
        :param labels: list of corresponding labels
        :param sample_freq_init: fraction of patches to be used for dictionary initialization.
        :param lab_sim_thresh: label similarity threshold to create new class.
        :return: number of classes, dictionary of patches, dictionary of labels.
        """

    dictext = {}
    dictlab = {}
    dsize = {}

    maxp = 0

    sample = {}
    samplesize = 0

    for i in range(len(labels)):
        if random.random() < sample_freq_init:
            sample[samplesize] = i
            samplesize += 1

    random.shuffle(sample)  # shuffle works in place

    for s in sample:

        candidate_labpatch = 1.0 * labels[sample[s]]
        candidate_txtpatch = 1.0 * patchs[sample[s]]

        if not dictlab:
            dictext[maxp, 0] = candidate_txtpatch.copy()
            dictlab[maxp, 0] = candidate_labpatch.copy()
            dsize[maxp] = 1
            maxp += 1
            continue

        maxw = -1000  # a magic constant, beware!
        for p in range(maxp):
            mydictlab = _dict_mean(dictlab, p, dsize[p])
            w = _label_cmp3(mydictlab, candidate_labpatch)
            if (w > maxw):
                maxw = w
                fitp = p

        if maxw < lab_sim_thresh:
            dictext[maxp, 0] = candidate_txtpatch.copy()
            dictlab[maxp, 0] = candidate_labpatch.copy()
            dsize[maxp] = 1
            maxp += 1
        else:
            dictext[fitp, dsize[fitp]] = candidate_txtpatch.copy()
            dictlab[fitp, dsize[fitp]] = candidate_labpatch.copy()
            dsize[fitp] += 1

    for p in range(maxp):
        dictext[p] = _dict_mean(dictext, p, dsize[p])
        dictlab[p] = _dict_mean(dictlab, p, dsize[p])

    return maxp, dictext, dictlab


def dict_train3(patchs, labels, sample_freq_train, lab_sim_thresh, maxp, dictext, dictlab, n_iter, tau):
    """Train the dictionary model for a list of patches using the vector quantization method.

    :param patchs: list of patches
    :param labels: list of corresponding labels
    :param sample_freq_train: fraction of patches to be used for dictionary training.
    :param lab_sim_thresh: label similarity threshold for class membership.
    :param maxp: number of classes.
    :param dictext: dictionary of patches.
    :param dictlab: dictionary of labels.
    :param n_iter: number of iterations
    :param tau: drag coefficient (see Dahl&Larsen 2011)
    :return: number of classes, dictionary of patches, dictionary of labels, dictionary of idealized labels.
    """

    dictidl = {}
    dictxtn = {}
    dictxtp = {}
    dictla2 = {}
    dpsize = {}
    dnsize = {}
    dlsize = {}

    pxmax = len(dictext[0])
    pymax = len(dictext[0][0])
    pzmax = len(dictext[0][0][0])

    for _ in range(n_iter):

        dictidl = dictlab.copy()
        for p in range(maxp):
            dpsize[p] = 0
            dnsize[p] = 0
            dlsize[p] = 0
            for x in range(pxmax):
                for y in range(pymax):
                    for z in range(pzmax):
                        dictidl[p][x][y][z] = 1.0 * (dictlab[p][x][y][z] > 0.5)

        for i in range(len(labels)):

            if random.random() < sample_freq_train:

                candidate_labpatch = 1.0 * labels[i]
                candidate_txtpatch = 1.0 * patchs[i]

                minv = 1000  # a magic constant, beware!
                fitp = -1
                for p in range(maxp):
                    v = _patch_cmp3(dictext[p], candidate_txtpatch)
                    if v < minv:
                        minv = v
                        fitp = p
                w = _label_cmp3(dictidl[fitp], candidate_labpatch)

                if w >= lab_sim_thresh:
                    dictxtp[fitp, dpsize[fitp]] = candidate_txtpatch.copy()
                    dpsize[fitp] += 1

                    if w < 0.0:
                        dictxtn[fitp, dnsize[fitp]] = candidate_txtpatch.copy()
                        dnsize[fitp] += 1

                    dictla2[fitp, dlsize[fitp]] = candidate_labpatch.copy()
                    dlsize[fitp] += 1

        for p in range(maxp):

            if dlsize[p] > 0:
                dictlab[p] = _dict_mean(dictla2, p, dlsize[p])
            if dpsize[p] > 0:
                pmean = _dict_mean(dictxtp, p, dpsize[p])
            else:
                pmean = 0.0
            if dnsize[p] > 0:
                nmean = _dict_mean(dictxtn, p, dnsize[p])
            else:
                nmean = 0.0
            dictext[p] = dictext[p].copy() + tau * (pmean - nmean)

    for p in range(maxp):

        for x in range(pxmax):
            for y in range(pymax):
                for z in range(pzmax):
                    dictidl[p][x][y][z] = 1.0 * (dictlab[p][x][y][z] > 0.5)

    return maxp, dictext, dictlab, dictidl


def dict_label_patch3(patch, maxp, mydictext, mydictlab):
    """
    Label a 3d patch.

    :param patch: a 3d patch
    :param maxp: number of classes.
    :param dictext: dictionary of patches.
    :param dictlab: dictionary of labels.
    :return: a labelled 3d patch.
    """

    current_txtpatch = 1.0 * patch
    mintextdiff = 1000.0  # a magic constant, beware!
    matchtext = -1
    for p in range(maxp):
        diff = _patch_cmp3(mydictext[p], current_txtpatch)
        if diff < mintextdiff:
            matchtext = p
            mintextdiff = diff

    return mydictlab[matchtext]


def dict_label_image3(image, maxp, mydictext, mydictlab):
    """
    Label a 3d image.

    Note that the resulting label for each pixel is the average of all patch labels overlapping with the pixel.
    For binary outcome, please apply a threshold, e.g. 0.5.

    :param image: a 3d image
    :param maxp: number of classes.
    :param dictext: dictionary of patches.
    :param dictlab: dictionary of labels.
    :return: a labelled 3d image.
    """

    xmax = len(image)
    ymax = len(image[0])
    zmax = len(image[0][0])

    image_label = np.zeros((xmax, ymax, zmax))

    patch_sizex = len(mydictext[0])
    patch_sizey = len(mydictext[0][0])
    patch_sizez = len(mydictext[0][0][0])

    for x in range(xmax - patch_sizex + 1):
        for y in range(ymax - patch_sizey + 1):
            for z in range(zmax - patch_sizez + 1):
                current_txtpatch = image[x:x + patch_sizex, y:y + patch_sizey, z:z + patch_sizez]
                predict_patchlab = dict_label_patch3(current_txtpatch, maxp, mydictext, mydictlab)
                image_label[x:x + patch_sizex, y:y + patch_sizey, z:z + patch_sizez] += predict_patchlab

    return image_label / (patch_sizex * patch_sizey * patch_sizez)
