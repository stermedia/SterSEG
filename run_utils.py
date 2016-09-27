# Copyright (c) 2016 by Stermedia Sp. z o.o.
"""
This module provides helper functions for running segmentation of PET images.

Functions:
    load_pet: Load PET scan from NIfTI file.
    save_segmentation: Save segmentation of the PET scan into NIfTI or text format.
"""

import nibabel as nib
import numpy as np

from backend.input import NII2DInput


def load_pet(in_path):
    """Load PET scan from NIfTI file.

     :param in_path: name of the input file in the NIfTI format.
     :return: a list of 2d PET slices.
    """

    nii = NII2DInput()
    nii.load(in_path)

    pet = []
    for i in range(nii.max()):
        s = nii.slices(i)
        pet.append(s[0])

    pet /= np.max(pet)

    return pet


def save_segmentation(segmented_image, in_path, out_path, save_type):
    """Save segmentation of the PET scan into NIfTI or text format.

     :param segmented_image: segmented image
     :param in_path: name of the input file. Used only when exporting to NIfTI to get 'affine' and 'header'
     :param out_path: name of the output file.
     :param save_type: output file type: 'nii' for NIfTI, otherwise text."""

    if save_type == 'nii':
        data = np.array(segmented_image, dtype=np.int16).transpose(1, 2, 0)
        iimg = nib.load(in_path)
        img = nib.Nifti1Image(data, iimg.get_affine(), iimg.get_header())
        nib.save(img, out_path)
    else:
        org_size = segmented_image.shape
        np.savetxt(out_path, segmented_image.flatten().astype(int), fmt='%d', newline="\n", header=str(org_size))
