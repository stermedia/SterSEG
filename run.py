# Copyright (c) 2016 by Stermedia Sp. z o.o.
"""
This module provides interface for running several methods for binary segmentation of PET images in 3d.

Please cite:
[1] J.Czakon, F.Drapejkowski, G.Zurek, P.Giedziun, J.Zebrowski, W.Dyrka. Machine learning methods for accurate
delineation of tumors in PET images. MICCAI 2016 - PETSEG challenge, Athens, 21.10.2016

Part of this work was conducted under the Support Programme of the Partnership between Higher Education
and Science and Business Activity Sector financed by City of Wroclaw.

Example:
    python run.py -m KM -kn 2 -kf 1 -i your_pet.nii -ot nii -o your_output.nii
    python run.py --help
"""

import argparse
import warnings

import numpy as np

from run_utils import load_pet, save_segmentation
from models import kmeans3d, gmm3d, fcm3d, dict3d

warnings.filterwarnings('ignore')

# Parse arguments:

parser = argparse.ArgumentParser(prog='RUN', conflict_handler='resolve')
parser.add_argument('-m', '--method', help='used method')
parser.add_argument('-i', '--input', default='clinical_1_PET.nii', help='input file')
parser.add_argument('-o', '--output', default='clinical_1_PRE.txt', help='output file')
parser.add_argument('-ot', default='txt', help='output type ("txt", "nii")')

parser.add_argument('-kn', default=2, help='[KM] number of clusters (int)')
parser.add_argument('-kf', default=1, help='[KM] number of selected clusters (int)')

parser.add_argument('-gn', default=4, help='[GMM] number of clusters (int)')
parser.add_argument('-gf', default=1, help='[GMM] number of selected clusters (int)')

parser.add_argument('-fn', default=2, help='[FCM] number of clusters (int)')
parser.add_argument('-ff', default=1, help='[FCM] number of selected clusters (int)')
parser.add_argument('-fm', default=2, help='[FCM] degree of fuzzy classification (float)')
parser.add_argument('-fl', default=0.5, help='[FCM] weight of spatial features (float)')
parser.add_argument('-fnn', default=1, help='[FCM] number of neighbors (int)')

parser.add_argument('-dth', default=0.5, help='[DICT] th (float)')

parser.add_argument('-at', default='avg', help='[ALL] type ("avg", "and") (string)')

args = parser.parse_args()

# Run selected method:
if args.method in ['KM', 'GMM', 'FCM', 'DICT', 'ALL']:

    pet = load_pet(args.input)

    if args.method == 'KM':
        segmented_image = kmeans3d(pet, **{'k': int(args.kn), 'first': int(args.kf)})
    elif args.method == 'GMM':
        segmented_image = gmm3d(pet, **{'n': int(args.gn), 'first': int(args.gf)})
    elif args.method == 'FCM':
        segmented_image = fcm3d(pet, **{'n': int(args.fn), 'first': int(args.ff),
                                        'm': float(args.fm), 'l': float(args.fl),
                                        'num_nei': int(args.fnn)})
    elif args.method == 'DICT':
        segmented_image = dict3d(pet, **{'th': float(args.dth)})
    elif args.method == 'ALL':
        results = [dict3d(pet, **{'th': float(args.dth)}),
                   fcm3d(pet, **{'n': int(args.fn), 'first': int(args.ff),
                                 'm': float(args.fm), 'l': float(args.fl),
                                 'num_nei': int(args.fnn)}),
                   gmm3d(pet, **{'n': int(args.gn), 'first': int(args.gf)}),
                   kmeans3d(pet, **{'k': int(args.kn), 'first': int(args.kf)})]

        if args.at == 'avg':  # consensus: tumor if 3/4 methods agree
            segmented_image = np.zeros(results[0].shape)
            for r in results:
                segmented_image += r
            segmented_image = (segmented_image > len(results) / 2).astype(int)
        else:  # consensus: tumor if all methods agree (use this for high specificity)
            segmented_image = np.ones(results[0].shape)
            for r in results:
                segmented_image = np.logical_and(r, segmented_image)

    # Save results to NIfTI or text format
    if args.ot == 'nii':
        save_segmentation(segmented_image, args.input, args.output, save_type='nii')
    else:
        save_segmentation(segmented_image, args.input, args.output, save_type='txt')

else:
    parser.print_help()
