# SterSEG

## Information

    Copyright (c) 2016 by Stermedia Sp. z o.o.

    Contributors:
    J.Czakon, W.Dyrka, P.Giedziun

    Please cite:
    [1] J.Czakon, F.Drapejkowski, G.Zurek, P.Giedziun, J.Zebrowski, W.Dyrka. Machine learning methods for accurate
        delineation of tumors in PET images. MICCAI 2016 - PETSEG challenge, Athens, 21.10.2016

    Part of this work has been conducted under the Support Programme of the Partnership between Higher Education
        and Science and Business Activity Sector financed by City of Wroclaw.

## RUN

    docker-compose build

    # kmeans, k=2
    docker-compose run --rm pet python run.py -m KM -kn 2 -kf 1 -i your_pet.nii -ot nii -o your_output.nii

    # more ..
    docker-compose run --rm pet python run.py --help
