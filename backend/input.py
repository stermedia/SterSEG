# Copyright (c) 2015-2016 by Stermedia Sp. z o.o.
"""
This module provides classes for loading and accessing radiology images
in the NIfTI format.
"""

from abc import abstractmethod

import nibabel as nib

from skimage.transform import resize
import numpy as np

from backend.manager import NiiSetManager


class Input(object):
    """Load and access data consisting of multiple slices."""

    @abstractmethod
    def load(self, sample, data_types):
        pass

    @abstractmethod
    def slices(self, n, data_type):
        pass


class NII2DInput(Input):
    """Load and access NifTI images."""

    def __init__(self, verbose=False):
        self.verbose = verbose

        # init
        self._sample = None
        self._types = None
        self._n = {}
        self.max_slices = None

    def load(self, sample, data_types=("PET",), max_slices=None):
        """Load a NIfTI sample.

        :param sample: name of the NIfTI sample.
        :param data_types: modality of the NIfTI sample.
        :param max_slices: maximum number of slices.
        :return: nothing.
        """
        self.max_slices = max_slices
        self._sample = sample
        self._types = data_types

        for t in data_types:
            self._n[t] = {}
            self._n[t]["data"] = nib.load(sample)
            (self._n[t]["z"],
             self._n[t]["x"],
             self._n[t]["y"],
             self._n[t]["max"], _, _, _, _) = self._n[t]["data"].header["dim"]

    def _convert_types(self, data_types):
        if not isinstance(data_types, list) \
                and not isinstance(data_types, tuple):
            data_types = (data_types, )

        assert set(self._types).issuperset(set(data_types))

        return tuple(data_types)

    def slices(self, n, data_types=None, scale=False, labeled=False):
        """Return a list or dictionary with n-th slice in selected modalities (data types).

        :param n: number of the slice.
        :param data_types: selected modalities.
        :type data_types: list
        :param scale: indicate whether slice intensities should be scaled or not into the range <0.255>.
        :type scale: bool
        :param labeled: indicate whether list or dictionary should be returned.
        :type scale: bool
        :return: a list or dictionary with n-th slice in selected modalities.
        """

        if not data_types:
            data_types = [k for k, v in self._n.iteritems()]

        data_types = self._convert_types(data_types)

        if self.verbose:
            print "slices for", data_types

        sm = NiiSetManager()
        for data_type in data_types:
            sm.add(data_type, range(self._n[data_type]["max"]))

        if self.max_slices:
            sm.add("max", range(self.max_slices))

        # calculate min dims
        min_h = min([v["x"] for k, v in self._n.iteritems()])
        min_w = min([v["y"] for k, v in self._n.iteritems()])

        slices = []
        labels = []

        for data_type in data_types:
            number = sm.get(n, data_type)

            n_data = self._n[data_type]["data"].get_data()

            if n_data.ndim == 3:
                data = n_data[:, :, number]
            else:
                data = n_data[:, :]

            if self.verbose:
                print "old size", data.shape[0], data.shape[1], "(", min_h, min_w, ")"

            # smart resize
            in_aspect = float(data.shape[0])/float(data.shape[1])
            out_aspect = float(min_h)/float(min_w)

            if data.shape == (min_h, min_w):
                data_resize = data
            elif in_aspect >= out_aspect:
                data_resize = resize(data,
                                     (min_h,
                                      int((float(min_h)/in_aspect)+0.5)),
                                     preserve_range=True)
            else:
                data_resize = resize(data,
                                     (int((float(min_w)*in_aspect) + 0.5),
                                      min_w),
                                     preserve_range=True)

            # fill blank space
            data = np.zeros((min_h, min_w))
            diff_h = min_h-data_resize.shape[0]
            diff_w = min_w-data_resize.shape[1]

            if self.verbose:
                print "diff", diff_h, diff_w

            data[diff_h/2:data.shape[0]-(diff_h/2), diff_w/2:data.shape[1]-(diff_w/2)] = data_resize

            # scale values
            if scale:
                max_val = np.max(data)
                if max_val != 0:
                    data = (data/max_val)*255

            if self.verbose:
                print "new size", data.shape[0], data.shape[1]

            if labeled:
                labels.append(data_type)

            slices.append(data)

        if labeled:
            result = {}
            for k, v in enumerate(labels):
                result[v] = slices[k]
            return result

        return slices

    def max(self, data_types=None):
        """Return the minimum of the maximal number of slices for each data type.
        :param data_types: selected modalities.
        :return: the minimum of the maximal number of slices for each data type.
        """

        if not data_types:
            data_types = [k for k, _ in self._n.iteritems()]

        data_types = self._convert_types(data_types)
        min_val = min([self._n[data_type]["max"] for data_type in data_types])

        if self.max_slices:
            return min(min_val, self.max_slices)

        return min_val
