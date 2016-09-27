# Copyright (c) 2015-2016 by Stermedia Sp. z o.o.
"""
This module provides classes for managing sets of studies in the NifTI format.
"""

class NiiSetManager(object):
    """Manage a set of NifTI data.
    """

    def __init__(self):
        self.data = {}

    def add(self, label, numbers):
        self.data[label] = numbers

    def size(self):
        return min([len(v) for _, v in self.data.iteritems()])

    def get(self, n, data_type):
        items = self.size()
        assert n < items

        v = self.data[data_type]
        return v[int((len(v)*n)/items)]
        