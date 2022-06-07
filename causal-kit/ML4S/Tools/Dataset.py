#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, data_path, downsample=None):
        """
        note: we calculate unique at this data prep step,
                so that each variable's observations are inversed indexs 0, 1, ...
                and so no need to unique again and again during CITest.
              in fact, the benchmarks .txt/.npy is already saved as inversed indexes.
        :param data_path: string, endswith .npy or .npz or .txt, where it's a frame
            of (sampleSize, varCount) shape, and categorical int32 datatype
        :param downsample: downsample from all samples. default by None.
            if int: randomly pick int samples from all
            if float in range (0, 1]: randomly pick percent from all
            if list: pick samples accoding to list of indexes
        """
        raw_data = np.loadtxt(data_path) if data_path.endswith('.txt') else np.load(data_path)  # now (sampleSize, varCount)
        raw_data = raw_data.astype(np.int32)
        raw_sample_size = raw_data.shape[0]
        if downsample is not None:
            if isinstance(downsample, list):
                raw_data = raw_data[downsample]
            else:
                downsample = downsample if isinstance(downsample, int) else raw_sample_size * downsample
                assert downsample <= raw_sample_size
                raw_data = raw_data[np.random.choice(range(raw_sample_size), downsample, replace=False)]
        raw_data = raw_data.T  # now (varCount, sampleSize)

        def _unique(row):
            return np.unique(row, return_inverse=True)[1]

        self.IndexedDataT = np.apply_along_axis(_unique, 1, raw_data).astype(np.int32)
        self.SampleSize = self.IndexedDataT.shape[1]
        self.VarCount = self.IndexedDataT.shape[0]

    def get_data_by_index(self, index):
        return self.IndexedDataT[index]

    def to_pandas(self):
        return pd.DataFrame(self.IndexedDataT.T, columns=[str(i) for i in range(self.VarCount)])
