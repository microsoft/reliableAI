# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import random
from itertools import combinations, product
from ML4S_Tools import IndependentTest

class CITester(object):
    def __init__(self, indexedDataT, confidenceLevel=0.99, maxConSizeForBackSearch=2, maxCountOfSepsets=50, maxPairsCount=100, maxEstimandCondCount=1000):
        self.DataT = indexedDataT # (#variables, #datasamples), already unique indexed
        self.Cardinalities = np.max(self.DataT, axis=1) + 1
        self.VariableCount = self.DataT.shape[0]
        self.ConfidenceLevel = confidenceLevel
        self.maxConSizeForBackSearch = maxConSizeForBackSearch
        self.maxCountOfSepsets = maxCountOfSepsets
        self.maxPairsCount = maxPairsCount
        self.maxEstimandCondCount = maxEstimandCondCount

        self.pValueSeverityCache = dict() # e.g. {(2, 3, frozenset([4, 5])): (0.05, 10.0), }
        self.ConditionalIndependenceCache = dict()

    def ConditionalIndependenceTest(self, x, y, S=None):
        '''
        Use independent as null hypothesis.
        The larger the pValue, the smaller the severity, the less relevant x and y is.
        The smaller the pValue, the larger the severity, chance that two variables are dependent is larger.
        :param x: int
        :param y: int
        :param S: set (or tuple or list) or None (default). conditioning set. None for independence test without condition
        :return: tuple (double pValue, double severity)
        '''
        S = frozenset() if S == None else frozenset(S) # frozenset is created faster than tuple(sorted(S))
        assert x != y and not x in S and not y in S
        x, y = (x, y) if (x < y) else (y, x)
        encoded = (x, y, S)
        if encoded in self.pValueSeverityCache:
            return self.pValueSeverityCache[encoded]
        if not S:
            pValue, severity = IndependentTest.BinomialIndependenceTest(self.DataT[[x, y]], self.Cardinalities[[x, y]])
        else:
            inds = list(S) + [x, y]
            pValue, severity = IndependentTest.CITest(self.DataT[inds], self.Cardinalities[inds])
        self.pValueSeverityCache[encoded] = (pValue, severity)
        return pValue, severity

    def GetSeparatingSetsFromPC(self, X, Y, PCX, PCY):
        # search from all subset of PCX and subset of PCY, get all subset s.t. pvalue > threshold
        # if no such subset, return one with the largest pValue
        # note: combinations(PCX, condsize), PCX is set, not list, so already sorted. no worry (x,y) and (y,x) repeated
        search_from_pcx = set.union(*[set(combinations(PCX, condsize)) for condsize in
                                      range(1 + min(self.maxConSizeForBackSearch, len(PCX)))])
        search_from_pcy = set.union(*[set(combinations(PCY, condsize)) for condsize in
                                      range(1, 1 + min(self.maxConSizeForBackSearch, len(PCY)))]) # condsize from 1, bcs empty set already tested.
        search_from = list(search_from_pcx.union(search_from_pcy))
        random.shuffle(search_from)

        valid_sepsets = set()
        maximum_invalid_sepset = (None, -1)
        # though not pval > 0.01, save the most nearest (subset, pval), e.g. pval=0.009
        for subset in search_from:
            pValue, severity = self.ConditionalIndependenceTest(X, Y, subset)
            if pValue > 1. - self.ConfidenceLevel:
                valid_sepsets.add(subset)
            elif pValue > maximum_invalid_sepset[1]:
                maximum_invalid_sepset = (subset, pValue)
            if len(valid_sepsets) == self.maxCountOfSepsets:
                return valid_sepsets
        return valid_sepsets if valid_sepsets else {maximum_invalid_sepset[0]}

    def ExtractTForkFeatureBasedOnPC(self, T, X, Y, PCT, PCX, PCY, Xpcypairs, Ypcxpairs, PCX_PCY_pairs):
        if len(Xpcypairs) > self.maxPairsCount: Xpcypairs = random.sample(Xpcypairs, self.maxPairsCount)
        if len(Ypcxpairs) > self.maxPairsCount: Ypcxpairs = random.sample(Ypcxpairs, self.maxPairsCount)
        if len(PCX_PCY_pairs) > self.maxPairsCount: PCX_PCY_pairs = random.sample(PCX_PCY_pairs, self.maxPairsCount)
        XY_sepsets = list(self.GetSeparatingSetsFromPC(X, Y, PCX, PCY)) # should it be [set()] if the maxpval still <= 0.01
        # list<Tuple<int>>. these tuples are sorted inside: (a,b) with a<b. so no repeats
        XY_sepsets_uT = [tuple(set(_).union({T})) for _ in XY_sepsets]  # has order, corresponding to XY_sepsets

        def _overlap(set1, set2):
            set1, set2 = set(set1), set(set2) # maybe input is list
            min_size = min(len(set1), len(set2)) # if minsize=0, return 1 because ∅∈anyset
            return 1. if min_size == 0 else len(set1.intersection(set2)) / min_size

        def _avg_overlap(set1, list_of_set2):
            if len(list_of_set2) == 0: return 0.
            return np.mean([_overlap(set1, set2) for set2 in list_of_set2])

        def _condon(estimands, conditions):
            '''
            :param estimands (bivariable): list or set of tuples, and these tuples are all with len=2
            :param conditions: list or set of tuples, and these tuples can be various in length, e.g. 0, 1, 2, 3,...
            :return: list of tuples (flatten, 1D)
            '''
            est_cond_pairs = [(e0, e1, c) for ((e0, e1), c) in product(estimands, conditions) if e0 not in c and e1 not in c]
            if len(est_cond_pairs) > self.maxEstimandCondCount: est_cond_pairs = random.sample(est_cond_pairs, self.maxEstimandCondCount)
            return [self.ConditionalIndependenceTest(e0, e1, c) for (e0, e1, c) in est_cond_pairs]

        scalings = [
            len(PCT),
            len(PCX),
            len(PCY),
            len(XY_sepsets),
            np.average([len(s) for s in XY_sepsets])
        ]

        overlaps = [
            _overlap(PCX, PCY),
            _overlap(PCX, PCT),
            _overlap(PCY, PCT),
            _avg_overlap({T}, XY_sepsets),
            _avg_overlap(PCX, XY_sepsets), # how many of sepsets are from PCX?
            _avg_overlap(PCY, XY_sepsets), # how many of sepsets are from PCY? (sum >= 1, bcs of repeats e.g. ∅)
            _avg_overlap(PCT, XY_sepsets),
        ]

        estimands_catagories = [
            [(X, Y)],
            Xpcypairs,
            Ypcxpairs,
            PCX_PCY_pairs
        ]

        conditions_categories = [
            [(T,)],
            XY_sepsets,
            XY_sepsets_uT,
            [(pct,) for pct in PCT - {X, Y}],
            [tuple(set(s).union({pct})) for pct in PCT - {X, Y} for s in XY_sepsets if pct not in s]
        ]

        return scalings + overlaps + [[_condon(ests, conds) for ests in estimands_catagories] for conds in conditions_categories]
        # est_cond pairs are in the product order: [XY_T, Xpcy_T, Y_pcx_T, pcxpcy_T, XY_S, ...]
