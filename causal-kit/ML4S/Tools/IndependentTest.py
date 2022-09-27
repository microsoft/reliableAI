# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
from numpy.core.fromnumeric import shape
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from Tools import Utility

NumericalPrecision = 1.0e-8

def Fill2DCountTable(arraysXY, cardsXY):
    '''
    e.g. arraysXY: the observed dataset contains 5 samples, on variable x and y they're
        x: 0 1 2 3 0
        y: 1 0 1 2 1
    cardsXY: [4, 3]
    fill in the counts by index, we have the joint count table in 4 * 3:
        xy| 0 1 2
        --|-------
        0 | 0 2 0
        1 | 1 0 0
        2 | 0 1 0
        3 | 0 0 1
    note: if sample size is large enough, in theory:
            min(arraysXY[i]) == 0 && max(arraysXY[i]) == cardsXY[i] - 1
        however some values may be missed.
        also in joint count, not every value in [0, cardX * cardY - 1] occurs.
        that's why we pass cardinalities in, and use `minlength=...` in bincount
    '''
    cardX, cardY = cardsXY
    xyIndexed = arraysXY[0] * cardY + arraysXY[1]
    xyJointCounts = np.bincount(xyIndexed, minlength=cardX*cardY).reshape(cardsXY)
    xMarginalCounts = np.sum(xyJointCounts, axis=1)
    yMarginalCounts = np.sum(xyJointCounts, axis=0)
    return xyJointCounts, xMarginalCounts, yMarginalCounts

def Fill3DCountTable(arraysSsXY, cardsSsXY):
    cardX, cardY = cardsSsXY[-2:]
    cardS = np.prod(cardsSsXY[:-2])

    cardCumProd = np.ones_like(cardsSsXY)
    cardCumProd[:-1] = np.cumprod(cardsSsXY[1:][::-1])[::-1]
    SxyIndexed = np.dot(cardCumProd[None], arraysSsXY)[0]
    SxyJointCounts = np.bincount(SxyIndexed, minlength=cardS*cardX*cardY).reshape((cardS, cardX, cardY))

    SMarginalCounts = np.sum(SxyJointCounts, axis=(1, 2))
    SMarginalCountsNonZero = SMarginalCounts != 0
    SMarginalCounts = SMarginalCounts[SMarginalCountsNonZero]
    SxyJointCounts = SxyJointCounts[SMarginalCountsNonZero]

    SxJointCounts = np.sum(SxyJointCounts, axis=2)
    SyJointCounts = np.sum(SxyJointCounts, axis=1)
    return SxyJointCounts, SMarginalCounts, SxJointCounts, SyJointCounts

def CalculateRareness(p, pObserved, sampleSizes):
    '''
    calculate the rareness of an observation from a given binomial distribution with certan sample size.
    the closer p and pObserved are, the smaller severity is, the larger the return value(pValue) is. (in our case, the more independent)
    k, m, n are respectively the cardinality of S, x, y. if S=empty, k==1.
    :param p: tensor, (k, m, n) the probability of binomial distribution, float32
    :param pObserved: tensor, (k, m, n) the observed empirical distribution, float32
            check: sum(pObserved, axis=(1,2)) should be all 1. (for each value of condition)
    :param sampleSizes: np.array in shape (k,)
    :return:
        pValue: np.array, the closer the two probabilities, the higher the return value. bounded within [0,1]
        severity: np.array, very often that the rareness is too small that the numerical precision is quesitonable, and we use this #sigma as another equivalent indicator
    '''
    delta = np.abs(p - pObserved)

    p1 = sampleSizes / (9. + sampleSizes)
    p2 = 1. - p1
    pMax = np.maximum(p1, p2)
    pMin = np.minimum(p1, p2)
    p = np.maximum(np.minimum(p, pMax[:, None, None]), pMin[:, None, None])

    sigma = np.sqrt(p * (1. - p) / sampleSizes[:, None, None])

    severity = delta / sigma
    pValue = 1. - Utility.GaussianSignificance(delta, 0, sigma)

    forcePrecisionPoints = delta < NumericalPrecision
    pValue[forcePrecisionPoints] = 1.
    severity[forcePrecisionPoints] = 0.

    return pValue, severity

def BinomialIndependenceTest(arraysXY, cardsXY):
    '''
    if xArray has cardinality m, yArray has cardinality n,
        then for each cell in the m*n table, calculate the rareness and severity
        and pick the cell with the largest severity as result
    rareness here is used to indicate p-value, the smaller the p-value,
        the more difference between two probabilities, which means the more against the null hypothess (independent)
    because only if two variables are independent,
        the joint probability equals to the multiplication of two marginal probabilities.
    :param arraysXY: np.array, (2, sampleSize)
    :param cardsXY: np.array, (2, )
    :return: (float pValue, float severity)
    '''
    sampleSize = arraysXY.shape[1]
    xyJointCounts, xMarginalCounts, yMarginalCounts = Fill2DCountTable(arraysXY, cardsXY)
    PxyJointTable = xyJointCounts * 1. / sampleSize
    PxPyProductTable = np.outer(xMarginalCounts * 1. / sampleSize, yMarginalCounts * 1. / sampleSize)
    pValue, severity = CalculateRareness(PxPyProductTable[None], PxyJointTable[None], np.array([sampleSize]))
    bestCellID = np.argmax(severity)
    return pValue.ravel()[bestCellID], severity.ravel()[bestCellID]

def CITest(arraysSsXY, cardsSsXY):
    '''
    if Ss contains k variables:
    :param arraysSsXY: np.array, int (indexed, 0 to ...), (k+2, sampleSize, )
    :param cardsSsXY: np.array, int, (k+2, )
    '''
    SxyJointCounts, SMarginalCounts, SxJointCounts, SyJointCounts = Fill3DCountTable(arraysSsXY, cardsSsXY)

    PSxyJointTable = SxyJointCounts * 1. / SMarginalCounts[:, None, None]
    PxTable = SxJointCounts * 1. / SMarginalCounts[:, None]
    PyTable = SyJointCounts * 1. / SMarginalCounts[:, None]

    PxPyProductTable = PxTable[:, : ,None] * PyTable[:, None, :]
    pValue, severity = CalculateRareness(PxPyProductTable, PSxyJointTable, SMarginalCounts)
    bestCellID = np.argmax(severity)

    return pValue.ravel()[bestCellID], severity.ravel()[bestCellID]

def RobustCITest(arraysSsXY, cardsSsXY):
    '''
    if Ss contains k variables:
    :param arraysSsXY: np.array, int (indexed, 0 to ...), (k+2, sampleSize, )
    :param cardsSsXY: np.array, int, (k+2, )
    '''
    SxyJointCounts, SMarginalCounts, SxJointCounts, SyJointCounts = Fill3DCountTable(arraysSsXY, cardsSsXY)

    PSxyJointTable = SxyJointCounts * 1. / SMarginalCounts[:, None, None]
    PxTable = SxJointCounts * 1. / SMarginalCounts[:, None]
    PyTable = SyJointCounts * 1. / SMarginalCounts[:, None]

    PxPyProductTable = PxTable[:, : ,None] * PyTable[:, None, :]
    pValue, severity = CalculateRareness(PxPyProductTable, PSxyJointTable, SMarginalCounts)

    sortedPValue = np.sort(pValue.ravel())
    sortedSeverity = np.sort(severity.ravel())
    if sortedPValue.shape[0] > 5:
        idx = int(0.8 * sortedPValue.shape[0])
    elif sortedPValue.shape[0] > 1:
        idx = sortedPValue.shape[0] - 1
    else:
        idx = 0
    return sortedPValue[idx], sortedSeverity[idx]

if __name__ == '__main__':

    for _ in range(1):
        np.random.seed(0)
        sample_size = 10000
        xs = np.random.randint(0, 10, (sample_size))
        ys = np.random.randint(0, 10, (sample_size))
        # Ss = np.vstack([xs.reshape((1, -1)), np.random.randint(0, 100, (10))])
        # print(xs,ys,Ss)
        # # print(BinomialIndependenceTest(np.vstack([xs, ys]), np.array([10, 100])))
        # # Ss = np.random.randint(0, 10, (1, 10000))
        # print(CITest(np.vstack([Ss, xs, ys]), np.array([10, 100, 10, 100])))
        print(BinomialIndependenceTest(np.vstack([xs, ys]), np.array([10,100])))

