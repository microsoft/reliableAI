import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy import stats
import random
from itertools import combinations, product
from tools.utils import GaussianSignificance


class CITester(object):
    def __init__(self, indexedDataT, confidenceLevel=0.99, maxConSizeForBackSearch=2, maxCountOfSepsets=50, maxPairsCount=100, maxEstimandCondCount=1000):
        self.DataT = indexedDataT  # shape is [varCount, sampleSize], already unique indexed
        self.Cardinalities = np.max(self.DataT, axis=1) + 1
        self.VariableCount = self.DataT.shape[0]
        self.ConfidenceLevel = confidenceLevel
        self.maxConSizeForBackSearch = maxConSizeForBackSearch
        self.maxCountOfSepsets = maxCountOfSepsets
        self.maxPairsCount = maxPairsCount
        self.maxEstimandCondCount = maxEstimandCondCount
        self.pValueSeverityCache = dict()  # e.g. {(2, 3, frozenset([4, 5])): (0.05, 10.0), }
        self.ConditionalIndependenceCache = dict()
    
    def ConditionalIndependenceTest(self, x, y, S=None):
        '''
        Use independent as null hypothesis(H0).
        The larger the pValue, the smaller the severity, the less relevant x and y is.  (if pValue > alpha, accept H0)
        The smaller the pValue, the larger the severity, chance that two variables are dependent is larger.  (if pValue < alpha, reject H0)
        :param x: int
        :param y: int
        :param S: set (or tuple or list) or None (default). conditioning set. None for independence test without condition
        :return: tuple (double pValue, double severity)
        '''
        S = frozenset() if S == None else frozenset(S)  # frozenset is created faster than tuple(sorted(S))
        assert x != y and not x in S and not y in S
        x, y = (x, y) if (x < y) else (y, x)
        encoded = (x, y, S)
        if encoded in self.pValueSeverityCache:
            return self.pValueSeverityCache[encoded]
        if not S:
            pValue, severity = BinomialIndependenceTest(self.DataT[[x, y]], self.Cardinalities[[x, y]])
        else:
            indexs = list(S) + [x, y]
            pValue, severity = CITest(self.DataT[indexs], self.Cardinalities[indexs])
            # pValue, severity = IndependentTest.RobustCITest(self.DataT[indexs], self.Cardinalities[indexs])
        self.pValueSeverityCache[encoded] = (pValue, severity)
        return pValue, severity

    def DumpCICache(self, x, y, depth):
        x, y = (x, y) if (x < y) else (y, x)
        result = []
        for encoded in self.pValueSeverityCache:
            if encoded[0] == x and encoded[1] == y and len(encoded[2]) == depth:
                result.append((encoded, self.pValueSeverityCache[encoded]))
        return result

    def scipy_Pearson(self, x, y):
        co, p = stats.pearsonr(self.DataT[x], self.DataT[y])
        if abs(co) < 1e-8 or np.isnan(co): co = .0
        if abs(p) < 1e-8 or np.isnan(co): p = 0
        return co, p

    def GetSeparatingSetsFromPC(self, X, Y, PCX, PCY):
        # search from all subset of PCX and subset of PCY, get all subset s.t. pvalue > threshold
        # if no such subset, return one with the largest pValue
        # note: combinations(PCX, condsize), PCX is set, not list, so already sorted. no worry (x,y) and (y,x) repeated
        search_from_pcx = set.union(*[set(combinations(PCX, condsize)) for condsize in range(1 + min(self.maxConSizeForBackSearch, len(PCX)))])
        search_from_pcy = set.union(*[set(combinations(PCY, condsize)) for condsize in range(1, 1 + min(self.maxConSizeForBackSearch, len(PCY)))])  # condsize from 1, bcs empty set already tested.
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
        '''
        :param T: int
        :param X: int
        :param Y: int
        :param PCT: set<int>
        :param PCX: set<int>
        :param PCY: set<int>
        :return:
        '''
        if len(Xpcypairs) > self.maxPairsCount: Xpcypairs = random.sample(Xpcypairs, self.maxPairsCount)
        if len(Ypcxpairs) > self.maxPairsCount: Ypcxpairs = random.sample(Ypcxpairs, self.maxPairsCount)
        if len(PCX_PCY_pairs) > self.maxPairsCount: PCX_PCY_pairs = random.sample(PCX_PCY_pairs, self.maxPairsCount)
        XY_sepsets = list(self.GetSeparatingSetsFromPC(X, Y, PCX, PCY))  # should it be [set()] if the maxpval still <= 0.01
        # list<Tuple<int>>. these tuples are sorted inside: (a,b) with a<b. so no repeats
        XY_sepsets_uT = [tuple(set(_).union({T})) for _ in XY_sepsets]  # has order, corresponding to XY_sepsets

        def _overlap(set1, set2):
            set1, set2 = set(set1), set(set2)  # maybe input is list
            min_size = min(len(set1), len(set2))  # if minsize=0, return 1 because ∅∈anyset
            return 1. if min_size == 0 else len(set1.intersection(set2)) / min_size

        def _avg_overlap(set1, list_of_set2):
            if len(list_of_set2) == 0: return 0.
            return np.mean([_overlap(set1, set2) for set2 in list_of_set2])

        def _condon(estimands, conditions):
            '''
            :param estimands (bivariable): list or set of tuples, and these tuples are all with len=2
            :param conditions: list or set of tuples, and these tuples can be various in length, e.g. 0, 1, 2, 3,...
            :return: list of tuples (flatten, 1D)

            Deprecated: list of list of tuples:
                        [
                            [(pval, svrt), (pval, svrt), ..],
                            [(pval, svrt), (pval, svrt), ..], ..
                        ], where len(outerlist)==len(conditions) and each of len(innerlist)~=len(estimands)
            '''
            # return [[self.ConditionalIndependenceTest(e0, e1, c) for e0, e1 in estimands if e0 not in c and e1 not in c]
            #         for c in conditions]

            est_cond_pairs = [(e0, e1, c) for ((e0, e1), c) in product(estimands, conditions) if
                              e0 not in c and e1 not in c]
            if len(est_cond_pairs) > self.maxEstimandCondCount: est_cond_pairs = random.sample(est_cond_pairs,
                                                                                               self.maxEstimandCondCount)
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
            _avg_overlap(PCX, XY_sepsets),  # how many of sepsets are from PCX?
            _avg_overlap(PCY, XY_sepsets),  # how many of sepsets are from PCY? (sum >= 1, bcs of repeats e.g. ∅)
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
    xyJointCounts = np.bincount(xyIndexed, minlength=cardX * cardY).reshape(cardsXY)
    xMarginalCounts = np.sum(xyJointCounts, axis=1)
    yMarginalCounts = np.sum(xyJointCounts, axis=0)
    return xyJointCounts, xMarginalCounts, yMarginalCounts


def Fill3DCountTable(arraysSsXY, cardsSsXY):
    cardX, cardY = cardsSsXY[-2:]
    cardS = np.prod(cardsSsXY[:-2])

    cardCumProd = np.ones_like(cardsSsXY)
    cardCumProd[:-1] = np.cumprod(cardsSsXY[1:][::-1])[::-1]
    SxyIndexed = np.dot(cardCumProd[None], arraysSsXY)[0]
    SxyJointCounts = np.bincount(SxyIndexed, minlength=cardS * cardX * cardY).reshape((cardS, cardX, cardY))

    SMarginalCounts = np.sum(SxyJointCounts, axis=(1, 2))
    SMarginalCountsNonZero = SMarginalCounts != 0
    SMarginalCounts = SMarginalCounts[SMarginalCountsNonZero]
    SxyJointCounts = SxyJointCounts[SMarginalCountsNonZero]

    SxJointCounts = np.sum(SxyJointCounts, axis=2)
    SyJointCounts = np.sum(SxyJointCounts, axis=1)
    return SxyJointCounts, SMarginalCounts, SxJointCounts, SyJointCounts


def CalculateRareness(p, pObserved, sampleSizes):
    '''
    calculate the rareness of an observation from a given binomial distribution with certain sample size.
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
    pValue = 1. - GaussianSignificance(delta, 0, sigma)
    NumericalPrecision = 1.0e-8

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

    PxPyProductTable = PxTable[:, :, None] * PyTable[:, None, :]
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

    PxPyProductTable = PxTable[:, :, None] * PyTable[:, None, :]
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
        print(BinomialIndependenceTest(np.vstack([xs, ys]), np.array([10, 100])))
