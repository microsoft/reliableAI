#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pickle, sys, os, copy
from Tools import CITester, Dataset
from Tools import Graph, Utility
from itertools import combinations
from p_tqdm import p_umap

DATA_FOLDER = '/home/yang/ml4s-data/synthetic'
DIR_NAME = 'siblings-tune'
sys.path.append(os.path.dirname(os.getcwd()))

# TODO unify in config
FEATURE_FOLDER = "/home/yang/ml4s-data/feature"
CIT_FOLDER = f"/home/yang/ml4s-data/cit/{DIR_NAME}"

os.makedirs(CIT_FOLDER, exist_ok=True)


def FirstOrder(bname):
    train_path = f"{DATA_FOLDER}/{DIR_NAME}/{bname}"
    infer_path = f"../benchmarks/"
    sv_dir = f'{FEATURE_FOLDER}/{DIR_NAME}/{bname}/'
    cit_dit = os.path.join(CIT_FOLDER, bname)
    os.makedirs(sv_dir, exist_ok=True)
    os.makedirs(cit_dit, exist_ok=True)

    all = [(_.replace(".npy", ""), True) for _ in os.listdir(train_path) if _.endswith(".npy")] + [(bname, False)]

    def export_one(parameter):
        name, is_train = parameter
        if is_train:
            data_path = os.path.join(train_path, f"{name}.npy")
            graph_path = os.path.join(train_path, f"{name}.txt")
            feature_path = os.path.join(sv_dir, f"{name}-first-in.pkl")
            cit_path = os.path.join(cit_dit, f"{name}.pkl")
        else:
            data_path = os.path.join(infer_path, f"npy10000/{bname}.npy")
            graph_path = os.path.join(infer_path, f"{bname}_graph.txt")
            feature_path = os.path.join(sv_dir, f"infer-first-in.pkl")
            cit_path = os.path.join(cit_dit, f"infer.pkl")

        truth = Graph.DiGraph(graph_path)
        truth_bn = truth.GetBN()
        dataset = Dataset.Dataset(data_path)
        est_graph = Graph.MixedGraph(numberOfNodes=dataset.VarCount)
        if os.path.exists(cit_path):
            try:
                with open(cit_path, "rb") as fp:
                    cit = pickle.load(fp)
            except:
                cit = CITester.CITester(dataset.IndexedDataT, maxCountOfSepsets=50)
        else:
            cit = CITester.CITester(dataset.IndexedDataT, maxCountOfSepsets=50)
        zero_order_dependency = {}

        thres = .1 if bname != "diabetes" else .01

        def d_sep(x: int, y: int, vicinity: set, cit: CITester, level=2):
            vicinity.discard(x)
            vicinity.discard(y)
            for z in vicinity:
                pval, severity = cit.ConditionalIndependenceTest(x, y, [z])
                if pval > thres: return True
            return False

        # print(data_path)
        for edge in combinations(est_graph.NodeIDs, 2):

            x, y = edge

            pval, severity = cit.ConditionalIndependenceTest(x, y)
            if pval < thres:
                vicinity = est_graph.getNeighbor(x).union(est_graph.getNeighbor(y))
                if not d_sep(x, y, vicinity, cit):
                    est_graph.add_undi_edge(x, y)
                    zero_order_dependency[edge] = min(10, severity)
        if not is_train: print("zero-order estimation", est_graph.Compare(truth.GetSkeleton()))
        # generate edge-wise feature
        data = []
        for edge in est_graph.UndirectedEdges:
            x, y = edge

            if truth.is_adjacent(x, y):
                label = 1
            else:
                label = 0

            x_vicinity = est_graph.getNeighbor(x)
            y_vicinity = est_graph.getNeighbor(y)

            sparsity_feature = np.array([len(x_vicinity), len(y_vicinity)]) if len(x_vicinity) < len(
                y_vicinity) else np.array([len(y_vicinity), len(x_vicinity)])

            common_vicinity = x_vicinity.intersection(y_vicinity)
            overlapping_ratio = len(common_vicinity) / min(len(x_vicinity), len(y_vicinity))
            overlapping_feature = np.array([overlapping_ratio])

            xy_vicinity_zero_order_pred = []
            for z in x_vicinity:
                dep = zero_order_dependency[(x, z)] if (x, z) in zero_order_dependency else zero_order_dependency[
                    (z, x)]
                xy_vicinity_zero_order_pred.append(dep)
            for z in y_vicinity:
                dep = zero_order_dependency[(y, z)] if (y, z) in zero_order_dependency else zero_order_dependency[
                    (z, y)]
                xy_vicinity_zero_order_pred.append(dep)

            xy_vicinity_zero_order_pred_feature = np.hstack([Utility.meanstdmaxmin(xy_vicinity_zero_order_pred),
                                                             Utility.percentileEmbedding(xy_vicinity_zero_order_pred)])

            _, xy_sev = cit.ConditionalIndependenceTest(x, y)
            x_count = 0
            for z in x_vicinity:
                _, zx_sev = cit.ConditionalIndependenceTest(x, z)
                if xy_sev > zx_sev: x_count += 1
            x_ratio = x_count / len(x_vicinity)
            y_count = 0
            for z in y_vicinity:
                _, zy_sev = cit.ConditionalIndependenceTest(y, z)
                if xy_sev > zy_sev: y_count += 1
            y_ratio = y_count / len(y_vicinity)

            vicinity_relative_feature = np.array([x_ratio, y_ratio]) if x_ratio < y_ratio else np.array(
                [y_ratio, x_ratio])

            all_vicinity = x_vicinity.union(y_vicinity)
            all_vicinity.discard(x)
            all_vicinity.discard(y)
            one_order_ci = []
            for z in all_vicinity:
                pval, severity = cit.ConditionalIndependenceTest(x, y, [z])
                one_order_ci.append(min(10, severity))
            if len(one_order_ci) != 0:
                max_one_order_sev = min(one_order_ci)
                monotonic_feature = np.array([xy_sev - max_one_order_sev])
            else:
                monotonic_feature = np.array([.0])

            one_order_ci_feature = np.hstack(
                [Utility.meanstdmaxmin(one_order_ci), Utility.percentileEmbedding(one_order_ci, k=20)])

            data.append((label, np.hstack((sparsity_feature, overlapping_feature, vicinity_relative_feature,
                                           monotonic_feature, xy_vicinity_zero_order_pred_feature, \
                                           one_order_ci_feature)), (x, y)))

        with open(feature_path, 'wb') as fp:
            pickle.dump(data, fp)
        with open(cit_path, "wb") as fp:
            pickle.dump(cit, fp)

    p_umap(export_one, all)


def SecondOrder(bname):
    train_path = f"{DATA_FOLDER}/{DIR_NAME}/{bname}"
    infer_path = f"../../benchmarks/"
    sv_dir = f'{FEATURE_FOLDER}/{DIR_NAME}/{bname}/'
    cit_dit = os.path.join(CIT_FOLDER, bname)
    os.makedirs(sv_dir, exist_ok=True)
    os.makedirs(cit_dit, exist_ok=True)

    all = [(_.replace(".npy", ""), True) for _ in os.listdir(train_path) if _.endswith(".npy")] + [(bname, False)]

    def export_one(parameter):
        name, is_train = parameter
        if is_train:
            data_path = os.path.join(train_path, f"{name}.npy")
            graph_path = os.path.join(train_path, f"{name}.txt")
            feature_path = os.path.join(sv_dir, f"{name}-second-in.pkl")
            model_out_path = os.path.join(sv_dir, f"{name}-first-out.pkl")
            cit_path = os.path.join(cit_dit, f"{name}.pkl")
        else:
            data_path = os.path.join(infer_path, f"npy10000/{bname}.npy")
            graph_path = os.path.join(infer_path, f"{bname}_graph.txt")
            feature_path = os.path.join(sv_dir, f"infer-second-in.pkl")
            model_out_path = os.path.join(sv_dir, f"infer-first-out.pkl")
            cit_path = os.path.join(cit_dit, f"infer.pkl")

        truth = Graph.DiGraph(graph_path)
        truth_bn = truth.GetBN()
        dataset = Dataset.Dataset(data_path)
        est_graph = Graph.MixedGraph(numberOfNodes=dataset.VarCount)
        with open(cit_path, "rb") as fp:
            cit = pickle.load(fp)
        cit: CITester.CITester

        with open(model_out_path, "rb") as fp:
            onedataset = pickle.load(fp)
        prev_order_pred = {}
        for sample in onedataset:
            v1, v2 = sample[3]
            est_graph.add_undi_edge(v1, v2)
            prev_order_pred[sample[3]] = sample[2]

        # generate edge-wise feature
        data = []
        for edge in est_graph.UndirectedEdges:
            x, y = edge

            x_vicinity = est_graph.getNeighbor(x)
            y_vicinity = est_graph.getNeighbor(y)
            sparsity_feature = np.array([len(x_vicinity), len(y_vicinity)]) if len(x_vicinity) < len(
                y_vicinity) else np.array([len(y_vicinity), len(x_vicinity)])
            x_y_prev_order_pred = prev_order_pred[(x, y)] if (x, y) in prev_order_pred else prev_order_pred[(y, x)]

            if truth.is_adjacent(x, y):
                label = 1
            else:
                label = 0

            if min(sparsity_feature) < 3 and not isinstance(x_y_prev_order_pred, str):
                min_vicinity = x_vicinity if len(x_vicinity) < len(y_vicinity) else y_vicinity
                min_vicinity.discard(x)
                min_vicinity.discard(y)
                pval, _ = cit.ConditionalIndependenceTest(x, y, min_vicinity)
                if pval < .05:
                    data.append((label, "skip", (x, y)))
                continue
            if isinstance(x_y_prev_order_pred, str):
                data.append((label, "skip", (x, y)))
                continue
            if not isinstance(x_y_prev_order_pred, str) and x_y_prev_order_pred > .9:
                data.append((label, "skip", (x, y)))
                continue

            common_vicinity = x_vicinity.intersection(y_vicinity)
            overlapping_ratio = len(common_vicinity) / min(len(x_vicinity), len(y_vicinity))
            overlapping_feature = np.array([overlapping_ratio])

            x_count = 0
            for z in x_vicinity:
                pred = prev_order_pred[(x, z)] if (x, z) in prev_order_pred else prev_order_pred[(z, x)]
                if not isinstance(x_y_prev_order_pred, str) and not isinstance(pred,
                                                                               str) and x_y_prev_order_pred > pred: x_count += 1
            x_ratio = x_count / len(x_vicinity)
            y_count = 0
            for z in y_vicinity:
                pred = prev_order_pred[(y, z)] if (y, z) in prev_order_pred else prev_order_pred[(z, y)]
                if not isinstance(x_y_prev_order_pred, str) and not isinstance(pred,
                                                                               str) and x_y_prev_order_pred > pred: x_count += 1
            y_ratio = y_count / len(y_vicinity)

            vicinity_relative_feature = np.array([x_ratio, y_ratio]) if x_ratio < y_ratio else np.array(
                [y_ratio, x_ratio])

            all_vicinity = x_vicinity.union(y_vicinity)
            all_vicinity.discard(x)
            all_vicinity.discard(y)

            second_order_selective_ci = []

            prev_order_ci = [(i[0][2], i[1][1]) for i in cit.DumpCICache(x, y, 1) if
                             i[0][2].issubset(all_vicinity)]  # [(condition set, severity), ...]
            prev_order_ci.sort(key=lambda x: -x[1])
            if len(prev_order_ci) > 20: prev_order_ci = prev_order_ci[:20]
            candidates = set()
            for i in prev_order_ci: candidates = candidates.union(i[0])
            monotonic_residual = []
            for ci in prev_order_ci:
                curr_cond_set = set(ci[0])
                sev = ci[1]
                min_sev = sev
                for candidate in candidates:
                    if candidate in curr_cond_set: continue
                    pval, new_sev = cit.ConditionalIndependenceTest(x, y, curr_cond_set.union({candidate}))
                    second_order_selective_ci.append(min(5, new_sev))
                    if new_sev < min_sev: min_sev = new_sev
                monotonic_residual.append(sev - min_sev)

            # cond_sets = list(combinations(all_vicinity, 2))
            # pruned_ci = 0
            # if len(cond_sets) > 50:
            #     cond_sets = cond_sets[:50]
            #     pruned_ci = 1

            # for cond_set in combinations(all_vicinity, 2):
            #     _, sev = cit.ConditionalIndependenceTest(x, y, cond_set)
            #     second_order_selective_ci.append(min(5, sev))

            monotonic_residual_feature = np.hstack(
                [Utility.meanstdmaxmin(monotonic_residual), Utility.percentileEmbedding(monotonic_residual, k=20)])
            second_order_selective_ci_feature = np.hstack([Utility.meanstdmaxmin(second_order_selective_ci),
                                                           Utility.percentileEmbedding(second_order_selective_ci,
                                                                                       k=20)])

            data.append((label, np.hstack(([x_y_prev_order_pred], sparsity_feature, overlapping_feature,
                                           vicinity_relative_feature, monotonic_residual_feature, \
                                           second_order_selective_ci_feature)), (x, y)))
            # data.append((label, np.hstack((second_order_selective_ci_feature)), (x, y)))

        with open(feature_path, 'wb') as fp:
            pickle.dump(data, fp)

    p_umap(export_one, all)


def ThirdOrder(bname):
    train_path = f"{DATA_FOLDER}/{DIR_NAME}/{bname}"
    infer_path = f"../../benchmarks/"
    sv_dir = f'{FEATURE_FOLDER}/{DIR_NAME}/{bname}/'
    cit_dit = os.path.join(CIT_FOLDER, bname)
    os.makedirs(sv_dir, exist_ok=True)
    os.makedirs(cit_dit, exist_ok=True)

    all = [(_.replace(".npy", ""), True) for _ in os.listdir(train_path) if _.endswith(".npy")] + [(bname, False)]

    def export_one(parameter):
        name, is_train = parameter
        if is_train:
            data_path = os.path.join(train_path, f"{name}.npy")
            graph_path = os.path.join(train_path, f"{name}.txt")
            feature_path = os.path.join(sv_dir, f"{name}-third-in.pkl")
            model_out_path = os.path.join(sv_dir, f"{name}-second-out.pkl")
            cit_path = os.path.join(cit_dit, f"{name}.pkl")
        else:
            data_path = os.path.join(infer_path, f"npy10000/{bname}.npy")
            graph_path = os.path.join(infer_path, f"{bname}_graph.txt")
            feature_path = os.path.join(sv_dir, f"infer-third-in.pkl")
            model_out_path = os.path.join(sv_dir, f"infer-second-out.pkl")
            cit_path = os.path.join(cit_dit, f"infer.pkl")

        truth = Graph.DiGraph(graph_path)
        truth_bn = truth.GetBN()
        dataset = Dataset.Dataset(data_path)
        est_graph = Graph.MixedGraph(numberOfNodes=dataset.VarCount)
        with open(cit_path, "rb") as fp:
            cit = pickle.load(fp)
        cit: CITester.CITester

        with open(model_out_path, "rb") as fp:
            onedataset = pickle.load(fp)
        prev_order_pred = {}
        for sample in onedataset:
            v1, v2 = sample[3]
            est_graph.add_undi_edge(v1, v2)
            prev_order_pred[sample[3]] = sample[2]

        # generate edge-wise feature
        data = []
        for edge in est_graph.UndirectedEdges:
            x, y = edge

            x_vicinity = est_graph.getNeighbor(x)
            y_vicinity = est_graph.getNeighbor(y)
            sparsity_feature = np.array([len(x_vicinity), len(y_vicinity)]) if len(x_vicinity) < len(
                y_vicinity) else np.array([len(y_vicinity), len(x_vicinity)])
            x_y_prev_order_pred = prev_order_pred[(x, y)] if (x, y) in prev_order_pred else prev_order_pred[(y, x)]

            # if truth.is_adjacent(x, y): label = 1
            # elif len(truth_bn.minimal_dseparator(str(x), str(y))) > 3: label = 0
            # else: 
            #     hops = truth.reachable(x, y, 3)
            #     if hops > 3: label = 0
            #     else:
            #         label =1 / hops
            if truth.is_adjacent(x, y):
                label = 1
            else:
                label = 0

            if min(sparsity_feature) < 4 and not isinstance(x_y_prev_order_pred, str):
                min_vicinity = x_vicinity if len(x_vicinity) < len(y_vicinity) else y_vicinity
                min_vicinity.discard(x)
                min_vicinity.discard(y)
                pval, _ = cit.ConditionalIndependenceTest(x, y, min_vicinity)
                if pval < .1:
                    data.append((label, "skip", (x, y)))
                continue
            if not isinstance(x_y_prev_order_pred, str) and x_y_prev_order_pred > .95:
                data.append((label, "skip", (x, y)))
                continue
            if isinstance(x_y_prev_order_pred, str):
                data.append((label, "skip", (x, y)))
                continue

            common_vicinity = x_vicinity.intersection(y_vicinity)
            overlapping_ratio = len(common_vicinity) / min(len(x_vicinity), len(y_vicinity))
            overlapping_feature = np.array([overlapping_ratio])

            x_count = 0
            for z in x_vicinity:
                pred = prev_order_pred[(x, z)] if (x, z) in prev_order_pred else prev_order_pred[(z, x)]
                if not isinstance(x_y_prev_order_pred, str) and not isinstance(pred,
                                                                               str) and x_y_prev_order_pred > pred: x_count += 1
            x_ratio = x_count / len(x_vicinity)
            y_count = 0
            for z in y_vicinity:
                pred = prev_order_pred[(y, z)] if (y, z) in prev_order_pred else prev_order_pred[(z, y)]
                if not isinstance(x_y_prev_order_pred, str) and not isinstance(pred,
                                                                               str) and x_y_prev_order_pred > pred: x_count += 1
            y_ratio = y_count / len(y_vicinity)

            vicinity_relative_feature = np.array([x_ratio, y_ratio]) if x_ratio < y_ratio else np.array(
                [y_ratio, x_ratio])

            all_vicinity = x_vicinity.union(y_vicinity)
            all_vicinity.discard(x)
            all_vicinity.discard(y)

            third_order_selective_ci = []

            prev_order_ci = [(i[0][2], i[1][1]) for i in cit.DumpCICache(x, y, 2) if
                             i[0][2].issubset(all_vicinity)]  # [(condition set, severity), ...]
            prev_order_ci.sort(key=lambda x: -x[1])
            if len(prev_order_ci) > 10: prev_order_ci = prev_order_ci[:10]
            candidates = set()
            for i in prev_order_ci: candidates = candidates.union(i[0])
            monotonic_residual = []
            for ci in prev_order_ci:
                curr_cond_set = ci[0]
                sev = ci[1]
                min_sev = sev
                for candidate in candidates:
                    if candidate in curr_cond_set: continue
                    pval, new_sev = cit.ConditionalIndependenceTest(x, y, curr_cond_set + {candidate})
                    if new_sev < min_sev: min_sev = new_sev
                monotonic_residual.append(sev - min_sev)

            for cond_set in combinations(all_vicinity, 3):
                _, sev = cit.ConditionalIndependenceTest(x, y, cond_set)
                third_order_selective_ci.append(min(5, sev))

            monotonic_residual_feature = np.hstack(
                [Utility.meanstdmaxmin(monotonic_residual), Utility.percentileEmbedding(monotonic_residual, k=20)])
            third_order_selective_ci_feature = np.hstack([Utility.meanstdmaxmin(third_order_selective_ci),
                                                          Utility.percentileEmbedding(third_order_selective_ci, k=20)])

            data.append((label, np.hstack(
                ([x_y_prev_order_pred], sparsity_feature, overlapping_feature, vicinity_relative_feature, \
                 monotonic_residual_feature, third_order_selective_ci_feature)), (x, y)))

        with open(feature_path, 'wb') as fp:
            pickle.dump(data, fp)

    p_umap(export_one, all)


def ForthOrder(bname):
    train_path = f"{DATA_FOLDER}/{DIR_NAME}/{bname}"
    infer_path = f"../../benchmarks/"
    sv_dir = f'{FEATURE_FOLDER}/{DIR_NAME}/{bname}/'
    cit_dit = os.path.join(CIT_FOLDER, bname)
    os.makedirs(sv_dir, exist_ok=True)
    os.makedirs(cit_dit, exist_ok=True)

    all = [(bname, False)] + [(_.replace(".npy", ""), True) for _ in os.listdir(train_path) if _.endswith(".npy")]

    def export_one(parameter):
        name, is_train = parameter
        if is_train:
            data_path = os.path.join(train_path, f"{name}.npy")
            graph_path = os.path.join(train_path, f"{name}.txt")
            feature_path = os.path.join(sv_dir, f"{name}-forth-in.pkl")
            model_out_path = os.path.join(sv_dir, f"{name}-third-out.pkl")
            cit_path = os.path.join(cit_dit, f"{name}.pkl")
        else:
            data_path = os.path.join(infer_path, f"npy10000/{bname}.npy")
            graph_path = os.path.join(infer_path, f"{bname}_graph.txt")
            feature_path = os.path.join(sv_dir, f"infer-forth-in.pkl")
            model_out_path = os.path.join(sv_dir, f"infer-third-out.pkl")
            cit_path = os.path.join(cit_dit, f"infer.pkl")
        truth = Graph.DiGraph(graph_path)
        truth_bn = truth.GetBN()
        dataset = Dataset.Dataset(data_path)
        est_graph = Graph.MixedGraph(numberOfNodes=dataset.VarCount)
        with open(cit_path, "rb") as fp:
            cit = pickle.load(fp)
        cit: CITester.CITester

        with open(model_out_path, "rb") as fp:
            onedataset = pickle.load(fp)
        prev_order_pred = {}
        for sample in onedataset:
            v1, v2 = sample[3]
            est_graph.add_undi_edge(v1, v2)
            prev_order_pred[sample[3]] = sample[2]

        # generate edge-wise feature
        data = []
        for edge in est_graph.UndirectedEdges:
            x, y = edge

            x_vicinity = est_graph.getNeighbor(x)
            y_vicinity = est_graph.getNeighbor(y)
            sparsity_feature = np.array([len(x_vicinity), len(y_vicinity)]) if len(x_vicinity) < len(
                y_vicinity) else np.array([len(y_vicinity), len(x_vicinity)])
            x_y_prev_order_pred = prev_order_pred[(x, y)] if (x, y) in prev_order_pred else prev_order_pred[(y, x)]

            # if truth.is_adjacent(x, y): label = 1
            # elif len(truth_bn.minimal_dseparator(str(x), str(y))) > 4: label = 0
            # else: 
            #     hops = truth.reachable(x, y, 3)
            #     if hops > 3: label = 0
            #     else:
            #         label =1 / hops
            if truth.is_adjacent(x, y):
                label = 1
            else:
                label = 0

            if min(sparsity_feature) < 5 and not isinstance(x_y_prev_order_pred, str):
                min_vicinity = x_vicinity if len(x_vicinity) < len(y_vicinity) else y_vicinity
                min_vicinity.discard(x)
                min_vicinity.discard(y)
                pval, _ = cit.ConditionalIndependenceTest(x, y, min_vicinity)
                if pval < .1:
                    data.append((label, "skip", (x, y)))
                continue
            if not isinstance(x_y_prev_order_pred, str) and x_y_prev_order_pred > .95:
                data.append((label, "skip", (x, y)))
                continue
            if isinstance(x_y_prev_order_pred, str):
                data.append((label, "skip", (x, y)))
                continue

            common_vicinity = x_vicinity.intersection(y_vicinity)
            overlapping_ratio = len(common_vicinity) / min(len(x_vicinity), len(y_vicinity))
            overlapping_feature = np.array([overlapping_ratio])

            x_count = 0
            for z in x_vicinity:
                pred = prev_order_pred[(x, z)] if (x, z) in prev_order_pred else prev_order_pred[(z, x)]
                if not isinstance(x_y_prev_order_pred, str) and not isinstance(pred,
                                                                               str) and x_y_prev_order_pred > pred: x_count += 1
            x_ratio = x_count / len(x_vicinity)
            y_count = 0
            for z in y_vicinity:
                pred = prev_order_pred[(y, z)] if (y, z) in prev_order_pred else prev_order_pred[(z, y)]
                if not isinstance(x_y_prev_order_pred, str) and not isinstance(pred,
                                                                               str) and x_y_prev_order_pred > pred: x_count += 1
            y_ratio = y_count / len(y_vicinity)

            vicinity_relative_feature = np.array([x_ratio, y_ratio]) if x_ratio < y_ratio else np.array(
                [y_ratio, x_ratio])

            all_vicinity = x_vicinity.union(y_vicinity)
            all_vicinity.discard(x)
            all_vicinity.discard(y)

            forth_order_selective_ci = []

            prev_order_ci = [(i[0][2], i[1][1]) for i in cit.DumpCICache(x, y, 3) if
                             i[0][2].issubset(all_vicinity)]  # [(condition set, severity), ...]
            prev_order_ci.sort(key=lambda x: -x[1])
            if len(prev_order_ci) > 10: prev_order_ci = prev_order_ci[:10]
            candidates = set()
            for i in prev_order_ci: candidates = candidates.union(i[0])
            monotonic_residual = []
            for ci in prev_order_ci:
                curr_cond_set = ci[0]
                sev = ci[1]
                min_sev = sev
                for candidate in candidates:
                    if candidate in curr_cond_set: continue
                    pval, new_sev = cit.ConditionalIndependenceTest(x, y, curr_cond_set + {candidate})
                    forth_order_selective_ci.append(new_sev)
                    if new_sev < min_sev: min_sev = new_sev
                monotonic_residual.append(sev - min_sev)

            for cond_set in combinations(all_vicinity, 4):
                try:
                    _, sev = cit.ConditionalIndependenceTest(x, y, cond_set)
                    forth_order_selective_ci.append(min(5, sev))
                except:
                    pass

            monotonic_residual_feature = np.hstack(
                [Utility.meanstdmaxmin(monotonic_residual), Utility.percentileEmbedding(monotonic_residual, k=20)])
            forth_order_selective_ci_feature = np.hstack([Utility.meanstdmaxmin(forth_order_selective_ci),
                                                          Utility.percentileEmbedding(forth_order_selective_ci, k=20)])

            # x_vicinity = est_graph.getNeighbor(x)
            # x_vicinity.discard(y)
            # y_vicinity = est_graph.getNeighbor(y)
            # y_vicinity.discard(x)

            # common_vicinity = x_vicinity.intersection(y_vicinity)
            # uncommon_vicinity = x_vicinity.union(y_vicinity) - x_vicinity.intersection(y_vicinity)
            # vicinity_cmb1 = list(combinations(common_vicinity, 4))
            # vicinity_cmb2 = [(i[0], i[1][0], i[1][1], i[1][2]) for i in list(product(common_vicinity, combinations(uncommon_vicinity, 3)))]
            # if len(vicinity_cmb1) + len(vicinity_cmb2) <= 100:
            #     vicinity_cmb = vicinity_cmb1 + vicinity_cmb2
            #     vicinity_cmb += list(combinations(uncommon_vicinity, 3))[:100-len(vicinity_cmb)]
            # elif len(vicinity_cmb1) < 100 and len(vicinity_cmb1) + len(vicinity_cmb2) >= 100:
            #     random.shuffle(vicinity_cmb2)
            #     vicinity_cmb = vicinity_cmb1 + vicinity_cmb2[:100-len(vicinity_cmb1)]
            # else:
            #     vicinity_cmb = vicinity_cmb1

            # second_order_selective_ci = []
            # for z in vicinity_cmb:
            #     try:
            #         pval, severity = cit.ConditionalIndependenceTest(x, y, z)
            #         second_order_selective_ci.append(min(10, severity))
            #     except:
            #         continue
            # if len(second_order_selective_ci) == 0: second_order_selective_ci = [-1]
            # second_order_selective_ci_feature = np.hstack([Utility.meanstdmaxmin(second_order_selective_ci), Utility.percentileEmbedding(second_order_selective_ci, k=20)])

            data.append((label, np.hstack(
                ([x_y_prev_order_pred], sparsity_feature, overlapping_feature, vicinity_relative_feature, \
                 monotonic_residual_feature, forth_order_selective_ci_feature)), (x, y)))
            # data.append((label, np.hstack((second_order_selective_ci_feature)), (x, y)))

        with open(feature_path, 'wb') as fp:
            pickle.dump(data, fp)

    p_umap(export_one, all)


def Prune(bname, order_tag):
    train_path = f"{DATA_FOLDER}/{DIR_NAME}/{bname}"
    infer_path = f"../../benchmarks/"
    sv_dir = f'{FEATURE_FOLDER}/{DIR_NAME}/{bname}/'
    cit_dit = os.path.join(CIT_FOLDER, bname)
    os.makedirs(sv_dir, exist_ok=True)
    os.makedirs(cit_dit, exist_ok=True)

    if order_tag == "second":
        curr_order = 3
    if order_tag == "third":
        curr_order = 4
    if order_tag == "forth":
        curr_order = 5

    def export_one(parameter):
        name, is_train = parameter
        if is_train:
            data_path = os.path.join(train_path, f"{name}.npy")
            graph_path = os.path.join(train_path, f"{name}.txt")
            model_out_path = os.path.join(sv_dir, f"{name}-{order_tag}-out.pkl")
            cit_path = os.path.join(cit_dit, f"{name}.pkl")
        else:
            data_path = os.path.join(infer_path, f"npy10000/{bname}.npy")
            graph_path = os.path.join(infer_path, f"{bname}_graph.txt")
            model_out_path = os.path.join(sv_dir, f"infer-{order_tag}-out.pkl")
            cit_path = os.path.join(cit_dit, f"infer.pkl")

        truth = Graph.DiGraph(graph_path)
        truth_skl = truth.GetSkeleton()
        dataset = Dataset.Dataset(data_path)
        est_graph = Graph.MixedGraph(numberOfNodes=dataset.VarCount)
        with open(cit_path, "rb") as fp:
            cit = pickle.load(fp)
        cit: CITester.CITester
        # print(cit_path)
        # cit = CITester.CITester(dataset.IndexedDataT, maxCountOfSepsets=50)

        with open(model_out_path, "rb") as fp:
            onedataset = pickle.load(fp)
        prev_order_pred = {}
        for sample in onedataset:
            v1, v2 = sample[3]
            est_graph.add_undi_edge(v1, v2)
            prev_order_pred[sample[3]] = sample[2]

        pruned_est_graph = Graph.MixedGraph(numberOfNodes=dataset.VarCount)

        # generate edge-wise feature
        data = []
        for edge in copy.copy(est_graph.UndirectedEdges):
            x, y = edge

            x_vicinity = est_graph.getNeighbor(x)
            y_vicinity = est_graph.getNeighbor(y)
            sparsity_feature = np.array([len(x_vicinity), len(y_vicinity)]) if len(x_vicinity) < len(
                y_vicinity) else np.array([len(y_vicinity), len(x_vicinity)])
            x_y_prev_order_pred = prev_order_pred[(x, y)] if (x, y) in prev_order_pred else prev_order_pred[(y, x)]

            if truth.is_adjacent(x, y):
                label = 1
            else:
                label = 0

            if min(sparsity_feature) < (curr_order + 1) and not isinstance(x_y_prev_order_pred, str):
                min_vicinity = x_vicinity if len(x_vicinity) < len(y_vicinity) else y_vicinity
                min_vicinity.discard(x)
                min_vicinity.discard(y)
                pval, _ = cit.ConditionalIndependenceTest(x, y, min_vicinity)
                if pval < .1:
                    pruned_est_graph.add_undi_edge(x, y)
                continue
            if not isinstance(x_y_prev_order_pred, str) and x_y_prev_order_pred > .75:
                data.append((label, "skip", (x, y)))
                pruned_est_graph.add_undi_edge(x, y)
                continue
            if isinstance(x_y_prev_order_pred, str):
                data.append((label, "skip", (x, y)))
                pruned_est_graph.add_undi_edge(x, y)
                continue

        print("Pruned", bname, pruned_est_graph.Compare(truth_skl))

        with open(cit_path, "wb") as fp:
            pickle.dump(cit, fp)

    export_one([bname, False])
