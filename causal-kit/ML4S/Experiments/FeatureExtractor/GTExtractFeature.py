# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pickle, sys, os
from Tools import Graph, Utility, CITester, Dataset
from Tools.Utility import d_sep
from itertools import combinations
from p_tqdm import p_umap
import logging
from sklearn.feature_selection import mutual_info_classif

sys.path.append(os.path.dirname(os.getcwd()))


class FeatureExtractor:
    def __init__(self, bname, data_folder, dir_name, feature_folder, cit_folder, feature_name_list, *args):
        self.bname = bname
        self.train_path = f"{data_folder}/{dir_name}/{self.bname}"
        self.infer_path = f"../benchmarks/"
        self.sv_dir = f'{feature_folder}/{dir_name}/{self.bname}/'
        self.cit_folder = cit_folder + dir_name
        os.makedirs(self.cit_folder, exist_ok=True)
        self.cit_dir = os.path.join(cit_folder, self.bname)
        os.makedirs(self.sv_dir, exist_ok=True)
        os.makedirs(self.cit_dir, exist_ok=True)
        self.order_name = ['zero(placeholder)', 'first', 'second', 'third', 'forth']
        self.feature_name_list = feature_name_list
        self.logger = logging.getLogger(__name__)

        # get all data
        self.all = [(self.bname, False)] + \
                   [(_.replace(".npy", ""), True) for _ in os.listdir(self.train_path) if _.endswith(".npy")]

    def higher_order_extract_feature_for_one(self, order, skip_edges=False):
        def export_one(parameter):
            truth, dataset, est_graph, cit, feature_path, cit_path, is_train, onedataset = \
                self._prepare_data_and_model(parameter, order)
            prev_order_pred = {}
            for sample in onedataset:
                v1, v2 = sample[3]
                est_graph.add_undi_edge(v1, v2)
                prev_order_pred[sample[3]] = sample[2]

            # generate edge-wise feature
            data = []
            for edge in est_graph.UndirectedEdges:
                feature_dict = {}
                x, y = edge

                # assign label
                label = self._assign_label(truth, x, y)

                # calculate structural features
                x_vicinity, y_vicinity, x_y_prev_order_pred, structural_features_dict = \
                    self._get_local_structural_information(est_graph, x, y, prev_order_pred=prev_order_pred)
                feature_dict["previous order prediction"] = x_y_prev_order_pred
                feature_dict["degrees of target nodes"] = structural_features_dict["sparsity_feature"]
                feature_dict["density"] = structural_features_dict["overlapping_feature"]
                feature_dict["superiority"] = structural_features_dict["vicinity_relative_feature"]

                # find out whether we skip this edge
                # skip means the there is a super high probability that this edge exists
                # so later we won't train on this edge, but consider this as identified
                # we skip edges based on three conditions
                # independent in subgraph given certain condition; model predict high prob.; already skipped
                if skip_edges:
                    if min(structural_features_dict["sparsity_feature"]) < (order + 1) and not isinstance(x_y_prev_order_pred, str):
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

                all_vicinity = x_vicinity.union(y_vicinity)
                all_vicinity.discard(x)
                all_vicinity.discard(y)
                prev_order_ci = [(i[0][2], i[1][1]) for i in cit.DumpCICache(x, y, order - 1) if
                                 i[0][2].issubset(all_vicinity)]  # [(condition set, severity), ...]
                prev_order_ci.sort(key=lambda x: -x[1])
                if len(prev_order_ci) > 10:
                    prev_order_ci = prev_order_ci[:10]
                candidates = set()
                for i in prev_order_ci:
                    candidates = candidates.union(i[0])
                monotonic_residual = []
                for ci in prev_order_ci:
                    curr_cond_set = set(ci[0])
                    sev = ci[1]
                    min_sev = sev
                    for candidate in candidates:
                        if candidate in curr_cond_set:
                            continue
                        pval, new_sev = cit.ConditionalIndependenceTest(x, y, curr_cond_set.union({candidate}))
                        if new_sev < min_sev:
                            min_sev = new_sev
                    monotonic_residual.append(sev - min_sev)
                feature_dict["residual of conditional dependencies"] = np.hstack(
                    [Utility.meanstdmaxmin(monotonic_residual), Utility.percentileEmbedding(monotonic_residual, k=20)])

                feature_dict["k-order conditional dependencies"] = self._get_ci_features(x, y, all_vicinity, order, cit)

                # calculate mutual information between two discrete variables
                feature_dict["mutual information"] = mutual_info_classif(dataset.get_data_by_index(x).reshape(-1, 1),
                                                                         dataset.get_data_by_index(y))

                data.append((label,
                             np.hstack(
                                 [feature_dict[key] for key in self.feature_name_list]),
                             (x, y)))

            with open(feature_path, 'wb') as fp:
                pickle.dump(data, fp)
            with open(cit_path, "wb") as fp:
                pickle.dump(cit, fp)

        return export_one

    def first_order(self):
        def export_one(parameter):
            truth, dataset, est_graph, cit, feature_path, cit_path, is_train, _ = \
                self._prepare_data_and_model(parameter, 1)
            zero_order_dependency = {}
            thres = .1 if self.bname not in ["munin1", "diabetes", "pigs"] else .01

            for edge in combinations(est_graph.NodeIDs, 2):
                x, y = edge
                pval, severity = cit.ConditionalIndependenceTest(x, y)
                if pval < thres:
                    vicinity = est_graph.getNeighbor(x).union(est_graph.getNeighbor(y))
                    if not d_sep(x, y, vicinity, cit, thres):
                        est_graph.add_undi_edge(x, y)
                        zero_order_dependency[edge] = min(10, severity)
            if not is_train:
                self.logger.info("\nzero-order estimation {}".format(est_graph.Compare(truth.GetSkeleton())))

            # generate edge-wise feature
            data = []
            for edge in est_graph.UndirectedEdges:
                feature_dict = {}
                x, y = edge

                # assign label
                label = self._assign_label(truth, x, y)

                # calculate structural features
                x_vicinity, y_vicinity, xy_sev, structural_features_dict = \
                    self._get_local_structural_information(est_graph, x, y, higher_order=False, cit=cit)
                feature_dict["degrees of target nodes"] = structural_features_dict["sparsity_feature"]
                feature_dict["density"] = structural_features_dict["overlapping_feature"]
                feature_dict["superiority"] = structural_features_dict["vicinity_relative_feature"]

                # vicinity zero order
                #TODO explanation
                xy_vicinity_zero_order_pred = []
                for z in x_vicinity:
                    dep = zero_order_dependency[(x, z)] if (x, z) in zero_order_dependency else zero_order_dependency[
                        (z, x)]
                    xy_vicinity_zero_order_pred.append(dep)
                for z in y_vicinity:
                    dep = zero_order_dependency[(y, z)] if (y, z) in zero_order_dependency else zero_order_dependency[
                        (z, y)]
                    xy_vicinity_zero_order_pred.append(dep)

                feature_dict["previous order prediction"] = np.hstack([Utility.meanstdmaxmin(xy_vicinity_zero_order_pred),
                                                                 Utility.percentileEmbedding(
                                                                     xy_vicinity_zero_order_pred)])

                all_vicinity = x_vicinity.union(y_vicinity)
                all_vicinity.discard(x)
                all_vicinity.discard(y)
                one_order_ci = []
                for z in all_vicinity:
                    _, severity = cit.ConditionalIndependenceTest(x, y, [z])
                    one_order_ci.append(min(10, severity))
                if len(one_order_ci) != 0:
                    min_one_order_sev = min(one_order_ci)
                    feature_dict["residual of conditional dependencies"] = np.array([xy_sev - min_one_order_sev])
                else:
                    feature_dict["residual of conditional dependencies"] = np.array([.0])

                feature_dict["k-order conditional dependencies"] = self._get_ci_features(x, y, all_vicinity, 1, cit)

                # calculate mutual information between two discrete variables
                feature_dict["mutual information"] = mutual_info_classif(dataset.get_data_by_index(x).reshape(-1, 1),
                                                                         dataset.get_data_by_index(y))
                data.append((label,
                             np.hstack(
                                 [feature_dict[key] for key in self.feature_name_list]),
                             (x, y)))

            with open(feature_path, 'wb') as fp:
                pickle.dump(data, fp)
            with open(cit_path, "wb") as fp:
                pickle.dump(cit, fp)

        self.logger.info('#'*100)
        self.logger.info(f'start 1st order feature extraction...')
        p_umap(export_one, self.all)

    def second_order(self):
        self.logger.info('#'*100)
        self.logger.info(f'start 2nd order feature extraction...')
        p_umap(self.higher_order_extract_feature_for_one(2), self.all)

    def third_order(self):
        self.logger.info('#'*100)
        self.logger.info(f'start 3rd order feature extraction...')
        p_umap(self.higher_order_extract_feature_for_one(3), self.all)

    def forth_order(self):
        self.logger.info('#'*100)
        self.logger.info(f'start 4th order feature extraction...')
        p_umap(self.higher_order_extract_feature_for_one(4), self.all)

    def _prepare_data_and_model(self, parameter, order):
        """ prepare data for feature extraction """
        name, is_train = parameter
        if is_train:
            data_path = os.path.join(self.train_path, f"{name}.npy")
            graph_path = os.path.join(self.train_path, f"{name}.txt")
            feature_path = os.path.join(self.sv_dir, f"{name}-{self.order_name[order]}-in.pkl")
            if order > 1:
                model_out_path = os.path.join(self.sv_dir, f"{name}-{self.order_name[order - 1]}-out.pkl")
            cit_path = os.path.join(self.cit_dir, f"{name}.pkl")
        else:
            data_path = os.path.join(self.infer_path, f"npy10000/{self.bname}.npy")
            graph_path = os.path.join(self.infer_path, f"{self.bname}_graph.txt")
            feature_path = os.path.join(self.sv_dir, f"infer-{self.order_name[order]}-in.pkl")
            if order > 1:
                model_out_path = os.path.join(self.sv_dir, f"infer-{self.order_name[order - 1]}-out.pkl")
            cit_path = os.path.join(self.cit_dir, f"infer.pkl")
        truth = Graph.DiGraph(graph_path)
        dataset = Dataset.Dataset(data_path)
        est_graph = Graph.MixedGraph(numberOfNodes=dataset.VarCount)
        if order > 1:
            with open(model_out_path, "rb") as fp:
                onedataset = pickle.load(fp)
        else:
            onedataset = None
        if os.path.exists(cit_path):
            try:
                with open(cit_path, "rb") as fp:
                    cit = pickle.load(fp)
            except:
                cit = CITester.CITester(dataset.IndexedDataT, maxCountOfSepsets=50)
        else:
            cit = CITester.CITester(dataset.IndexedDataT, maxCountOfSepsets=50)
        return truth, dataset, est_graph, cit, feature_path, cit_path, is_train, onedataset

    @staticmethod
    def _assign_label(truth, x, y):
        """ assign label based on the truth graph"""
        if truth.is_adjacent(x, y):
            label = 1
        else:
            label = 0
        return label

    @staticmethod
    def _get_local_structural_information(est_graph, x, y, higher_order=True, prev_order_pred=None, cit=None):
        """

        Args:
            est_graph: estimated graph
            x: node x
            y: nody y
            higher_order: if this is used for higher order
            prev_order_pred: edge prediction from the previous stage
            cit: CI tester, only for first order feature extraction
        Returns:
            vicinity information, prediction information, feature dict
        """
        # calculate node degrees as feature
        x_vicinity = est_graph.getNeighbor(x)
        y_vicinity = est_graph.getNeighbor(y)
        sparsity_feature = np.array([len(x_vicinity), len(y_vicinity)]) if len(x_vicinity) < len(
            y_vicinity) else np.array([len(y_vicinity), len(x_vicinity)])

        # calculate density feature (overlapping ratio)
        common_vicinity = x_vicinity.intersection(y_vicinity)
        overlapping_ratio = len(common_vicinity) / min(len(x_vicinity), len(y_vicinity))
        overlapping_feature = np.array([overlapping_ratio])

        # superiority
        # get the edge prediction of x and y from the previous model
        # then calculates the superiority of this egde over other neighbouring edges
        x_count = 0
        y_count = 0
        if higher_order:
            x_y_prev_order_pred = prev_order_pred[(x, y)] if (x, y) in prev_order_pred else prev_order_pred[(y, x)]
            for z in x_vicinity:
                pred = prev_order_pred[(x, z)] if (x, z) in prev_order_pred else prev_order_pred[(z, x)]
                if not isinstance(x_y_prev_order_pred, str) and not isinstance(pred, str) and x_y_prev_order_pred > pred:
                    x_count += 1
            for z in y_vicinity:
                pred = prev_order_pred[(y, z)] if (y, z) in prev_order_pred else prev_order_pred[(z, y)]
                if not isinstance(x_y_prev_order_pred, str) and not isinstance(pred, str) and x_y_prev_order_pred > pred:
                    x_count += 1
        else:
            _, xy_sev = cit.ConditionalIndependenceTest(x, y)
            for z in x_vicinity:
                _, zx_sev = cit.ConditionalIndependenceTest(x, z)
                if xy_sev > zx_sev:
                    x_count += 1
            for z in y_vicinity:
                _, zy_sev = cit.ConditionalIndependenceTest(y, z)
                if xy_sev > zy_sev:
                    y_count += 1
        x_ratio = x_count / len(x_vicinity)
        y_ratio = y_count / len(y_vicinity)
        vicinity_relative_feature = np.array([x_ratio, y_ratio]) if x_ratio < y_ratio else np.array(
            [y_ratio, x_ratio])
        structural_features_dict = {
            "sparsity_feature": sparsity_feature,
            "overlapping_feature": overlapping_feature,
            "vicinity_relative_feature": vicinity_relative_feature
        }
        edge_information = x_y_prev_order_pred if higher_order else xy_sev
        return x_vicinity, y_vicinity, edge_information, structural_features_dict

    @staticmethod
    def _get_ci_features(x, y, all_vicinity, order, cit, k=20):
        # this min value is a hyperparameter which I keep the original value for different order
        min_value = 10 if order == 1 else 5
        order_selective_ci = []
        for cond_set in combinations(all_vicinity, order):
            _, sev = cit.ConditionalIndependenceTest(x, y, cond_set)
            order_selective_ci.append(min(min_value, sev))
        ci_feature = np.hstack([Utility.meanstdmaxmin(order_selective_ci),
                                Utility.percentileEmbedding(order_selective_ci, k=k)])
        return ci_feature
