import time
import numpy as np
import pickle, os
# from experiments.Tools import Graph, Utility, CITester, Dataset
from itertools import combinations
# from p_tqdm import p_umap
from sklearn.feature_selection import mutual_info_classif


from pathos.multiprocessing import Pool, cpu_count
from base.dag import DiGraph
from base.pdag import MixedGraph
from base.dataloader import Dataset
from base.citester import CITester


class SkeletonFeatureExtractor:
    def __init__(self, benchmark_name, train_dir, infer_dir, feature_dir, cit_dir, contain_env, intervention_size):
        self.benchmark_name = benchmark_name
        self.train_dir = train_dir
        self.infer_dir = infer_dir
        self.feature_dir = feature_dir
        self.cit_dir = cit_dir
        self.contain_env = contain_env
        self.intervention_size = intervention_size
        self.order_name = ['zero', 'first', 'second', 'third', 'forth']
        self.all_datasets = [(file_name.replace(".npy", ""), True) for file_name in os.listdir(self.train_dir) if file_name.endswith(".npy")] + [(self.benchmark_name, False)]
    
    
    def order_extractor(self, order, logger):
        logger.info("#" * 100)
        logger.info(f"start {order} order feature extraction...")
        order_num = self.order_name.index(order)
        order_fun = self.export_first_order if order_num == 1 else self.export_higher_order
        parameters = [(item1, item2, order_num) for item1, item2 in self.all_datasets]

        start_time = time.time()
        with Pool(cpu_count() / 2) as p:
            p.map(order_fun, parameters)
        end_time = time.time()
        logger.info(f"{order} order feature extraction total time is {end_time - start_time}.")
    

    def export_first_order(self, parm):
        """
        In this function, we take (0-order partial graph & 0-order CD & 1-order CD) as trainset, to train 1-order model

        Args:
            parm (tuple): (file_name, is_sibling, order)
        """
        file_name, is_sibling, order = parm
        truth_graph, dataset, estimate_graph, cit, cit_path, samples_in_path, _ = self._prepare_current_order_data(file_name, is_sibling, order)
        thres = .1 if self.benchmark_name not in ["munin1", "diabetes", "pigs"] else .01  # which means significance level
        zero_order_pred = {}
        
        for edge in combinations(estimate_graph.NodeIDs, 2):
            x, y = edge
            if self._skip_training_edge(x, y):
                continue
            pval, severity = cit.ConditionalIndependenceTest(x, y)  # here, we get p_value and z-sigma (which called conditional dependency), not z means marginal independent test
            if pval < thres:  # if pValue < thres, reject H0, which means x and y are dependent (*** Here, equivalent to using Marginal Independence Test first, note that it is not affected by the following Conditional Independence Test ***)
                vicinity = estimate_graph.getNeighbor(x).union(estimate_graph.getNeighbor(y))  # Pruning Strategy 2 for 1-Order Models CI Tests
                if not self._d_separation(x, y, vicinity, cit, thres):  # if all pValue < thres, reject H0, which means x and y are dependent
                    estimate_graph.add_undi_edge(x, y)
                    zero_order_pred[edge] = min(10, severity)
        
        if not is_sibling:
            print("zero-order estimation {}".format(estimate_graph.Compare(truth_graph.GetSkeleton())))
        
        # generate edge-wise feature
        all_samples = []
        for edge in estimate_graph.UndirectedEdges:
            x, y = edge
            feature_dict = {}
            label = self._assign_label(truth_graph, x, y)
            
            # calculate local structural features
            x_vicinity, y_vicinity, xy_severity, structural_features_dict = self._get_local_structural_information(estimate_graph, x, y, higher_order=False, cit=cit)
            feature_dict["superiority"] = structural_features_dict["vicinity_relative_feature"]
            feature_dict["degrees of target nodes"] = structural_features_dict["sparsity_feature"]
            feature_dict["density"] = structural_features_dict["overlapping_feature"]
            
            xy_prev_order_pred = [zero_order_pred[(x, z) if (x, z) in zero_order_pred else (z, x)] for z in x_vicinity] + [zero_order_pred[(y, z) if (y, z) in zero_order_pred else (z, y)] for z in y_vicinity]
            feature_dict["previous order prediction"] = np.hstack([Utility.meanstdmaxmin(xy_prev_order_pred), Utility.percentileEmbedding(xy_prev_order_pred)])

            all_vicinity = x_vicinity.union(y_vicinity)
            all_vicinity.discard(x)
            all_vicinity.discard(y)
            one_order_ci = []
            for z in all_vicinity:
                _, severity = cit.ConditionalIndependenceTest(x, y, [z])
                one_order_ci.append(min(10, severity))
            
            feature_dict["residual of conditional dependencies"] = np.array([xy_severity - min(one_order_ci)]) if len(one_order_ci) != 0 else np.array([.0])
            
            feature_dict["k-order conditional dependencies"] = self._get_ci_features(x, y, all_vicinity, 1, cit)  # here seem not need new function(just one_order_ci)!!!!                
            
            feature_dict["mutual information"] = mutual_info_classif(dataset.get_data_by_index(x).reshape(-1, 1), dataset.get_data_by_index(y))
            

            all_samples.append((label, np.hstack([feature_dict[key] for key in self.feature_name_list]), (x, y)))
        
        with open(cit_path, "wb") as fp:
            pickle.dump(cit, fp)
        with open(samples_in_path, 'wb') as fp:
            pickle.dump(all_samples, fp)
        


    def export_higher_order(self, parm):
        order, skip_edges, parameter = parm
        truth, dataset, est_graph, cit, feature_path, cit_path, onedataset = self._prepare_data_and_model(parameter, order)
        prev_order_pred = {}
        for sample in onedataset:  # onedataset is : [predict, edge] or ['skip', edge](we not have currently)
            v1, v2 = sample[1]
            est_graph.add_undi_edge(v1, v2)
            prev_order_pred[sample[1]] = sample[0]
            
        # generate edge-wise feature
        data = []
        for edge in est_graph.UndirectedEdges:
            x, y = edge
            if not self._determine_training_sample(x, y):
                continue
            feature_dict = {}
            label = self._assign_label(truth, x, y)

            # calculate structural features
            x_vicinity, y_vicinity, x_y_prev_order_pred, structural_features_dict = self._get_local_structural_information(est_graph, x, y, prev_order_pred=prev_order_pred)
            feature_dict["degrees of target nodes"] = structural_features_dict["sparsity_feature"]
            feature_dict["density"] = structural_features_dict["overlapping_feature"]
            feature_dict["superiority"] = structural_features_dict["vicinity_relative_feature"]
            feature_dict["previous order prediction"] = x_y_prev_order_pred

            # find out whether we skip this edge, skip means the there is a super high probability that this edge exists, so later we won't train on this edge, but consider this as identified
            # we skip edges based on three conditions independent in subgraph given certain condition; model predict high prob.; already skipped
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
            prev_order_ci = [(i[0][2], i[1][1]) for i in cit.DumpCICache(x, y, order - 1) if i[0][2].issubset(all_vicinity)]  # [((x, y, S), (pValue, severity)), ...]  --> [(S, serverity)]
            prev_order_ci.sort(key=lambda x: -x[1])
            if len(prev_order_ci) > 10:
                prev_order_ci = prev_order_ci[:10]
            candidates = set()
            for i in prev_order_ci:
                candidates = candidates.union(i[0])
            monotonic_residual = []
            for ci in prev_order_ci:
                curr_cond_set, severity = set(ci[0]), ci[1]
                min_severity = severity
                for candidate in candidates:
                    if candidate in curr_cond_set:
                        continue
                    _, new_severity = cit.ConditionalIndependenceTest(x, y, curr_cond_set.union({candidate}))
                    min_severity = min(new_severity, min_severity)
                monotonic_residual.append(severity - min_severity)
            feature_dict["residual of conditional dependencies"] = np.hstack([Utility.meanstdmaxmin(monotonic_residual), Utility.percentileEmbedding(monotonic_residual, k=20)])
            feature_dict["k-order conditional dependencies"] = self._get_ci_features(x, y, all_vicinity, order, cit)

            # calculate mutual information between two discrete variables
            feature_dict["mutual information"] = mutual_info_classif(dataset.get_data_by_index(x).reshape(-1, 1), dataset.get_data_by_index(y))

            # if self.augmented_data and self.training_target == "skeleton":
            #     feature_dict["intervention_target"] = np.array([self._get_intervention_target_mark(x), self._get_intervention_target_mark(y)])
            #     if "intervention_target" not in self.feature_name_list:
            #         self.feature_name_list.append("intervention_target")

            # if self.augmented_data and self.training_target == "augmented_data":
            #     feature_dict["intervention_target"] = np.array(self._get_edge_type(x, y))
            #     if "intervention_target" not in self.feature_name_list:
            #         self.feature_name_list.append("intervention_target")

            data.append((label, np.hstack([feature_dict[key] for key in self.feature_name_list]), (x, y)))
            
        with open(feature_path, 'wb') as fp:
            pickle.dump(data, fp)
        with open(cit_path, "wb") as fp:
            pickle.dump(cit, fp)
    

    @staticmethod
    def _prepare_current_order_data(self, file_name, is_sibling, order):
        """ prepare data for feature extraction """
        
        if is_sibling:
            graph_path = f"{self.train_dir}/{file_name}.txt"
            dataset_path = f"{self.train_dir}/{file_name}.npy"
            cit_path = f"{self.cit_dir}/{file_name}.pkl"
            samples_in_path = f"{self.feature_dir}/{file_name}--{self.order_name[order]}-in.pkl"
            if order > 1:
                samples_out_path = f"{self.feature_dir}/{file_name}--{self.order_name[order-1]}-out.pkl"
        else:
            graph_path = f"{self.infer_dir}/aug_graphs/{self.benchmark_name}_aug_graph.txt"
            dataset_path = f"{self.infer_dir}/aug_samples/{self.benchmark_name}_aug_dataset.npy"
            cit_path = f"{self.cit_dir}/infer.pkl"
            samples_in_path = f"{self.feature_dir}/infer--{self.order_name[order]}-in.pkl"
            if order > 1:
                samples_out_path = f"{self.feature_dir}/infer--{self.order_name[order-1]}-out.pkl"
        
        truth_graph = DiGraph(graph_path)
        dataset = Dataset(dataset_path)
        estimate_graph = MixedGraph(numberOfNodes=dataset.VarCount)

        if os.path.exists(cit_path):
            with open(cit_path, "rb") as f:
                cit = pickle.load(f)
        else:
            cit = CITester(dataset.IndexedDataT, maxCountOfSepsets=50)
        
        if order > 1:
            with open(samples_out_path, "rb") as f:
                pre_predictions = pickle.load(f)
        else:
            pre_predictions = None
        
        return truth_graph, dataset, estimate_graph, cit, cit_path, samples_in_path, pre_predictions
    

    @staticmethod
    def _d_separation(self, x, y, vicinity, cit, threshold=None):
        vicinity.discard(x)
        vicinity.discard(y)
        if not threshold:
            threshold = 1 - cit.ConfidenceLevel
        for z in vicinity:
            pval, _ = cit.ConditionalIndependenceTest(x, y, [z])
            if pval > threshold:
                return True
        return False
    

    @staticmethod
    def _assign_label(truth, x, y):
        """ assign label based on the truth graph"""
        return 1 if truth.is_adjacent(x, y) else 0
    

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
        sparsity_feature = np.array([len(x_vicinity), len(y_vicinity)]) if len(x_vicinity) < len(y_vicinity) else np.array([len(y_vicinity), len(x_vicinity)])

        # calculate density feature (overlapping ratio)
        common_vicinity = x_vicinity.intersection(y_vicinity)
        overlapping_feature = np.array([len(common_vicinity) / min(len(x_vicinity), len(y_vicinity))])
        
        # superiority. get the edge prediction of x and y from the previous model, then calculates the superiority of this egde over other neighbouring edges
        x_count, y_count = 0, 0
        if higher_order:
            xy_prev_order_pred = prev_order_pred[(x, y)] if (x, y) in prev_order_pred else prev_order_pred[(y, x)]
            for z in x_vicinity:
                pred = prev_order_pred[(x, z)] if (x, z) in prev_order_pred else prev_order_pred[(z, x)]
                if not isinstance(xy_prev_order_pred, str) and not isinstance(pred, str) and xy_prev_order_pred > pred:
                    x_count += 1
            for z in y_vicinity:
                pred = prev_order_pred[(y, z)] if (y, z) in prev_order_pred else prev_order_pred[(z, y)]
                if not isinstance(xy_prev_order_pred, str) and not isinstance(pred, str) and xy_prev_order_pred > pred:
                    y_count += 1
        else:
            _, xy_severity = cit.ConditionalIndependenceTest(x, y)
            for z in x_vicinity:
                _, zx_severity = cit.ConditionalIndependenceTest(x, z)
                x_count += xy_severity > zx_severity
            for z in y_vicinity:
                _, zy_severity = cit.ConditionalIndependenceTest(y, z)
                y_count += xy_severity > zy_severity
        x_ratio, y_ratio = x_count / len(x_vicinity), y_count / len(y_vicinity)
        vicinity_relative_feature = np.array([x_ratio, y_ratio]) if x_ratio < y_ratio else np.array([y_ratio, x_ratio])
        
        structural_features_dict = {
            "sparsity_feature": sparsity_feature,
            "overlapping_feature": overlapping_feature,
            "vicinity_relative_feature": vicinity_relative_feature
        }
        edge_information = xy_prev_order_pred if higher_order else xy_severity

        return x_vicinity, y_vicinity, edge_information, structural_features_dict
    

    @staticmethod
    def _get_ci_features(x, y, all_vicinity, order, cit, k=20):
        # this min value is a hyperparameter which I keep the original value for different order
        min_value = 10 if order == 1 else 5
        order_selective_ci = []
        for cond_set in combinations(all_vicinity, order):
            _, severity = cit.ConditionalIndependenceTest(x, y, cond_set)
            order_selective_ci.append(min(min_value, severity))
        ci_feature = np.hstack([Utility.meanstdmaxmin(order_selective_ci), Utility.percentileEmbedding(order_selective_ci, k=k)])
        return ci_feature
    

    def _get_intervention_target_mark(self, node):
        # this is meant to return the intervention target of a node
        if not self.augmented_data:
            raise NotImplementedError
        if node < self.node_number:
            return node + self.node_number
        else:
            return -1
    

    def _get_edge_type(self, x, y):
        if x <= self.node_number and y <= self.node_number:
            return [1,0,0]
        elif x >= self.node_number and y >= self.node_number:
            return [0,0,1]
        else:
            return [0,1,0]
    
    def _is_sys_edge(self, x, y):
        if x <= self.node_number and y <= self.node_number:
            return True
        elif x >= self.node_number and y >= self.node_number:
            return False

    def _skip_training_edge(self, x, y):
        # sys->sys / env->env / sys->env / env->sys
        """ this function determines if a node pair should be included in training data """
        if self.augmented_data:
            if self.training_target == "augmented_data" and x >= self.node_number and y >= self.node_number: # exclude [env-env], only for [sys-sys, sys-env]
                return False
            elif self.training_target == "skeleton" and (x >= self.node_number or y >= self.node_number): # exclude [sys-env, env-env], only for [sys-sys]
                return False
            elif self.training_target == "intervention_target" and not ((x < self.node_number) ^ (y < self.node_number)):  # exclude [sys-sys, env-env], only for [sys-env]
                return False
        return True
    