import os
import time
import pickle
import numpy as np
from itertools import combinations
from sklearn.feature_selection import mutual_info_classif

from pathos.multiprocessing import Pool, cpu_count
from base.dag import DiGraph
from base.pdag import MixedGraph
from base.dataloader import Dataset
from base.citester import CITester
from tools.utils import meanstdmaxmin, percentileEmbedding
from tools.metric import get_compared_components, metric_skeleton_level, metric_target_level


class SkeletonFeatureExtractor:
    def __init__(self, benchmark_name, train_dir, infer_dir, feature_dir, cit_dir, contain_env, intervention_size):
        self.benchmark_name = benchmark_name
        self.train_dir = train_dir
        self.infer_dir = infer_dir
        self.feature_dir = feature_dir
        self.cit_dir = cit_dir
        self.contain_env = contain_env
        self.intervention_size = intervention_size
        self.order_name = ['zero', 'first', 'second', 'third', 'fourth']
        self.feature_name_list = ["relative superiority", "absolute superiority", "degrees of target nodes", "density", "k-order conditional dependencies", "residual of conditional dependencies"]
        self.all_datasets = [(file_name.replace(".npy", ""), True) for file_name in os.listdir(self.train_dir) if file_name.endswith(".npy")] + [(self.benchmark_name, False)]
    
    
    def order_extractor(self, order, logger):
        logger.info("#" * 100)
        logger.info(f"start {order} order feature extraction...")
        order_num = self.order_name.index(order)
        order_fun = self.export_first_order if order_num == 1 else self.export_higher_order
        parameters = [(item1, item2, order_num) for item1, item2 in self.all_datasets]

        start_time = time.time()
        with Pool(cpu_count()) as p:
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
            
            if pval < thres:  # if pValue < thres, reject H0, which means x and y are dependent (*** Here, equivalent to using Marginal Independence Test first ***)
                estimate_graph.add_undi_edge(x, y)
                zero_order_pred[edge] = min(10, severity)
                
        
        if not is_sibling:
            aug_graph_path = f"{self.infer_dir}/aug_graphs/{self.benchmark_name}_aug_graph.txt"
            zero_pred_graph_path = f"{self.train_dir}/../../pred_order_skeleton/pred_{self.benchmark_name}_0_graph.txt"
            zero_predict_adj = estimate_graph.getAdjacencyMatrix()
            with open(zero_pred_graph_path, 'wb') as f:
                np.savetxt(f, zero_predict_adj, fmt='%i')
            
            pred_I_SKELETON, pred_I_TARGET, _ = get_compared_components(zero_pred_graph_path, self.intervention_size, real=False)
            targ_I_SKELETON, targ_I_TARGET, _ = get_compared_components(aug_graph_path, self.intervention_size, real=True)

            mt_skeleton = metric_skeleton_level(pred_I_SKELETON, targ_I_SKELETON)
            mt_target = metric_target_level(pred_I_TARGET, targ_I_TARGET)
            print(f"zero order model performance of skeleton & intervention targets: \n {mt_skeleton} \n {mt_target} \n")
        
        # generate edge-wise feature
        all_samples = []
        for edge in estimate_graph.UndirectedEdges:
            x, y = edge
            feature_dict = {}
            label = self._assign_label(truth_graph, x, y)
            
            # calculate local structural features
            x_vicinity, y_vicinity, x_y_prev_order_pred, structural_features_dict = self._get_local_structural_information(estimate_graph, x, y, zero_order_pred)
            feature_dict["relative superiority"] = structural_features_dict["vicinity_relative_feature"]
            feature_dict["absolute superiority"] = structural_features_dict["vicinity_absolute_feature"]
            feature_dict["degrees of target nodes"] = structural_features_dict["sparsity_feature"]
            feature_dict["density"] = structural_features_dict["overlapping_feature"]
            
            # all_vicinity = x_vicinity.intersection(y_vicinity) # prune strategy 1
            all_vicinity = estimate_graph.getNeighbor(x).union(estimate_graph.getNeighbor(y))
            all_vicinity.discard(x)
            all_vicinity.discard(y)

            # order_selective_ci = []
            # for z in all_vicinity:
            #     _, new_severity = cit.ConditionalIndependenceTest(x, y, [z])
            #     order_selective_ci.append(min(10, new_severity))
            
            order_selective_ci = []
            x_vicinity.discard(y)
            y_vicinity.discard(x)
            for z in x_vicinity:
                _, new_severity = cit.ConditionalIndependenceTest(x, y, [z])
                order_selective_ci.append(min(10, new_severity))
            for z in y_vicinity:
                _, new_severity = cit.ConditionalIndependenceTest(x, y, [z])
                order_selective_ci.append(min(10, new_severity))
            
            feature_dict["k-order conditional dependencies"] = np.hstack([meanstdmaxmin(order_selective_ci), percentileEmbedding(order_selective_ci, k=20)])
            feature_dict["residual of conditional dependencies"] = np.array([x_y_prev_order_pred - min(order_selective_ci)]) if len(order_selective_ci) != 0 else np.array([.0])

            all_samples.append(((x, y), np.hstack([feature_dict[key] for key in self.feature_name_list]), label))
        
        with open(cit_path, "wb") as fp:
            pickle.dump(cit, fp)
        with open(samples_in_path, 'wb') as fp:
            pickle.dump(all_samples, fp)
    

    def export_higher_order(self, parm):
        file_name, is_sibling, order = parm
        truth_graph, dataset, estimate_graph, cit, cit_path, samples_in_path, pre_predictions = self._prepare_current_order_data(file_name, is_sibling, order)
        prev_order_pred = {}
        for edge, predict in pre_predictions:  # pre_predictions is : [predict, edge] or ['skip', edge](we not have currently)
            x, y = edge
            estimate_graph.add_undi_edge(x, y)
            prev_order_pred[edge] = predict
            
        
        # generate edge-wise feature
        all_samples = []
        for edge in estimate_graph.UndirectedEdges:
            x, y = edge
            if self._skip_training_edge(x, y):
                continue
            feature_dict = {}
            label = self._assign_label(truth_graph, x, y)
            
            # calculate structural features
            x_vicinity, y_vicinity, x_y_prev_order_pred, structural_features_dict = self._get_local_structural_information(estimate_graph, x, y, prev_order_pred)
            feature_dict["relative superiority"] = structural_features_dict["vicinity_relative_feature"]
            feature_dict["absolute superiority"] = structural_features_dict["vicinity_absolute_feature"]
            feature_dict["degrees of target nodes"] = structural_features_dict["sparsity_feature"]
            feature_dict["density"] = structural_features_dict["overlapping_feature"]
            
            # find out whether we skip this edge, skip means the there is a super high probability that this edge exists, so later we won't train on this edge, but consider this as identified
            # we skip edges based on three conditions independent in subgraph given certain condition; model predict high prob.; already skipped
            skip_edges = False
            if skip_edges:
                if min(structural_features_dict["sparsity_feature"]) < (order + 1) and not isinstance(x_y_prev_order_pred, str):
                    min_vicinity = x_vicinity if len(x_vicinity) < len(y_vicinity) else y_vicinity
                    min_vicinity.discard(x)
                    min_vicinity.discard(y)
                    pval, _ = cit.ConditionalIndependenceTest(x, y, min_vicinity)
                    if pval < .1:
                        all_samples.append((label, "skip", (x, y)))
                    continue
                if not isinstance(x_y_prev_order_pred, str) and x_y_prev_order_pred > .95:
                    all_samples.append((label, "skip", (x, y)))
                    continue
                if isinstance(x_y_prev_order_pred, str):
                    all_samples.append((label, "skip", (x, y)))
                    continue
            
            # all_vicinity = x_vicinity.union(y_vicinity)  # prune strategy 2
            # all_vicinity.discard(x)
            # all_vicinity.discard(y)
            # order_selective_ci = []
            # for cond_set in combinations(all_vicinity, order):
            #     _, new_severity = cit.ConditionalIndependenceTest(x, y, cond_set)
            #     order_selective_ci.append(min(5, new_severity))

            order_selective_ci = []
            x_vicinity.discard(y)
            y_vicinity.discard(x)
            for cond_set in combinations(x_vicinity, order):
                _, new_severity = cit.ConditionalIndependenceTest(x, y, cond_set)
                order_selective_ci.append(min(5, new_severity))
            for cond_set in combinations(y_vicinity, order):
                _, new_severity = cit.ConditionalIndependenceTest(x, y, cond_set)
                order_selective_ci.append(min(5, new_severity))

            feature_dict["k-order conditional dependencies"] = np.hstack([meanstdmaxmin(order_selective_ci), percentileEmbedding(order_selective_ci, k=20)])
            feature_dict["residual of conditional dependencies"] = np.array([x_y_prev_order_pred - min(order_selective_ci)]) if len(order_selective_ci) != 0 else np.array([.0])

            all_samples.append(((x, y), np.hstack([feature_dict[key] for key in self.feature_name_list]), label))
        
        with open(cit_path, "wb") as fp:
            pickle.dump(cit, fp)    
        with open(samples_in_path, 'wb') as fp:
            pickle.dump(all_samples, fp)
    
    
    def _prepare_current_order_data(self, file_name, is_sibling, order):
        """ prepare data for feature extraction """
        
        if is_sibling:
            graph_path = f"{self.train_dir}/{file_name}.txt"
            dataset_path = f"{self.train_dir}/{file_name}.npy"
            cit_path = f"{self.cit_dir}/{file_name}.pkl"
            samples_in_path = f"{self.feature_dir}/{file_name}-{self.order_name[order]}-in.pkl"
            if order > 1:
                samples_out_path = f"{self.feature_dir}/{file_name}-{self.order_name[order-1]}-out.pkl"
        else:
            graph_path = f"{self.infer_dir}/aug_graphs/{self.benchmark_name}_aug_graph.txt"
            dataset_path = f"{self.infer_dir}/aug_samples/{self.benchmark_name}_aug_dataset.npy"
            cit_path = f"{self.cit_dir}/infer.pkl"
            samples_in_path = f"{self.feature_dir}/infer-{self.order_name[order]}-in.pkl"
            if order > 1:
                samples_out_path = f"{self.feature_dir}/infer-{self.order_name[order-1]}-out.pkl"
        
        truth_graph = DiGraph(graph_path)
        dataset = Dataset(dataset_path)
        estimate_graph = MixedGraph(numberOfNodes=dataset.VarCount)
        self.observation_size = dataset.VarCount - self.intervention_size
        
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
    def _d_separation(x, y, vicinity, cit, threshold=None):
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
    def _get_local_structural_information(est_graph, x, y, prev_order_pred):
        """
        Args:
            est_graph: estimated graph
            x: node x
            y: nody y
            higher_order: if this is used for higher order
            prev_order_pred: edge prediction from the previous stage
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

        xy_prev_order_pred = prev_order_pred[(x, y)] if (x, y) in prev_order_pred else prev_order_pred[(y, x)]
        abs_prob = [xy_prev_order_pred]
        for z in x_vicinity:
            pred = prev_order_pred[(x, z)] if (x, z) in prev_order_pred else prev_order_pred[(z, x)]
            abs_prob.extend([pred])
            if not isinstance(xy_prev_order_pred, str) and not isinstance(pred, str) and xy_prev_order_pred > pred:
                x_count += 1
        for z in y_vicinity:
            pred = prev_order_pred[(y, z)] if (y, z) in prev_order_pred else prev_order_pred[(z, y)]
            abs_prob.extend([pred])
            if not isinstance(xy_prev_order_pred, str) and not isinstance(pred, str) and xy_prev_order_pred > pred:
                y_count += 1
        
        x_ratio, y_ratio = x_count / len(x_vicinity), y_count / len(y_vicinity)
        vicinity_relative_feature = np.array([x_ratio, y_ratio]) if x_ratio < y_ratio else np.array([y_ratio, x_ratio])
        vicinity_absolute_feature = np.hstack([meanstdmaxmin(abs_prob), percentileEmbedding(abs_prob)])
        structural_features_dict = {
            "sparsity_feature": sparsity_feature,
            "overlapping_feature": overlapping_feature,
            "vicinity_relative_feature": vicinity_relative_feature,
            "vicinity_absolute_feature": vicinity_absolute_feature
        }

        return x_vicinity, y_vicinity, xy_prev_order_pred, structural_features_dict
    

    @staticmethod
    def _get_ci_features(x, y, all_vicinity, order, cit, k=20):
        # this min value is a hyperparameter which I keep the original value for different order
        min_value = 10 if order == 1 else 5
        order_selective_ci = []
        for cond_set in combinations(all_vicinity, order):
            _, severity = cit.ConditionalIndependenceTest(x, y, cond_set)
            order_selective_ci.append(min(min_value, severity))
        ci_feature = np.hstack([meanstdmaxmin(order_selective_ci), percentileEmbedding(order_selective_ci, k=k)])
        return ci_feature
    
    def _skip_training_edge(self, x, y):
        """ this function determines if a node pair should be included in training data """
        # (sys->sys / env->env / sys->env / env->sys), here we skip env-env
        return x >= self.observation_size and y >= self.observation_size


    def _get_intervention_target_mark(self, node):
        # this is meant to return the intervention target of a node
        # TODO: for now, since we are work on the single intervention set, just add an offset
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
    