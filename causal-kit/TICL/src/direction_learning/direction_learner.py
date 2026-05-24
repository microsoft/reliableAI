import os
import pickle
import numpy as np
import xgboost as xgb
from base.pdag import MixedGraph
from direction_learning.modify_direction_feature_extractor import DirectionFeatureExtractor
from tools.metric import get_compared_components, metric_skeleton_level, metric_target_level, metric_cpdag_level


class DirectionTrainer:
    def __init__(self, args):
        self._unpack_parameters(args)
        self.train_dir = f"../datasets/experiment/siblings/{self.exp_number}/datasets/{self.benchmark_name}"
        self.infer_dir = f"../datasets/experiment/original/{self.exp_number}"
        self.feature_dir = f"../datasets/experiment/siblings/{self.exp_number}/ut-features/{self.benchmark_name}"
        self.cpdag_dir = f"../datasets/experiment/siblings/{self.exp_number}/pred_cpdag"
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.cpdag_dir, exist_ok=True)
        self.infer_data_path = f"{self.infer_dir}/aug_samples/{self.benchmark_name}_aug_dataset.npy"
        self.infer_skeleton_path = f"../datasets/experiment/siblings/{self.exp_number}/pred_skeleton/predict_{self.benchmark_name}_graph.txt"
        self.clf = None
        self.cpdag = None
        self.feature_extractor = DirectionFeatureExtractor(self.train_dir, self.infer_dir, self.feature_dir)
    
    
    def process_pipeline(self, logger):
        self.feature_extractor.get_trainsets(logger)
        # self.train(logger)
        self.clf = xgb.XGBClassifier()
        self.clf.load_model('../orientation_learner.model')
        self.infer(logger)
        

    def _unpack_parameters(self, args):
        self.benchmark_name = args.benchmark_name
        self.exp_number = args.exp_number
        self.intervention_size = args.intervention_size
        self.edge_threshold = args.edge_threshold
        self.unknown_type = args.unknown_type
    

    def train(self, logger):
        all_feature_file = [file_name for file_name in os.listdir(self.feature_dir)]
        for idx, feature_file in enumerate(all_feature_file):  # TODO need speed up!
            with open(os.path.join(self.feature_dir, feature_file), "rb") as fp:
                feature, label = pickle.load(fp)
            if len(label) == 0:
                continue
            train_features = np.concatenate((train_features, feature[:, 3:])) if idx!=0 else feature[:, 3:]
            train_labels = np.concatenate((train_labels, label)) if idx!=0 else label
        
        neg_len, pos_len = np.count_nonzero(train_labels == 0), np.count_nonzero(train_labels == 1)
        logger.info(f"Start training: training data shape: {train_features.shape} \n Positive samples: {pos_len}, Negative samples: {neg_len} \n")
        self.clf = xgb.XGBClassifier()
        self.clf.fit(train_features, train_labels)
    

    def infer(self, logger):
        def _conflict_va_wins(va, vb, sa, sb):
            if va[0] == vb[1] or va[0] == vb[2]:
                if va[1] == vb[0] or va[2] == vb[0]:
                    return sa > sb
            return True
        
        indexed_features = self.feature_extractor.get_all_tfork_features(self.infer_data_path, self.infer_skeleton_path)
        tfork_indexs, tfork_features = indexed_features[:, :3].astype(int), indexed_features[:, 3:]
        pred_scores = self.clf.predict_proba(tfork_features)[:, 1]
        is_vstrucs = pred_scores >= self.edge_threshold
        isv_TXYs = list(map(tuple, tfork_indexs[is_vstrucs]))
        isv_scores = pred_scores[is_vstrucs]
        vstrucs_candidates = list(zip(isv_TXYs, isv_scores))
        rmlists = set()
        for (va, sa) in vstrucs_candidates:
            flag = False
            for (vb, sb) in vstrucs_candidates:
                if not _conflict_va_wins(va, vb, sa, sb):
                    flag = True
                    break
            if flag:
                rmlists.add(va)
        
        skeleton = MixedGraph(graph_path=self.infer_skeleton_path)
        pdag = MixedGraph(nodeIDs=skeleton.NodeIDs)
        
        for ((j, i, k), _) in vstrucs_candidates:
            if (j, i, k) not in rmlists:
                pdag.add_di_edge(i, j)
                pdag.add_di_edge(k, j)
        for (fromnode, tonode) in skeleton.UndirectedEdges:
            if not (pdag.has_di_edge(fromnode, tonode) or pdag.has_di_edge(tonode, fromnode)):
                pdag.add_undi_edge(fromnode, tonode)
        
        
        sys_size = len(skeleton.NodeIDs) - self.intervention_size  # apply jci
        for (x, y) in skeleton.UndirectedEdges:
            if x >= sys_size:
                pdag.add_di_edge(x, y)
            elif y >= sys_size:
                pdag.add_di_edge(y, x)
                
        
        pdag.apply_meek_rules()
        self.icpdag = pdag.getAdjacencyMatrix()
        
        pred_graph_path = f"{self.cpdag_dir}/predict_{self.benchmark_name}_cpdag.txt"
        with open(pred_graph_path, 'wb') as f:
            np.savetxt(f, self.icpdag, fmt='%i')
        
        aug_graph_path = f"{self.infer_dir}/aug_graphs/{self.benchmark_name}_aug_graph.txt"
        
        pred_I_SKELETON, pred_I_TARGET, pred_I_CPDAG = get_compared_components(pred_graph_path, self.intervention_size, real=False)
        targ_I_SKELETON, targ_I_TARGET, targ_I_CPDAG = get_compared_components(aug_graph_path, self.intervention_size, real=True)

        mt_skeleton = metric_skeleton_level(pred_I_SKELETON, targ_I_SKELETON)
        mt_target = metric_target_level(pred_I_TARGET, targ_I_TARGET)
        mt_cpdag = metric_cpdag_level(pred_I_CPDAG, targ_I_CPDAG)

        logger.info(f"performance of cpdag & intervention targets: \n {mt_skeleton} \n {mt_target} \n {mt_cpdag} \n")

        