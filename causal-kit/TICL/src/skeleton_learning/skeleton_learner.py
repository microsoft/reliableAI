import os
import time
import random
import pickle
import numpy as np
import xgboost as xgb
from sklearn import linear_model
from pathos.multiprocessing import Pool, cpu_count
from skeleton_learning.modify_skeleton_feature_extractor import SkeletonFeatureExtractor
from tools.metric import get_compared_components, metric_skeleton_level, metric_target_level


class SkeletonTrainer:
    def __init__(self, args):
        self._unpack_parameters(args)
        self.train_dir = f"../datasets/experiment/siblings/{self.exp_number}/datasets/{self.benchmark_name}"
        self.infer_dir = f"../datasets/experiment/original/{self.exp_number}"
        self.feature_dir = f"../datasets/experiment/siblings/{self.exp_number}/features/{self.benchmark_name}"
        self.cit_dir = f"../datasets/experiment/siblings/{self.exp_number}/cits/{self.benchmark_name}"
        self.order_skeleton_dir = f"../datasets/experiment/siblings/{self.exp_number}/pred_order_skeleton"
        self.skeleton_dir = f"../datasets/experiment/siblings/{self.exp_number}/pred_skeleton"
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.cit_dir, exist_ok=True)
        os.makedirs(self.order_skeleton_dir, exist_ok=True)
        os.makedirs(self.skeleton_dir, exist_ok=True)
        self.clf = None
        self.skeleton = None
        self.feature_extractor = SkeletonFeatureExtractor(self.benchmark_name, self.train_dir, self.infer_dir, self.feature_dir, self.cit_dir, self.contain_env, self.intervention_size)
    
    
    def _unpack_parameters(self, args):
        self.benchmark_name = args.benchmark_name
        self.exp_number = args.exp_number
        self.skeleton_train_type = args.skeleton_train_type
        self.contain_env = args.contain_env
        self.decision_thres = args.decision_thres
        self.intervention_size = args.intervention_size
    
    
    def process_pipeline(self, logger):
        for order in ["first", "second", "third", "fourth"]:
            try:
                self.feature_extractor.order_extractor(order, logger)
                
                if self.skeleton_train_type == "together":
                    self.train_together(order, logger)
                    self.infer_together(order, logger)
                else:
                    self.train_separate(order)
                    self.infer_separate(order)
            except:
                logger.info(f"No {order} Order")
                break
        
        with open(f'{self.skeleton_dir}/predict_{self.benchmark_name}_graph.txt', 'wb') as f:
            np.savetxt(f, self.skeleton, fmt='%i')
        
        logger.info("Skeleton Stage finished!\n")
        
    
    
    def train_together(self, order, logger, model="xgboost"):
        
        logger.info(f'start {order} order training...')

        all_feature_file = [file_name for file_name in os.listdir(self.feature_dir) if file_name.endswith(f"{order}-in.pkl") and "infer" not in file_name]
        all_trainsets = {0:[], 1:[]}
        for feature_file in all_feature_file:
            with open(os.path.join(self.feature_dir, feature_file), "rb") as fp:
                all_samples = pickle.load(fp)
            for _, feature, label in all_samples:
                all_trainsets[label].append(feature)
        
        neg_len, pos_len = len(all_trainsets[0]), len(all_trainsets[1])
        
        # if neg_len < pos_len and neg_len > 100:
        #     all_trainsets[1] = all_trainsets[1][:neg_len]
        # elif pos_len < neg_len and pos_len > 100:
        #     all_trainsets[0] = all_trainsets[0][:pos_len]
        # neg_len, pos_len = len(all_trainsets[0]), len(all_trainsets[1])
        
        trainsets = [(feature, 0) for feature in all_trainsets[0]] + [(feature, 1) for feature in all_trainsets[1]]
        random.seed(42)
        random.shuffle(trainsets)
        train_features = np.array([sample[0] for sample in trainsets], dtype=float)
        train_labels = np.array([sample[1] for sample in trainsets], dtype=float)
        logger.info(f"{order} order training: training data shape: {train_features.shape} \n Positive samples: {pos_len}, Negative samples: {neg_len} \n")
        
        if model == "xgboost":
            self.clf = xgb.XGBClassifier()
            self.clf.fit(train_features, train_labels)
        
    def infer_together(self, order, logger):

        def infer_one(feature_file):
            
            with open(os.path.join(self.feature_dir, feature_file), "rb") as fp:
                all_samples = pickle.load(fp)
            
            test_features, test_labels, all_edges = [], [], []
            for edge, feature, label in all_samples:
                all_edges.append(edge)
                test_features.append(feature)
                test_labels.append(label)
            
            test_features = np.array(test_features)
            test_labels = np.array(test_labels)

            samples_out = [(all_edges[idx], pred) for idx, pred in enumerate(self.clf.predict(test_features)) if pred > self.decision_thres] if len(test_features) != 0 else []
            dataset_basename = feature_file.replace(f"-{order}-in.pkl", "")

            if dataset_basename == "infer":
                aug_graph_path = f"{self.infer_dir}/aug_graphs/{self.benchmark_name}_aug_graph.txt"
                order_pred_graph_path = f"{self.order_skeleton_dir}/pred_{self.benchmark_name}_{order}_graph.txt"
                predict_adj = self._get_adj_from_predict(samples_out, len(np.loadtxt(aug_graph_path)))
                self.skeleton = predict_adj
                with open(order_pred_graph_path, 'wb') as f:
                    np.savetxt(f, self.skeleton, fmt='%i')
                pred_I_SKELETON, pred_I_TARGET, _ = get_compared_components(order_pred_graph_path, self.intervention_size, real=False)
                targ_I_SKELETON, targ_I_TARGET, _ = get_compared_components(aug_graph_path, self.intervention_size, real=True)

                mt_skeleton = metric_skeleton_level(pred_I_SKELETON, targ_I_SKELETON)
                mt_target = metric_target_level(pred_I_TARGET, targ_I_TARGET)
                logger.info(f"{order} order model performance of skeleton & intervention targets: \n {mt_skeleton} \n {mt_target} \n")
            
            with open(os.path.join(self.feature_dir, f"{dataset_basename}-{order}-out.pkl"), 'wb') as fp:
                pickle.dump(samples_out, fp)
        
        
        logger.info(f'start {order} order inference...')
        
        if order == 'fourth':
            all_feature_file = [file_name for file_name in os.listdir(self.feature_dir) if file_name.endswith(f"{order}-in.pkl") and file_name.startswith("infer")]  # only infer for test
        else:
            all_feature_file = [file_name for file_name in os.listdir(self.feature_dir) if file_name.endswith(f"{order}-in.pkl")]  # infer for train & test
        
        for feature_file in all_feature_file:
            infer_one(feature_file)
    

    def train_separate(self, order_tag, model="xgboost"):
        print(f'start {order_tag} order training...')
        all_feature_file = [file_name for file_name in os.listdir(self.feature_path) if file_name.endswith(f"{order_tag}-in.pkl") and "infer" not in file_name]
        
        def _is_sys_edge(x, y):
            node_number = np.load(f'/home/ouc_asc01/chenwei/MSRA/iML4C-Wei/benchmarks_exp/exp_1/origin/npy10000/{self.benchmark_name}_aug_dataset.npy').shape[1] - 10
            if x <= node_number and y <= node_number:
                return True
            elif x >= node_number and y >= node_number:
                return False
        
        if order_tag == "first":
            data = {}
            for feature_file in tqdm(all_feature_file):
                with open(os.path.join(self.feature_path, feature_file), "rb") as fp:
                    onedataset = pickle.load(fp)
                for sample in onedataset:
                    if isinstance(sample[1], int) or isinstance(sample[1], str):
                        continue
                    if sample[0] in data:
                        data[sample[0]].append((sample[1], sample[2]))
                    else:
                        data[sample[0]] = [(sample[1], sample[2])]
            if len(data) == 0:
                return
            sys_train_data, env_train_data = [], []
            for label in data:
                random.shuffle(data[label])
                for content in data[label]:
                    feature, edge = content[0], content[1]
                    if _is_sys_edge(edge[0], edge[1]):
                        sys_train_data += [(feature, label)]
                    else:
                        env_train_data += [(feature, label)]
            random.shuffle(sys_train_data)
            random.shuffle(env_train_data)
            sys_train_labels = np.array([i[1] for i in sys_train_data], dtype=float)
            sys_train_features = np.array([i[0] for i in sys_train_data], dtype=float)
            env_train_labels = np.array([i[1] for i in env_train_data], dtype=float)
            env_train_features = np.array([i[0] for i in env_train_data], dtype=float)
        else:
            data = {}
            for feature_file in tqdm(all_feature_file):
                with open(os.path.join(self.feature_path, feature_file), "rb") as fp:
                    onedataset = pickle.load(fp)
                for sample in onedataset:
                    if not isinstance(sample[1], np.ndarray):
                        continue
                    if sample[0] in data:
                        data[sample[0]].append((sample[1], sample[2]))
                    else:
                        data[sample[0]] = [(sample[1], sample[2])]
            if len(data) == 0:
                return
            sys_train_data, env_train_data = [], []
            for label in data:
                random.shuffle(data[label])
                for content in data[label]:
                    feature, edge = content[0], content[1]
                    if _is_sys_edge(edge[0], edge[1]):
                        sys_train_data += [(feature, label)]
                    else:
                        env_train_data += [(feature, label)]
            random.shuffle(sys_train_data)
            random.shuffle(env_train_data)
            sys_train_labels = np.array([i[1] for i in sys_train_data], dtype=float)
            sys_train_features = np.array([i[0] for i in sys_train_data], dtype=float)
            env_train_labels = np.array([i[1] for i in env_train_data], dtype=float)
            env_train_features = np.array([i[0] for i in env_train_data], dtype=float)

        print(f"{order_tag} order training: training data shape: {sys_train_features.shape, env_train_features.shape} Positive samples: {(sys_train_labels == 1).sum(), (env_train_labels == 1).sum()}")

        if model == "xgboost":
            sys_clf = xgb.XGBClassifier()
            sys_clf.fit(sys_train_features, sys_train_labels)
            env_clf = xgb.XGBClassifier()
            env_clf.fit(env_train_features, env_train_labels)
        self.sys_model = sys_clf
        self.env_model = env_clf
    
    
    def infer_separate(self, order_tag, threshold, infer_only=False):
        print(f'start {order_tag} order inference...')
        sys_clf = self.sys_model
        env_clf = self.env_model
        
        def _is_sys_edge(x, y):
            node_number = np.load(f'../../datasets/experiment/exp_1/origin/npy10000/{self.benchmark_name}_aug_dataset.npy').shape[1] - 10
            if x <= node_number and y <= node_number:
                return True
            elif x >= node_number and y >= node_number:
                return False
            
        if infer_only:
            all = [file_name for file_name in os.listdir(self.feature_path) if file_name.endswith(f"{order_tag}-in.pkl") and file_name.startswith("infer")]
        else:
            all = [file_name for file_name in os.listdir(self.feature_path) if file_name.endswith(f"{order_tag}-in.pkl")]
        
        for npyfile in all:
            test_labels, test_features = [], []
            edges, skipped_edges = [], []
            with open(os.path.join(self.feature_path, npyfile), "rb") as fp:
                onedataset = pickle.load(fp)
            for sample in onedataset:
                if isinstance(sample[1], str):
                    skipped_edges.append((sample[0], sample[2]))
                    continue
                test_labels.append(sample[0])
                test_features.append(sample[1])
                edges.append(sample[2])
            
            test_labels = np.array(test_labels)
            test_features = np.array(test_features)
            if len(test_features) != 0:
                preds = []
                for idx, (x, y) in enumerate(edges):
                    if _is_sys_edge(x, y):
                        preds.append(sys_clf.predict(test_features[idx].reshape(1,-1)))
                    else:
                        preds.append(env_clf.predict(test_features[idx].reshape(1,-1)))
                y_pred = [(edges[idx], pred) for idx, pred in enumerate(preds)]
            else:
                y_pred = []
            dataset_name = npyfile.replace(f"-{order_tag}-in.pkl", "")
            graph_path = os.path.join(self.infer_path, f"graphs/{self.benchmark_name}_aug_graph.txt")

            # set up threshold for precision recall calculation
            if threshold == "mean":
                _threshold = np.array(y_pred)[:, 1].mean()
            elif threshold == "mean + 0.5 * std":
                _threshold = np.array(y_pred)[:, 1].mean() + .5 * np.array(y_pred)[:, 1].std()
            else:
                _threshold = threshold

            if dataset_name == "infer":  # True:#
                print(self._compute_metric(graph_path, y_pred, _threshold, skipped_edges))
                if self.augmented_data:
                    print(self._compute_metric(graph_path, y_pred, _threshold, skipped_edges, mode='system'))
                    # print(self._compute_metric(graph_path, y_pred, _threshold, skipped_edges, mode='environment'))
            
            data = [(test_labels[i], test_features[i], y_pred[i][1], y_pred[i][0]) for i in range(len(test_labels)) if y_pred[i][1] > _threshold]
            data += [(e[0], "skip", "skip", e[1]) for e in skipped_edges]

            # save prediction result
            with open(os.path.join(self.feature_path, f"{dataset_name}-{order_tag}-out.pkl"), 'wb') as fp:pickle.dump(data, fp)
    
    def _get_adj_from_predict(self, samples_out, var_count):
        adj_matrix = np.zeros((var_count, var_count), dtype=int)
        sys_size = var_count - self.intervention_size
        for edge, _ in samples_out:
            x, y = edge
            if (x > sys_size and y > sys_size):
                continue
            adj_matrix[x, y] = 1
            adj_matrix[y, x] = 1
        return adj_matrix