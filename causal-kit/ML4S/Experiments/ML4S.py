# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import xgboost as xgb
from sklearn import linear_model
import numpy as np
import pickle, sys, os, copy, random, joblib
from tqdm import tqdm
from Tools import Graph, CITester, Dataset
from Tools.Utility import d_sep, get_logger
from Experiments.FeatureExtractor.GTExtractFeature import FeatureExtractor
import Experiments.FeatureExtractor.GTScalabelExtractFeature as Scalable
import argparse
import yaml

sys.path.append(os.path.dirname(os.getcwd()))


class Trainer:
    def __init__(self, bname, dir_name, feature_folder, model_folder, logger):
        self.bname, self.dir_name, self.feature_folder, self.model_folder = \
            bname, dir_name, feature_folder, model_folder
        self.feature_dir = f'{self.feature_folder}/{self.dir_name}/{self.bname}/'
        self.model_dir = f"{self.model_folder}/{self.dir_name}"
        self.logger = logger
        self.model = None

    def train(self, order_tag, model="xgboost"):
        self.logger.info(f'start {order_tag} order training...')
        os.makedirs(self.model_dir, exist_ok=True)
        all = [_ for _ in os.listdir(self.feature_dir) if _.endswith(f"{order_tag}-in.pkl") and "infer" not in _]

        if order_tag == "first":
            pos_features = []
            neg_features = []
            pos_labels = []
            for feature_file in tqdm(all):
                with open(os.path.join(self.feature_dir, feature_file), "rb") as fp:
                    onedataset = pickle.load(fp)
                for sample in onedataset:
                    if isinstance(sample[1], int) or isinstance(sample[1], str):
                        continue
                    if sample[0] == 0:
                        neg_features.append(sample[1])
                    else:
                        pos_features.append(sample[1])
                        pos_labels.append(sample[0])
            random.shuffle(neg_features)
            if len(neg_features) > len(pos_features):
                neg_features = neg_features[:len(pos_features)]
            train_labels = np.array(pos_labels + [0 for _ in range(len(neg_features))])
            train_features = np.array(pos_features + neg_features)
        else:
            data = {}
            for feature_file in tqdm(all):
                with open(os.path.join(self.feature_dir, feature_file), "rb") as fp:
                    onedataset = pickle.load(fp)
                for sample in onedataset:
                    if not isinstance(sample[1], np.ndarray):
                        continue
                    if sample[0] in data:
                        data[sample[0]].append(sample[1])

                    else:
                        data[sample[0]] = [sample[1]]
            if len(data) == 0:
                return
            # sample_num = sorted([len(data[label]) for label in data])[1]
            train_data = []
            for label in data:
                random.shuffle(data[label])
                # if len(data[label]) > sample_num:
                #     train_data += [(feature, label) for feature in data[label][:sample_num]]
                # else:
                train_data += [(feature, label) for feature in data[label]]
            random.shuffle(train_data)

            train_labels = np.array([i[1] for i in train_data], dtype=float)
            train_features = np.array([i[0] for i in train_data], dtype=float)

        self.logger.info(
            f"{order_tag} order training: training data shape: {train_features.shape} Positive samples: {(train_labels == 1).sum()}")

        if model == "xgboost":
            clf = xgb.XGBRegressor()
            clf.fit(train_features, train_labels)
            clf.save_model(os.path.join(self.model_dir, f'xgboost_{self.bname}_{order_tag}.model'))
        else:
            clf = linear_model.BayesianRidge()
            clf.fit(train_features, train_labels)
            joblib.dump(clf, os.path.join(self.model_dir, f'ridge_{self.bname}_{order_tag}.model'))
        self.model = clf

    def _compute_metric(self, graph_path, y_pred, threshold, skipped_edges=[]):
        truth = Graph.DiGraph(graph_path)
        truth_skeleton = Graph.MixedGraph(numberOfNodes=len(truth.NodeIDs))
        for edge in truth.DirectedEdges:
            truth_skeleton.add_undi_edge(edge[0], edge[1])

        estimated_skeleton = Graph.MixedGraph(numberOfNodes=len(truth.NodeIDs))

        y_pred.sort(key=lambda x: -x[1])

        for pred in y_pred:
            if pred[1] > threshold:
                x, y = pred[0]
                estimated_skeleton.add_undi_edge(x, y)
        for edge in skipped_edges:
            estimated_skeleton.add_undi_edge(edge[1][0], edge[1][1])
        self.logger.info(f"#Input Edge: {len(y_pred) + len(skipped_edges)}, #Skipped Edge, "
                         f"{len(skipped_edges)}, #Remained Edge:, {len(estimated_skeleton.UndirectedEdges)}")

        precision = len(truth_skeleton.UndirectedEdges.intersection(estimated_skeleton.UndirectedEdges)) / max(
            len(estimated_skeleton.UndirectedEdges), 1)
        recall = len(truth_skeleton.UndirectedEdges.intersection(estimated_skeleton.UndirectedEdges)) / max(
            len(truth_skeleton.UndirectedEdges), 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

        return {"Dataset": self.bname, "F1": f1, "Precision": precision, "Recall": recall}

    def predict_skeleton(self, data):
        pass

    def infer(self, order_tag, threshold, model="xgboost", infer_only=False):
        self.logger.info(f'start {order_tag} order inference...')
        # TODO this loading is not needed if we save the model after training
        # if model == "xgboost":
        #     self.logger.info(f'model for inference is xgboost_{self.bname}_{order_tag}.model')
        #     if not os.path.exists(os.path.join(self.model_dir, f'xgboost_{self.bname}_{order_tag}.model')):
        #         self.logger.info("no model found")
        #         return
        #     clf = xgb.XGBRegressor()
        #     clf.load_model(os.path.join(self.model_dir, f'xgboost_{self.bname}_{order_tag}.model'))
        # else:
        #     self.logger.info(f'model for inference is ridge_{self.bname}_{order_tag}.model')
        #     clf = joblib.load(os.path.join(self.model_dir, f'ridge_{self.bname}_{order_tag}.model'))
        clf = self.model

        if infer_only:
            all = [feature_file for feature_file in os.listdir(self.feature_dir) if
                   feature_file.endswith(f"{order_tag}-in.pkl") and feature_file.startswith("infer")]
        else:
            all = [feature_file for feature_file in os.listdir(self.feature_dir) if feature_file.endswith(f"{order_tag}-in.pkl")]

        for npyfile in all:
            test_labels = []
            test_features = []
            edges = []
            skipped_edges = []
            with open(os.path.join(self.feature_dir, npyfile), "rb") as fp:
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
                y_pred = [(edges[idx], pred) for idx, pred in enumerate(clf.predict(test_features))]
            else:
                y_pred = []
            dataset_name = npyfile.replace(f"-{order_tag}-in.pkl", "")
            graph_path = f"../benchmarks/{self.bname}_graph.txt"

            # set up threshold for precision recall calculation
            if threshold == "mean":
                _threshold = np.array(y_pred)[:, 1].mean()
            elif threshold == "mean + 0.5 * std":
                _threshold = np.array(y_pred)[:, 1].mean() + .5 * np.array(y_pred)[:, 1].std()
            else:
                _threshold = threshold

            if dataset_name == "infer":  # True:#
                self.logger.info(self._compute_metric(graph_path, y_pred, _threshold, skipped_edges))
            # else:
            #     graph_path = f"../siblings/{self.bname}/{dataset_name}.txt"
            #     print(order_tag, ComputeMetric(dataset_name, graph_path, y_pred, threshold))

            data = [(test_labels[i], test_features[i], y_pred[i][1], y_pred[i][0]) for i in range(len(test_labels)) if
                    y_pred[i][1] > _threshold]
            data += [(e[0], "skip", "skip", e[1]) for e in skipped_edges]

            # save prediction result
            with open(os.path.join(self.feature_dir, f"{dataset_name}-{order_tag}-out.pkl"), 'wb') as fp:
                pickle.dump(data, fp)

    def backward(self, order_tag):
        # TODO not used?
        with open(os.path.join(self.feature_dir, f"infer-{order_tag}-out.pkl"), 'rb') as fp:
            onedataset = pickle.load(fp)
        graph_path = f"../benchmarks/{self.bname}_graph.txt"
        txtPath = f"../benchmarks/npy10000/{self.bname}.npy"

        truth = Graph.DiGraph(graph_path)
        truth_skeleton = Graph.MixedGraph(numberOfNodes=len(truth.NodeIDs))
        for edge in truth.DirectedEdges:
            truth_skeleton.add_undi_edge(edge[0], edge[1])
        dataset = Dataset.Dataset(txtPath)
        est_graph = Graph.MixedGraph(numberOfNodes=dataset.VarCount)

        cit = CITester.CITester(dataset.IndexedDataT, confidenceLevel=.95)

        for sample in onedataset:
            v1, v2 = sample[3]
            est_graph.add_undi_edge(v1, v2)
        for edge in copy.copy(est_graph.UndirectedEdges):
            x, y = edge
            if not est_graph.adjacent_in_mixed_graph(x, y):
                continue
            vicinity = est_graph.getNeighbor(x).union(est_graph.getNeighbor(y))
            if d_sep(x, y, vicinity, cit):
                est_graph.del_undi_edge(x, y)

        self.logger.info("Backward", est_graph.Compare(truth_skeleton))


'''
barley wo blip+data 60
second {'Dataset': 'barley', 'F1': 0.5340909090909092, 'Precision': 0.5108695652173914, 'Recall': 0.5595238095238095}
Backward {'F1': 0.5584415584415584, 'Precision': 0.6142857142857143, 'Recall': 0.5119047619047619}
'''


def run_ml4s(bname, data_folder, dir_name, feature_folder, cit_folder, model_folder, feature_name_list, logger):
    # initialize feature extractor
    feature_extractor = FeatureExtractor(bname, data_folder, dir_name,
                                         feature_folder, cit_folder, feature_name_list)

    # initialize trainer
    trainer = Trainer(bname, dir_name, feature_folder, model_folder, logger)

    # start training process for ML4S
    feature_extractor.first_order()
    trainer.train("first")
    trainer.infer("first", .75)
    feature_extractor.second_order()
    trainer.train("second")
    trainer.infer("second", .75)
    feature_extractor.third_order()
    trainer.train("third")
    trainer.infer("third", .75)
    try:
        feature_extractor.forth_order()
        trainer.train("forth")
        trainer.infer("forth", .75, infer_only=True)
    except:
        logger.info("No Forth Order")
    logger.info("\nTraining finished.")
    return trainer.model


def run_multi_ml4s(bname, data_folder, dir_name, feature_folder, cit_folder, model_folder, feature_name_list, logger):
    # TODO this run multiple ml4s process, with additional data generation stage using previous results
    # start first stage training
    run_ml4s(bname, data_folder, dir_name, feature_folder, cit_folder, model_folder, feature_name_list, logger)
    # generate data

    # start another stage training with generated data
    run_ml4s(bname, data_folder, dir_name, feature_folder, cit_folder, model_folder, feature_name_list, logger)


def main():
    # need backward
    # mildew, hailfinder (good to hv)
    small = [
        "alarm",
        "barley",
        "child",
        "insurance",
        "mildew",
        "water",
        "hailfinder",
        "hepar2",
        "win95pts",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--bfile", help="benchmark name file", default="../configs/benchmark.yml")
    parser.add_argument("--config", help="config file", default="../configs/ml4s.yml")
    parser.add_argument("--config_data_generation", help="config file for data generation",
                        default="../configs/generate_sibling.yml")
    args = parser.parse_args()
    with open(args.bfile, 'r') as file:
        benchmark_names = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.config, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.config_data_generation, 'r') as file:
        cfg_data = yaml.load(file, Loader=yaml.FullLoader)

    feature_folder = cfg['feature_folder']
    cit_folder = cfg['cit_folder']
    model_folder = cfg['model_folder']
    feature_name_list = cfg['feature_name_list']
    data_folder = cfg_data['data_folder']
    dir_name = cfg_data['dir_name']

    logger = get_logger(cfg, cfg_data)

    # loop over benchmarks
    for bname in benchmark_names["benchmarks"]:
        logger.info(f'current benchmark is {bname}')
        if bname in small:
            run_ml4s(bname, data_folder, dir_name, feature_folder, cit_folder, model_folder, feature_name_list, logger)
        else:
            # TODO fix later with class Trainer
            pass
            # Scalable.FirstOrder(bname)
            # Train("first")
            # Infer("first", .4)
            # Scalable.SecondOrder(bname)

            # Train("second")
            # Infer("second", .5)
            # Scalable.ThirdOrder(bname)
            # Train("third")
            # Infer("third", .6)
            # Scalable.Prune("third")
            # Scalable.ForthOrder(bname)
            # trainer.train("forth")
            # trainer.infer("forth", .75, infer_only=True)
            # Scalable.Prune("forth")
        # for i in range(50):
        #     print(i/50)
        #     Infer(bname, "third", i/50, infer_only=True)
        # Backward(bname, "forth")


if __name__ == "__main__":
    main()
