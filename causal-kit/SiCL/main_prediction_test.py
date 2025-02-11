# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import argparse
import yaml
import random

import torch

print(torch.cuda.is_available())
# torch.autograd.set_detect_anomaly(True)
import numpy as np

from ML4S_Tools.Dataset import ContinousDataset, DiscreteDataset
from ML4C_Tools.Utility import cal_score
from model.model import Model
from model.orientation_model import WholeModel, WholeNodewiseModel
from utils.criterions import get_scores, vstruc_get_scores, tri_cpdag_get_scores
from utils.tools import batched_dag_to_vstrucs, batched_skeleton_to_tforks, dag_to_dag, set_seed
from utils.graph import DiGraph, MixedGraph

# set_seed(1896)

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="config file",
                    default=f"configs/prediction_test.yml")
args = parser.parse_args()
sample_size = 1000
print(sample_size)
if1s = []
if1_all = []


def adjacency_to_triangle_matrix(adj_matrix):
    tri_matrix = np.triu(adj_matrix) - np.triu(adj_matrix.T)
    double_edges = np.logical_and(adj_matrix, adj_matrix.T)
    tri_matrix[double_edges] = 2
    return np.triu(tri_matrix)


def adjacency_to_triangle_matrix2(adj_matrix):
    tri_matrix = np.zeros_like(adj_matrix)
    n = adj_matrix.shape[0]

    for i in range(n):
        for j in range(i, n):
            if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 1:
                tri_matrix[i, j] = 2
            elif adj_matrix[i, j] == 1:
                tri_matrix[i, j] = 1
            elif adj_matrix[j, i] == 1:
                tri_matrix[i, j] = -1

    return tri_matrix


def test(skeleton_model, vstruc_model, dataset, device, name, batch_size=200, ensembles=1, vstruc_threshold=0.5,
         skeleton_threshold=0.5):
    total_count, noedge_count, novstruc_count = 0, 0, 0
    skeleton_model.eval()
    if vstruc_model:
        vstruc_model.eval()
    gf1s, gaucs, gauprcs, ghammings = [], [], [], []
    sf1s, saucs, sauprcs, shammings = [], [], [], []
    vf1s, vaucs, vauprcs, vhammings = [], [], [], []
    caccs, chammings = [], []
    ml4c_scoress = []
    final_ml4c_scoress = []
    threshold = skeleton_threshold
    for idx, (data, label) in enumerate(dataset):
        total_count += 1
        data = torch.tensor(data.IndexedDataT).transpose(1, 0).to(device)
        l = label
        if config['useMEC']:
            label = torch.tensor(dag_to_dag(label.IndexedDataT.transpose(-1, -2))).to(device)
        else:
            label = torch.tensor(label.IndexedDataT.transpose(-1, -2)).to(device)
        if label.max() == 0:
            noedge_count += 1
            continue
        label_cube_tforks, label_cube_vstrucs = batched_dag_to_vstrucs(label.cpu().unsqueeze(0).numpy())
        label_vstrucs_set = set(map(tuple, np.argwhere(label_cube_vstrucs[0])))
        if len(label_vstrucs_set) == 0:
            novstruc_count += 1
            continue
        num_of_nodes = len(label)
        data = data[:sample_size]
        label_digraph = DiGraph(label)
        label_cpdag = label_digraph.getCPDAG().getAdjacencyMatrix()
        tri_cpdag_label = adjacency_to_triangle_matrix(label_cpdag)
        # assert np.allclose(tri_cpdag_label, adjacency_to_triangle_matrix2(label_cpdag))
        data_size = len(data)
        index = 0
        skeletons_sum = 0
        vstrucs_sum = 0
        graphs_sum = 0
        for j in range(ensembles):
            shuffled_data = torch.randperm(data.size(0))
            data = data[shuffled_data]
            for i in range(0, data_size - batch_size, batch_size):
                inputs, targets = data[i: i + batch_size], label
                skeleton_outputs = skeleton_model(inputs.unsqueeze(0))
                if config['learning_mode'] == 'graph':
                    graphs_sum += skeleton_outputs['graph'].detach()
                else:
                    skeletons_sum += skeleton_outputs['skeleton'].detach()
                    vstruc_outputs = vstruc_model(inputs.unsqueeze(0))
                    vstrucs_sum += vstruc_outputs['vstruc'].detach()
                index += 1
            if data_size <= batch_size:
                inputs, targets = data, label
                skeleton_outputs = skeleton_model(inputs.unsqueeze(0))
                if config['learning_mode'] == 'graph':
                    graphs_sum += skeleton_outputs['graph'].detach()
                else:
                    skeletons_sum += skeleton_outputs['skeleton'].detach()
                    vstruc_outputs = vstruc_model(inputs.unsqueeze(0))
                    vstrucs_sum += vstruc_outputs['vstruc'].detach()
                index += 1

        if config['learning_mode'] == 'graph':
            graphs = (graphs_sum / index)
            graph_scores = get_scores(targets, graphs, mode='graph', threshold=threshold,
                                      num_of_nodes=targets.shape[-1])
            graph_f1, graph_auc, graph_auprc = graph_scores['f1'], graph_scores['auc'], graph_scores['auprc']
            graph_hamming = graph_scores['hamming_distance']
            gf1s.append(graph_f1)
            gaucs.append(graph_auc)
            gauprcs.append(graph_auprc)
            ghammings.append(graph_hamming)

            skeletons = torch.max(graphs, graphs.transpose(-1, -2))
            skeleton_scores = get_scores(targets, skeletons, mode='skeleton', threshold=threshold,
                                         num_of_nodes=targets.shape[-1])
            skeleton_f1, skeleton_auc, skeleton_auprc = skeleton_scores['f1'], skeleton_scores['auc'], skeleton_scores[
                'auprc']
            skeleton_hamming = skeleton_scores['hamming_distance']
            sf1s.append(skeleton_f1)
            saucs.append(skeleton_auc)
            sauprcs.append(skeleton_auprc)
            shammings.append(skeleton_hamming)

            graphs = graphs > threshold
            # graphs = (targets > threshold).unsqueeze(0) ##
            graph_cube_tforks, graph_cube_vstrucs = batched_dag_to_vstrucs(graphs.cpu().numpy())
            graph_cube_vstrucs_set = set(map(tuple, np.argwhere(graph_cube_vstrucs[0])))
            # else:
            #     print(len(graph_cube_vstrucs_set))
            final_predicted_cpdag = MixedGraph(numberOfNodes=num_of_nodes)
            for i in range(num_of_nodes):
                for j in range(num_of_nodes):
                    if graphs[0, i, j] or graphs[0, j, i]:
                        final_predicted_cpdag.add_undi_edge(i, j)
            for v in graph_cube_vstrucs_set:
                final_predicted_cpdag.del_undi_edge(v[0], v[1])
                final_predicted_cpdag.del_undi_edge(v[0], v[2])
                final_predicted_cpdag.add_di_edge(v[1], v[0])
                final_predicted_cpdag.add_di_edge(v[2], v[0])
            final_predicted_cpdag.apply_meek_rules()
            final_cpdag_ml4c_scores_adjmat = final_predicted_cpdag.getAdjacencyMatrix()
            final_predicted_cpdag.IdentifiableEdges = final_predicted_cpdag.DirectedEdges
            final_predicted_cpdag.vstrucs = graph_cube_vstrucs_set
            final_cpdag_ml4c_scores = cal_score(label_digraph, final_predicted_cpdag)
            final_ml4c_scoress.append(final_cpdag_ml4c_scores)
            continue
        else:
            skeletons = skeletons_sum / index
            skeleton_scores = get_scores(targets, skeletons, mode='skeleton', threshold=threshold)
            skeleton_f1, skeleton_auc, skeleton_auprc = skeleton_scores['f1'], skeleton_scores['auc'], skeleton_scores[
                'auprc']
            skeleton_hamming = skeleton_scores['hamming_distance']
            sf1s.append(skeleton_f1)
            saucs.append(skeleton_auc)
            sauprcs.append(skeleton_auprc)
            shammings.append(skeleton_hamming)

            vstrucs = vstrucs_sum / index

            label_cube_tforks, label_cube_vstrucs = batched_dag_to_vstrucs(targets.cpu().unsqueeze(0).numpy())
            vstruc_scores = vstruc_get_scores(label_cube_vstrucs, vstrucs, label_cube_tforks,
                                              threshold=vstruc_threshold)
            vstruc_f1, vstruc_auc, vstruc_auprc = vstruc_scores['f1'], vstruc_scores['auc'], vstruc_scores['auprc']
            vstruc_hamming = vstruc_scores['hamming_distance']
            label_vstrucs_set = set(map(tuple, np.argwhere(label_cube_vstrucs[0])))
            vstruc_set = set(map(tuple, np.argwhere((vstrucs.cpu().numpy() * label_cube_tforks)[0] > vstruc_threshold)))

            if vstruc_auc and not np.isnan(vstruc_auc):
                vf1s.append(vstruc_f1)
                vaucs.append(vstruc_auc)
                vauprcs.append(vstruc_auprc)
                vhammings.append(vstruc_hamming)
            else:
                continue

            vstrucs = vstrucs.cpu().numpy()

            predicted_skeleton = (skeletons > threshold).cpu().numpy()
            predicted_skeleton = (predicted_skeleton + predicted_skeleton.transpose(0, 2, 1)) > 0
            assert np.allclose(predicted_skeleton, predicted_skeleton.transpose(0, 2, 1))
            cube_tforks = batched_skeleton_to_tforks(predicted_skeleton)[0]
            cube_tforks_set = set(map(tuple, np.argwhere(cube_tforks)))
            label_cube_tforks_set = set(map(tuple, np.argwhere(label_cube_tforks[0])))
            vstrucs_on_skeletons = vstrucs * cube_tforks
            # vstrucs_on_skeletons += vstrucs_on_skeletons.transpose(0, 2, 1)
            jiscores = np.sum(vstrucs_on_skeletons, axis=1) + np.sum(vstrucs_on_skeletons, axis=2)
            ijscores = jiscores.transpose()

            direction_scores = ijscores - jiscores
            predicted_edges = direction_scores > vstruc_threshold

            predicted_cpdag = MixedGraph(numberOfNodes=num_of_nodes)
            for i in range(num_of_nodes):
                for j in range(num_of_nodes):
                    if predicted_skeleton[0, i, j]:
                        if predicted_edges[i, j]:
                            predicted_cpdag.del_undi_edge(i, j)
                            predicted_cpdag.add_di_edge(i, j)
                        elif not predicted_cpdag.has_di_edge(j, i):
                            assert not predicted_cpdag.has_di_edge(i, j)
                            predicted_cpdag.add_undi_edge(i, j)
            predicted_cpdag.apply_meek_rules()
            predicted_cpdag_adjmat = predicted_cpdag.getAdjacencyMatrix()
            tri_cpdag = adjacency_to_triangle_matrix(predicted_cpdag_adjmat)

            tri_cpdag_scores = tri_cpdag_get_scores(label=tri_cpdag_label, output=tri_cpdag)
            cpdag_acc = tri_cpdag_scores['accuracy']
            cpdag_hamming = tri_cpdag_scores['hamming_distance']
            caccs.append(cpdag_acc)
            chammings.append(cpdag_hamming)

            predicted_cpdag = MixedGraph(adjmat=predicted_cpdag_adjmat)
            cpdag_ml4c_scores = cal_score(label_digraph, predicted_cpdag)
            ml4c_scoress.append(cpdag_ml4c_scores)

            vstrucs_on_skeletons_set = set(map(tuple, np.argwhere(vstrucs_on_skeletons > vstruc_threshold)))
            rv = set()
            for v in vstrucs_on_skeletons_set:
                sv = vstrucs_on_skeletons[v]
                flag = False
                for vj in vstrucs_on_skeletons_set:
                    if v[0] in (vj[1], vj[2]) and vj[0] in (v[1], v[2]) and sv < vstrucs_on_skeletons[vj]:
                        flag = True
                        break
                if flag:
                    rv.add(v)
            fv = vstrucs_on_skeletons_set - rv

            final_predicted_cpdag = MixedGraph(numberOfNodes=num_of_nodes)
            for i in range(num_of_nodes):
                for j in range(num_of_nodes):
                    if predicted_skeleton[0, i, j]:
                        final_predicted_cpdag.add_undi_edge(i, j)
            for v in fv:
                final_predicted_cpdag.del_undi_edge(v[0], v[1])
                final_predicted_cpdag.del_undi_edge(v[0], v[2])
                final_predicted_cpdag.add_di_edge(v[1], v[0])
                final_predicted_cpdag.add_di_edge(v[2], v[0])
            x = len(final_predicted_cpdag.detectCycles())
            if x > 0:
                pass
            final_predicted_cpdag.apply_meek_rules()
            final_cpdag_ml4c_scores_adjmat = final_predicted_cpdag.getAdjacencyMatrix()
            final_predicted_cpdag.IdentifiableEdges = final_predicted_cpdag.DirectedEdges
            final_predicted_cpdag.vstrucs = fv
            final_cpdag_ml4c_scores = cal_score(label_digraph, final_predicted_cpdag)
            final_ml4c_scoress.append(final_cpdag_ml4c_scores)
            y = len(final_predicted_cpdag.detectCycles())
            # if y > 0:
            #     print(x, y)

            assert len(label_digraph.getCPDAG().detectCycles()) == 0

    print(f"total: {total_count}, noedge: {noedge_count}, novstruc: {novstruc_count}")
    print(f"{np.mean([i['vstrucs_edges_F1'] for i in final_ml4c_scoress]):.4f}", end="\t")
    print(f"{np.mean([i['identfb_edges_F1'] for i in final_ml4c_scoress]):.4f}", end="\t")
    print(f"{np.mean([i['SHD'] for i in final_ml4c_scoress]):.4f}")


if __name__ == '__main__':
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    for key, value in config.items():
        print(key, ':', value)

    if config['continuous']:
        Dataset = ContinousDataset
    else:
        Dataset = DiscreteDataset

    test_datasets = config['testset_path']
    for key, value in test_datasets.items():
        if value == 'benchmark':
            test_data = Dataset(f"ML4S_Experiments/benchmarks/npy10000/{key}.npy")
            test_label = Dataset(f"ML4S_Experiments/benchmarks/{key}_graph.txt")
            test_datasets[key] = ([(test_data, test_label)])
            standard_test_data = np.load(f"ML4S_Experiments/benchmarks/npy10000/{key}.npy")
        else:
            with open(value + '/config.yml', 'r') as file:
                dataset_config = yaml.load(file, Loader=yaml.FullLoader)
            datasets = []
            for i in range(dataset_config['case0_num']):
                data = Dataset(f"{value}/hc_{i}.npy")
                label = Dataset(f"{value}/hc_{i}.txt")
                datasets.append((data, label))
            for i in range(dataset_config['case1_num']):
                data = Dataset(f"{value}/cpd_{i}.npy")
                label = Dataset(f"{value}/cpd_{i}.txt")
                datasets.append((data, label))
            for i in range(dataset_config['case2_num']):
                data = Dataset(f"{value}/struc_{i}.npy")
                label = Dataset(f"{value}/struc_{i}.txt")
                datasets.append((data, label))
            for i in range(dataset_config['case3_num']):
                data = Dataset(f"{value}/synthetic_{i}.npy")
                label = Dataset(f"{value}/synthetic_{i}.txt")
                datasets.append((data, label))
            test_datasets[key] = datasets

    num_of_nodes = test_datasets[key][0][0].VarCount
    if not config['continuous']:
        num_of_classes = test_datasets[key][0][0].IndexedDataT.max() + 1
    else:
        num_of_classes = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config['continuous']:
        skeleton_graph_model = Model(graph_prediction=(config['learning_mode'] == 'graph'),
                                     pairwise=config['pairwise'],
                                     layers=config['layers']).to(device)
    else:
        skeleton_graph_model = Model(graph_prediction=(config['learning_mode'] == 'graph'),
                                     pairwise=config['pairwise'],
                                     continuous_data=False,
                                     num_of_nodes=num_of_nodes,
                                     num_of_classes=num_of_classes,
                                     input_embedding_dim=None).to(device)

    if config['learning_mode'] == 'graph':
        ori_model = None
    elif config['learning_mode'] == 'skeleton':
        if config['pairwise']:
            if config['continuous']:
                ori_model = WholeModel().to(device)
            else:
                ori_model = WholeModel(continuous_data=False,
                                       num_of_nodes=num_of_nodes,
                                       num_of_classes=num_of_classes,
                                       input_embedding_dim=None).to(device)
            ori_model.load(config['vstruc_predictor_path'])
        else:
            if config['continuous']:
                ori_model = WholeNodewiseModel().to(device)
            else:
                raise NotImplementedError
            ori_model.load(config['vstruc_predictor_path'])
    else:
        raise NotImplementedError

    start_epoch = 0

    skeleton_graph_model.load(config['skeleton_graph_predictor_path'])
    print(f"Loaded skeleton prediction model from {config['skeleton_graph_predictor_path']}")
    print(f"Loaded orientation model from {config['vstruc_predictor_path']}, testing")
    for name, test_dataset in test_datasets.items():
        test(skeleton_graph_model, ori_model, test_dataset, device, name,
             skeleton_threshold=skeleton_graph_model.best_threshold.item(),
             vstruc_threshold=ori_model.best_threshold.item(), batch_size=1000)
