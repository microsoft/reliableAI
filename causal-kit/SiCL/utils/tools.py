import random
import os

import torch
import numpy as np

from utils.graph import DiGraph, MixedGraph


def shuffle(inputs: torch.Tensor):
    """
    inputs: batch_size x dataset_size x num_of_nodes
    """
    inputs = inputs.clone()
    batch_size, dataset_size, num_of_nodes = inputs.shape
    for b in range(batch_size):
        num_of_classes = torch.max(inputs[b], dim=0)[0].type(torch.int) + 1
        for i in range(num_of_nodes):
            mapping = list(range(num_of_classes[i].item()))
            random.shuffle(mapping)
            pos = [torch.where(inputs[b, :, i] == j) for j in range(num_of_classes[i])]
            for j in range(num_of_classes[i]):
                assert i < num_of_nodes
                inputs[b, pos[j][0], i] = mapping[j]
    return inputs

def dag_to_vstrucs(di_adjmat):
    """
    batch, num_of_nodes, num_of_nodes
    """
    undi_adjmat = di_adjmat + di_adjmat.T # the skeleton, will not have value 2
    i_j_adjacent = undi_adjmat[:, :, None].astype(bool)
    k_j_adjacent = undi_adjmat[:, None, :].astype(bool)
    i_k_not_adjacent_and_i_less_than_k = np.triu(1 - undi_adjmat, k=1)[None, :, :].astype(bool)
    cube_tforks = i_j_adjacent * k_j_adjacent * i_k_not_adjacent_and_i_less_than_k

    i_point_to_j = di_adjmat.T[:, :, None].astype(bool)
    k_point_to_j = di_adjmat.T[:, None, :].astype(bool)
    i_k_not_adjacent_and_i_less_than_k = np.triu(1 - undi_adjmat, k=1)[None, :, :].astype(bool)
    cube_vstrucs = i_point_to_j * k_point_to_j * i_k_not_adjacent_and_i_less_than_k
    return cube_tforks, cube_vstrucs

def batched_dag_to_vstrucs(batched_di_adjmat):
    """
    batch, num_of_nodes, num_of_nodes
    j - i - k
    """
    batched_di_adjmat_transpose = batched_di_adjmat.transpose((0, -1, -2))
    batched_undi_adjmat = batched_di_adjmat + batched_di_adjmat_transpose # the skeleton, will not have value 2
    batched_i_j_adjacent = batched_undi_adjmat[:, :, :, None].astype(bool)
    batched_k_j_adjacent = batched_undi_adjmat[:, :, None, :].astype(bool)
    batched_i_k_not_adjacent_and_i_less_than_k = np.triu(1 - batched_undi_adjmat, k=1)[:, None, :, :].astype(bool)
    batched_cube_tforks = batched_i_j_adjacent * batched_k_j_adjacent * batched_i_k_not_adjacent_and_i_less_than_k

    batched_i_point_to_j = batched_di_adjmat_transpose[:, :, :, None].astype(bool)
    batched_k_point_to_j = batched_di_adjmat_transpose[:, :, None, :].astype(bool)
    batched_i_k_not_adjacent_and_i_less_than_k = np.triu(1 - batched_undi_adjmat, k=1)[:, None, :, :].astype(bool)
    batched_cube_vstrucs = batched_i_point_to_j * batched_k_point_to_j * batched_i_k_not_adjacent_and_i_less_than_k
    return batched_cube_tforks, batched_cube_vstrucs

def batched_skeleton_to_tforks(batched_skeleton):
    """j - i - k"""
    batched_i_j_adjacent = batched_skeleton[:, :, :, None].astype(bool)
    batched_k_j_adjacent = batched_skeleton[:, :, None, :].astype(bool)
    batched_i_k_not_adjacent_and_i_less_than_k = np.triu(1 - batched_skeleton, k=1)[:, None, :, :].astype(bool)
    batched_cube_tforks = batched_i_j_adjacent * batched_k_j_adjacent * batched_i_k_not_adjacent_and_i_less_than_k
    return batched_cube_tforks

def skeleton_vstruc_to_cpdag():
    pass

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def dag_to_dag(graph, verbose=False):
    """
    :param graph: input dag (adjmat)
    :return: another dag within a same MEC
    """
    g = DiGraph(graph)
    cpdag = g.getCPDAG()
    _, cube_ori_vstrucs = dag_to_vstrucs(graph)
    # ori_vstrucs = set(map(tuple, np.argwhere(cube_ori_vstrucs)))

    while True:
        newg = DiGraph(graph)
        for (fromnode, tonode) in cpdag.UndirectedEdges:
            newg.del_di_edge(fromnode=fromnode, tonode=tonode)
            newg.del_di_edge(fromnode=tonode, tonode=fromnode)
            if random.randint(0, 1) == 0:
                newg.add_di_edge(fromnode=fromnode, tonode=tonode)
            else:
                newg.add_di_edge(fromnode=tonode, tonode=fromnode)

        if newg.detectCycles():
            continue
        _, cube_new_vstrucs = dag_to_vstrucs(newg.getAdjacencyMatrix())
        if not np.allclose(cube_new_vstrucs, cube_ori_vstrucs):
            continue
        break

    assert np.allclose(newg.getCPDAG().getAdjacencyMatrix(), g.getCPDAG().getAdjacencyMatrix())
    newadj = newg.getAdjacencyMatrix()
    if verbose:
        print(len(cpdag.UndirectedEdges), np.sum(graph), np.sum(np.abs(graph - newadj)))
    return newadj




if __name__ == "__main__":
    
    test_input = np.array([
        [0,1,0,0,0],
        [0,0,0,0,0],
        [1,0,0,0,1],
        [1,1,0,0,0],
        [1,1,0,0,0],
    ])
    
    print(dag_to_dag(test_input))