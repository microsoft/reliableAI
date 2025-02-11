from functools import lru_cache

import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score


def skeleton_learning_loss(output, label):
    """
    Accept adjancency matrix for both DAG and skeleton
    :param output: torch tensor. batchsize x d x d. 0 to 1
    :param label:  torch tensor. batchsize x d x d. 0 or 1
    :return: binary cross entropy loss
    >>> output = torch.tensor([[[0.0, 0.5], [0.5, 0.0]]])
    >>> label = torch.tensor([[[0, 1], [0, 0]]])
    >>> skeleton_learning_loss(output, label)
    tensor(0.3466)   # np.log(0.5) * 2 / 4
    """
    target = torch.max(label, label.transpose(-1, -2))
    undirected_output = torch.max(output, output.transpose(-1, -2))
    return torch.nn.functional.binary_cross_entropy(undirected_output, target.type(torch.float))


def to_skeleton(output, threshold=0.5):
    undirected_output = torch.max(output, output.transpose(-1, -2))
    return (undirected_output > threshold).type(torch.int)


def to_graph(output, threshold=0.5):
    return (output > threshold).type(torch.int)


def graph_learning_loss(output, label):
    return torch.nn.functional.binary_cross_entropy(output, label.type(torch.float))


def vstruc_learning_loss(output, label, mask=None):
    """
    :param output: [..., num_of_nodes, num_of_nodes, num_of_nodes], from 0 to 1
    :param label: [..., num_of_nodes, num_of_nodes, num_of_nodes], 0 or 1
    :param mask: [..., num_of_nodes, num_of_nodes, num_of_nodes], 0 or 1
    :return:
    """
    return torch.nn.functional.binary_cross_entropy(output, label.type(torch.float), weight=mask)


def get_scores(label, output, mode='skeleton', sr=None, dr=None, num_of_nodes=None, threshold=0.5,
               check_all_threshold=False):
    if num_of_nodes is None:
        num_of_nodes = label.shape[-1]
    if mode == 'skeleton' or mode == 'graph':
        label = label.cpu()
        output = output.cpu()
        if mode == 'skeleton':
            target = torch.max(label, label.transpose(-1, -2)).numpy().reshape(-1)
            prediction = to_skeleton(output, threshold=threshold).numpy().reshape(-1)
        elif mode == 'graph':
            target = label.numpy().reshape(-1)
            prediction = to_graph(output, threshold=threshold).numpy().reshape(-1)
        else:
            raise NotImplementedError
        weight = (np.ones([num_of_nodes, num_of_nodes]) - np.identity(num_of_nodes)).reshape(
            -1)  # remove diagonal elements
        f1 = f1_score(target, prediction, sample_weight=weight)
        hamming_distance = np.sum((target == prediction) * weight) / (len(target) - num_of_nodes)
        if not np.all(np.logical_or(weight == target, weight == 0)):
            auc = roc_auc_score(target, torch.abs(output).detach().numpy().reshape(-1), sample_weight=weight)
        else:
            auc = None
        auprc = average_precision_score(target, torch.abs(output).detach().numpy().reshape(-1), sample_weight=weight)
        result = {'f1': f1, 'auc': auc, 'auprc': auprc, 'hamming_distance': hamming_distance}
        if check_all_threshold:
            for thres in np.arange(0.0, 1.0, 0.1):
                prediction = to_skeleton(output, threshold=thres).numpy().reshape(-1)
                f1 = f1_score(target, prediction, sample_weight=weight)
                hamming_distance = np.sum(target == prediction) * weight / (len(target) - num_of_nodes)
                result['f1' + str(thres)] = f1
                result['hamming_distance' + str(thres)] = hamming_distance
        return result
    else:
        raise NotImplementedError


def vstruc_get_scores(label, output, weight, threshold=0.5, check_all_threshold=False):
    label = label.astype(np.int64).reshape(-1)
    output = output.cpu().numpy().reshape(-1)
    predicted_output = (output > threshold).astype(np.int64)
    weight = weight.astype(np.int64).reshape(-1)

    hamming_distance = np.sum((label == predicted_output) * weight) / np.sum(weight)
    if (label * weight).max() != 0:
        f1 = f1_score(label, predicted_output, sample_weight=weight)
        auprc = average_precision_score(label, output, sample_weight=weight)
        if (((1 - weight) + weight * label) == 1).all():
            auc = None
        else:
            auc = roc_auc_score(label, output, sample_weight=weight)
    else:
        f1 = None
        auc = None
        auprc = None
    result = {'f1': f1, 'auc': auc, 'auprc': auprc, 'hamming_distance': hamming_distance}
    if (label * weight).max() != 0 and check_all_threshold:
        for thres in np.arange(0.0, 1.0, 0.1):
            predicted_output = (output > threshold).astype(np.int64)
            f1 = f1_score(label, predicted_output, sample_weight=weight)
            result['f1' + str(thres)] = f1
    return result


def tri_cpdag_get_scores(label, output):
    weight = np.triu(np.ones_like(label), 1)
    hamming_distance = np.sum((label == output) * weight) / np.sum(weight)
    accuracy = accuracy_score(label.reshape(-1), output.reshape(-1), sample_weight=weight.reshape(-1))
    return {'hamming_distance': hamming_distance, 'accuracy': accuracy}


@lru_cache
def get_masks(num_of_nodes, environ_nodes, return_torch=False):
    '''
        [i][j][k] means j - i - k !!
        also note that environ node index is always larger than system node index
        return: 
            sss_mask: num_of_nodes x num_of_nodes x num_of_nodes
            sss_mask[i][j][k] == 1 iff i j k are all system nodes else 0
            sse_mask: num_of_nodes x num_of_nodes x num_of_nodes
            sse_mask[i][j][k] == 1 iff i j are all system nodes and e is environ node else 0
    '''
    system_nodes = num_of_nodes - environ_nodes
    sss_mask = np.zeros((num_of_nodes, num_of_nodes, num_of_nodes))
    sss_mask[:system_nodes, :system_nodes, :system_nodes] = 1
    sse_mask = np.zeros((num_of_nodes, num_of_nodes, num_of_nodes))
    sse_mask[:system_nodes, :system_nodes, system_nodes:] = 1
    if return_torch:
        sss_mask, sse_mask = torch.tensor(sss_mask).cuda(), torch.tensor(sse_mask).cuda()

    return sss_mask, sse_mask