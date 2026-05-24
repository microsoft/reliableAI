import numpy as np
from base.pdag import MixedGraph
import rpy2.robjects as robjects


def get_compared_components(graph_path, intervention_size, real):
    aug_graph = MixedGraph(graph_path = graph_path)
    sys_size = len(aug_graph.Nodes)-intervention_size
    aug_Skeleton = aug_graph.getSkeleton().getAdjacencyMatrix()
    
    if real:
        aug_CPDAG = aug_graph.getAdjacencyMatrix()
    else:
        aug_CPDAG = aug_graph.getCPDAG().getAdjacencyMatrix()
        
    I_SKELETON = aug_Skeleton[:sys_size,:sys_size]
    I_TARGETS = set(map(lambda edge:tuple(sorted(edge, reverse=True)), [edge for edge in np.argwhere(aug_Skeleton == 1) if edge[0] >= sys_size or edge[1] >= sys_size]))
    I_MEG = aug_CPDAG[:sys_size,:sys_size]
    return I_SKELETON, I_TARGETS, I_MEG


def get_skeleton(adj_matrix):
    return np.maximum(adj_matrix, adj_matrix.T)


def _separate_edges(adj_matrix, undirected=False):
    directed_indices = np.transpose(np.where((adj_matrix == 1) & (adj_matrix.T != 1)))
    undirected_indices = [edge for edge in np.transpose(np.where((adj_matrix == 1) & (adj_matrix.T == 1))) if edge[0] > edge[1]]
    return set(map(tuple, undirected_indices)) if undirected else set(map(tuple, directed_indices))


def metric_skeleton_level(pred_I_SKELETON, targ_I_SKELETON):
    """ P / R / F1 (I-SKELETON Level) """
    pred_iden_edges, targ_iden_edges = _separate_edges(pred_I_SKELETON, True), _separate_edges(targ_I_SKELETON, True)
    precision = len(targ_iden_edges.intersection(pred_iden_edges)) / max(len(pred_iden_edges), 1)
    recall = len(targ_iden_edges.intersection(pred_iden_edges)) / max(len(targ_iden_edges), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    pred_iden_edges_num, targ_iden_edges_num = len(pred_iden_edges), len(targ_iden_edges)
    return {'p':round(precision,2), 'r':round(recall,2), 'f1':round(f1,2), '#pred_iden_edges':int(pred_iden_edges_num), '#targ_iden_edges':int(targ_iden_edges_num)}


def metric_target_level(pred_I_TARGET, targ_I_TARGET):
    """ P / R / F1 (I-TARGET Level) """
    precision = len(targ_I_TARGET.intersection(pred_I_TARGET)) / max(len(pred_I_TARGET), 1)
    recall = len(targ_I_TARGET.intersection(pred_I_TARGET)) / max(len(targ_I_TARGET), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    pred_iden_edges_num, targ_iden_edges_num = len(pred_I_TARGET), len(targ_I_TARGET)
    return {'p':round(precision,2), 'r':round(recall,2), 'f1':round(f1,2), '#pred_iden_edges':int(pred_iden_edges_num), '#targ_iden_edges':int(targ_iden_edges_num)}


def metric_cpdag_level(pred_I_CPDAG, targ_I_CPDAG):
    """ SHD / SID (I-CPDAG Level) and P / R / F1 (#iden edges on I-CPDAG Level)"""
    
    def _structural_distance(predict, target):
        """Compute the Structural Hamming Distance and Structural Intervention Distance use R package.

        Args:
            predict (numpy.ndarray): Prediction made by the
                algorithm to evaluate.
            target (numpy.ndarray): Target graph, must be a DAG.
            
        Returns:
            int: Structural Intervention Distance (original / lower / higher).
                Structural Hamming Distance (int). 
                The value tends to zero as the graphs tend to be identical.
        """
        predict_str = "Predict <- rbind(" + ", ".join(["c(" + ", ".join(map(str, row)) + ")" for row in predict]) + ")"
        target_str = "Target <- rbind(" + ", ".join(["c(" + ", ".join(map(str, row)) + ")" for row in target]) + ")"
        
        r_script = f'''library(SID)
        {predict_str}
        {target_str}
        sid_attribute <- structIntervDist(Target,Predict)
        shd <- hammingDist(Target,Predict)

        sid <- sid_attribute$sid
        sid_lower <- sid_attribute$sidLowerBound
        sid_upper <- sid_attribute$sidUpperBound

        list(shd, sid, sid_lower, sid_upper)
        '''

        # Execute the R script and get the result
        result = robjects.r(r_script)

        # Now you can access the variables G, H1, and H2
        shd, sid, sid_lower, sid_upper = result[0][0], result[1][0], result[2][0], result[3][0]
        return shd, (sid, sid_lower, sid_upper)
    
    def _cal_prf(pred, targ):
        pred_iden_edges, targ_iden_edges = _separate_edges(pred, False), _separate_edges(targ, False)
        precision = len(targ_iden_edges.intersection(pred_iden_edges)) / max(len(pred_iden_edges), 1)
        recall = len(targ_iden_edges.intersection(pred_iden_edges)) / max(len(targ_iden_edges), 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        pred_iden_edges_num, targ_iden_edges_num = len(pred_iden_edges), len(targ_iden_edges)
        return precision, recall, f1, pred_iden_edges_num, targ_iden_edges_num
    
    shd, sid = _structural_distance(pred_I_CPDAG, targ_I_CPDAG)
    p, r, f1, p_num, t_num = _cal_prf(pred_I_CPDAG, targ_I_CPDAG)
    return {'shd':int(shd), 'sid':sid, 'p':round(p,2), 'r':round(r,2), 'f1':round(f1,2), '#pred_iden_edges':int(p_num), '#targ_iden_edges':int(t_num)}
