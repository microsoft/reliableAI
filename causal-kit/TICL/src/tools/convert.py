from pgmpy.base import DAG
import numpy as np


def dag_to_adj(dag):
    adj = np.zeros((len(dag.nodes()), len(dag.nodes())), dtype=int)
    for x, y in dag.edges():
        adj[int(x), int(y)] = 1
    return adj


def adj_to_dag(adj):
    dag = DAG()
    dag.add_nodes_from(list(map(str, range(len(adj)))))
    dag.add_edges_from(list(map(lambda edge:(str(edge[0]), str(edge[1])), np.argwhere(adj))))
    return dag


def bn_to_adj(bn):
    adj = np.zeros((len(bn.nodes()), len(bn.nodes())), dtype=int)
    for x, y in bn.edges():
        adj[int(x), int(y)] = 1
    return adj