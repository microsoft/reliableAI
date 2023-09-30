# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''
This file is modified from the FCI impelmentation in causal_learn package.
'''

from itertools import combinations, permutations
from typing import List

import networkx as nx
import tqdm
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.search.ConstraintBased.FCI import ruleR3, rulesR1R2cycle


def dag2pag(dag: GeneralGraph, islatent: List[GraphNode], cutoff: int=4):
    '''
    Covert a DAG to its corresponding PAG
    Parameters
    ----------
    dag : Direct Acyclic Graph
    islatent: the indexes of latent variables. [] means there is no latent variable
    Returns
    -------
    PAG : Partial Ancestral Graph
    '''
    udg = nx.Graph()
    nodes = dag.get_nodes()
    nodes_ids = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    for x, y in combinations(range(n), 2):
        if dag.get_edge(nodes[x], nodes[y]):
            udg.add_edge(x, y)

    observed_nodes = list(set(nodes) - set(islatent))

    PAG = GeneralGraph(observed_nodes)
    for nodex, nodey in combinations(observed_nodes, 2):
        edge = Edge(nodex, nodey, Endpoint.CIRCLE, Endpoint.CIRCLE)
        edge.set_endpoint1(Endpoint.CIRCLE)
        edge.set_endpoint2(Endpoint.CIRCLE)
        PAG.add_edge(edge)

    sepset = {(nodex, nodey): set() for nodex, nodey in permutations(observed_nodes, 2)}

    for nodex, nodey in combinations(observed_nodes, 2):
        if nodex in islatent:
            continue
        if nodey in islatent:
            continue
        all_paths = nx.all_simple_paths(udg, nodes_ids[nodex], nodes_ids[nodey], cutoff)
        noncolider_path = []
        is_connected = False
        for path in all_paths:
            path_sep = True
            has_nonlatent = False
            for i in range(1, len(path) - 1):
                if nodes[path[i]] in observed_nodes:
                    has_nonlatent = True
                has_collider = is_endpoint(dag.get_edge(nodes[path[i - 1]], nodes[path[i]]), nodes[path[i]],
                                           Endpoint.ARROW) and \
                               is_endpoint(dag.get_edge(nodes[path[i + 1]], nodes[path[i]]), nodes[path[i]],
                                           Endpoint.ARROW)
                if has_collider:
                    path_sep = False
            if not path_sep:
                continue
            if has_nonlatent:
                noncolider_path.append(path)
            else:
                is_connected = True
                break
        if not is_connected:
            edge = PAG.get_edge(nodex, nodey)
            if edge:
                PAG.remove_edge(edge)
            for path in noncolider_path:
                for i in range(1, len(path) - 1):
                    if nodes[path[i]] in islatent:
                        continue
                    sepset[(nodex, nodey)] |= {nodes[path[i]]}
                    sepset[(nodey, nodex)] |= {nodes[path[i]]}

    for nodex, nodey in combinations(observed_nodes, 2):
        if PAG.get_edge(nodex, nodey):
            continue
        for nodez in observed_nodes:
            if nodez == nodex:
                continue
            if nodez == nodey:
                continue
            if nodez not in sepset[(nodex, nodey)]:
                edge_xz = PAG.get_edge(nodex, nodez)
                edge_yz = PAG.get_edge(nodey, nodez)
                if edge_xz and edge_yz:
                    PAG.remove_edge(edge_xz)
                    mod_endpoint(edge_xz, nodez, Endpoint.ARROW)
                    PAG.add_edge(edge_xz)

                    PAG.remove_edge(edge_yz)
                    mod_endpoint(edge_yz, nodez, Endpoint.ARROW)
                    PAG.add_edge(edge_yz)

    changeFlag = True

    while changeFlag:
        changeFlag = False
        changeFlag = rulesR1R2cycle(PAG, None, changeFlag, False)
        changeFlag = ruleR3(PAG, sepset, None, changeFlag, False)

    return PAG


def is_fully_directed(edge):
    if edge:
        if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
            return True
    return False


def is_endpoint(edge, z, end):
    if edge.get_node1() == z:
        if edge.get_endpoint1() == end:
            return True
        else:
            return False
    elif edge.get_node2() == z:
        if edge.get_endpoint2() == end:
            return True
        else:
            return False
    else:
        raise ValueError("z not in edge")


def mod_endpoint(edge, z, end):
    if edge.get_node1() == z:
        edge.set_endpoint1(end)
    elif edge.get_node2() == z:
        edge.set_endpoint2(end)
    else:
        raise ValueError("z not in edge")

if __name__ == "__main__":
    from causallearn.graph.Edges import Edges
    nodes = []
    nodes.append(GraphNode(f"Location "))
    nodes.append(GraphNode(f"Stress"))
    nodes.append(GraphNode(f"Smoke"))
    nodes.append(GraphNode(f"Lung Cancer"))
    nodes.append(GraphNode(f"Surgery"))
    nodes.append(GraphNode(f"5Y Survival"))
    dag = GeneralGraph(nodes)
    dag.add_edge(Edges().directed_edge(nodes[0], nodes[2]))
    dag.add_edge(Edges().directed_edge(nodes[1], nodes[2]))
    dag.add_edge(Edges().directed_edge(nodes[2], nodes[3]))
    dag.add_edge(Edges().directed_edge(nodes[3], nodes[4]))
    dag.add_edge(Edges().directed_edge(nodes[3], nodes[5]))
    dag.add_edge(Edges().directed_edge(nodes[4], nodes[5]))
    pag = dag2pag(dag, [])
    for e in pag.get_graph_edges():
        print(e)