# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#!/usr/bin/env python
#-*- coding: utf-8 -*-
import copy, os
from itertools import combinations
from sys import path
import numpy as np
from pgmpy.models import BayesianNetwork


class Node(object):
    def __init__(self):
        self.To = set()
        self.From = set()
        self.Neighbor = set() # for undirected edges
    def GetTo(self):
        return copy.deepcopy(self.To)
    def GetFrom(self):
        return copy.deepcopy(self.From)
    def GetNeighbor(self):
        return copy.deepcopy(self.Neighbor)
    def AddTo(self, x):
        self.To.add(x)
    def AddFrom(self, x):
        self.From.add(x)
    def AddNeighbor(self, x):
        self.Neighbor.add(x)
    def DelTo(self, x):
        self.To.remove(x)
    def DelFrom(self, x):
        self.From.remove(x)
    def DelNeighbor(self, x):
        self.Neighbor.remove(x)

class DiGraph(object):
    def __init__(self, graphtxtpath):
        self.graphtxtpath = graphtxtpath
        adjmat = np.loadtxt(graphtxtpath, dtype=np.int16)
        self.DirectedEdges = set()
        self.NodeIDs = list(range(len(adjmat)))  # ordered list
        self.Nodes = {i: Node() for i in self.NodeIDs}
        for i in self.NodeIDs:
            for j in self.NodeIDs:
                if adjmat[i, j]: self.add_di_edge(i, j)

        # tforks_pth = graphtxtpath.replace('_graph', '_TForks')
        # vstrucs_pth = graphtxtpath.replace('_graph', '_VStrucs')
        # if os.path.exists(tforks_pth) and os.path.exists(vstrucs_pth):
        #     # consider when txt is empty, eg. sachs, or only one line.
        #     self.vstrucs = set(map(tuple, np.loadtxt(vstrucs_pth, dtype=int).reshape((-1, 3))))
        #     self.tforks = set(map(tuple, np.loadtxt(tforks_pth, dtype=int).reshape((-1, 3))))
        # else:
        #     # returns in, and cube indexed in format (j, i, k), where i--j--k, and i < k
        #     di_adjmat = self.getAdjacencyMatrix()
        #     undi_adjmat = di_adjmat + di_adjmat.T # the skeleton, will not have value 2
        #     i_j_adjacent = undi_adjmat[:, :, None].astype(bool)
        #     k_j_adjacent = undi_adjmat[:, None, :].astype(bool)
        #     i_k_not_adjacent_and_i_less_than_k = np.triu(1 - undi_adjmat, k=1)[None, :, :].astype(bool)
        #     cube_tforks = i_j_adjacent * k_j_adjacent * i_k_not_adjacent_and_i_less_than_k
        #     self.tforks = set(map(tuple, np.argwhere(cube_tforks)))

        #     i_point_to_j = di_adjmat.T[:, :, None].astype(bool)
        #     k_point_to_j = di_adjmat.T[:, None, :].astype(bool)
        #     i_k_not_adjacent_and_i_less_than_k = np.triu(1 - undi_adjmat, k=1)[None, :, :].astype(bool)
        #     cube_vstrucs = i_point_to_j * k_point_to_j * i_k_not_adjacent_and_i_less_than_k
        #     self.vstrucs = set(map(tuple, np.argwhere(cube_vstrucs)))

        #     ###################### equiv. for loop version of the code above ######################
        #     ###################### to find all tforks and vstrucs from graph ######################
        #     # for j in self.NodeIDs:
        #     #     for i in self.NodeIDs:
        #     #         for k in self.NodeIDs:
        #     #             if k <= i or j == i or j == k: continue
        #     #             if self.is_adjacent(i, j) and self.is_adjacent(k, j) and not self.is_adjacent(i, k):
        #     #                 self.tforks.add((j, i, k))
        #     #                 if self.has_di_edge(i, j) and self.has_di_edge(k, j):
        #     #                     self.vstrucs.add((j, i, k))
        #     ###################### ###################### ###################### ##################
        #     ###################### ###################### ###################### ##################

        #     np.savetxt(vstrucs_pth, np.array(sorted(list(self.vstrucs)), dtype=int), fmt='%i')
        #     np.savetxt(tforks_pth, np.array(sorted(list(self.tforks)), dtype=int), fmt='%i')

        # identifiable_pth = graphtxtpath.replace('_graph', '_Identifiables')
        # if os.path.exists(identifiable_pth):
        #     self.IdentifiableEdges = set(map(tuple, np.loadtxt(identifiable_pth, dtype=int).reshape((-1, 2))))
        # else:
        #     pdag = MixedGraph(nodeIDs=self.NodeIDs)
        #     for (j, i, k) in self.vstrucs:
        #         pdag.add_di_edge(i, j)
        #         pdag.add_di_edge(k, j)
        #     for (fromnode, tonode) in self.DirectedEdges:
        #         if not pdag.has_di_edge(fromnode, tonode):
        #             pdag.add_undi_edge(fromnode, tonode)
        #     pdag.apply_meek_rules()
        #     self.IdentifiableEdges = pdag.DirectedEdges
        #     assert self.IdentifiableEdges.issubset(self.DirectedEdges)
        #     del pdag  # memory to self.IdentifiableEdges (original tmp_mix.DirectedEdges) still holds
        #     np.savetxt(identifiable_pth, np.array(sorted(list(self.IdentifiableEdges)), dtype=int), fmt='%i')
        
        self.topoSort()

    def is_adjacent(self, fromnode, tonode):
        return self.has_di_edge(fromnode, tonode) or self.has_di_edge(tonode, fromnode)

    def has_di_edge(self, fromnode, tonode):
        return (fromnode, tonode) in self.DirectedEdges

    def add_di_edge(self, fromnode, tonode):
        if not self.has_di_edge(fromnode, tonode):
            self.DirectedEdges.add((fromnode, tonode))
            self.Nodes[fromnode].AddTo(tonode)
            self.Nodes[tonode].AddFrom(fromnode)

    def del_di_edge(self, fromnode, tonode):
        if self.has_di_edge(fromnode, tonode):
            self.DirectedEdges.remove((fromnode, tonode))
            self.Nodes[fromnode].DelTo(tonode)
            self.Nodes[tonode].DelFrom(fromnode)

    def getCPDAG(self):
        pdag = MixedGraph(nodeIDs=self.NodeIDs)
        for (fromnode, tonode) in self.DirectedEdges:
            if (fromnode, tonode) not in self.IdentifiableEdges:
                pdag.add_undi_edge(fromnode, tonode)
            else:
                pdag.add_di_edge(fromnode, tonode)
        return pdag

    def getAdjacencyMatrix(self):
        adjmat = np.zeros((len(self.NodeIDs), len(self.NodeIDs)), dtype=int)
        if self.DirectedEdges:
            di_inds = tuple(np.array(list(self.DirectedEdges)).T)
            adjmat[di_inds] = 1
        return adjmat

    def withinThreeHop(self, x, y):
        oneHopPc = self.getPC(x)
        if y in self.getPC(x): return 1
        twoHopPC = set()
        for neighbor in oneHopPc:
            twoHopPC = twoHopPC.union(self.getPC(neighbor))
        if y in twoHopPC: return 1/2
        threeHopPC = set()
        for neighbor in threeHopPC:
            threeHopPC = threeHopPC.union(self.getPC(neighbor))
        if y in threeHopPC: return 1/3
        return 0
    
    def localityWeightZeroFaithful(self, x, y):
        if self.topo_order.index(x) > self.topo_order.index(y):
            x, y = y, x
        def xToY(x, y):
            oneHopPc = self.getTo(x)
            if y in self.getTo(x): return 1
            twoHopPC = set()
            for neighbor in oneHopPc:
                twoHopPC = twoHopPC.union(self.getTo(neighbor))
            if y in twoHopPC: return 1/2
            threeHopPC = set()
            for neighbor in threeHopPC:
                threeHopPC = threeHopPC.union(self.getTo(neighbor))
            if y in threeHopPC: return 1/3
            return 0
        return xToY(x, y)

    def localityWeightOneFaithful(self, x, y):
        def xToY(x, y):
            oneHopPc = self.getTo(x)
            if y in self.getTo(x): return 1
            twoHopPC = set()
            for neighbor in oneHopPc:
                twoHopPC = twoHopPC.union(self.getTo(neighbor))
            if y in twoHopPC: return 1/2
            threeHopPC = set()
            for neighbor in threeHopPC:
                threeHopPC = threeHopPC.union(self.getTo(neighbor))
            if y in threeHopPC: return 1/3
            return 0
        xy = xToY(x, y)
        if xy != 0:
            return xy
        else:
            return xToY(y, x)

    def getWeight(self, x, y, l, k):
        hops = self.reachable(x,y,l)
        # inreachable in l-hop
        if hops == -1: return .0
        # direct adjacency
        if self.is_adjacent(x, y): return 1
        # exist a common child
        if len(self.getTo(x).intersection(self.getTo(y))) > 0:
            return 0
        # construct moral graph
        all_ancestors = self.getAncestor(x, l).union(self.getAncestor(y, l))
        nodes = {x, y}.union(all_ancestors)
        moral = MoralGraph(nodes)
        for edge in combinations(nodes, 2):
            u, v = edge
            # add undirected edge to moral graph iff. they are adjacent or share a common child
            if self.is_adjacent(u, v) or len(self.getTo(x).intersection(self.getTo(y))) > 0: 
                moral.addEdge(u, v)
        initial_separators = all_ancestors
        minimal_sep_size = moral.minimal_d_sep(x, y, initial_separators)
        if minimal_sep_size == -1:
            return .0
        elif minimal_sep_size <= k:
            return 1 / hops
        else:
            return .0

    def reachable(self, x, y, l):
        # l-hop reachability
        visited = set()
        paths = [[x]]
        while len(paths) > 0:
            curr_path = paths.pop(0)
            curr_node = curr_path[-1]
            if curr_node in visited: continue
            else: visited.add(curr_node)
            for neighbor in self.getPC(curr_node):
                if neighbor == y: return len(curr_path)
                if len(curr_path) <= l and neighbor not in visited:
                    new_path = copy.copy(curr_path)
                    new_path.append(neighbor)
                    paths.append(new_path)
        return -1

    def semi_adjacent(self, x, y):
        return len(self.getNeighbour(x).intersection(self.getNeighbour(y))) != 0

    def getNeighbour(self, x):
        return self.getFrom(x).union(self.getTo(x))

    def getAncestor(self, x, l):
        if l==1: return self.getFrom(x)
        else:
            ancestors = set()
            for ancestor in self.getFrom(x):
                ancestors = ancestors.union(self.getAncestor(ancestor, l-1))
            return ancestors

    def getTo(self, x):
        return self.Nodes[x].GetTo()

    def getFrom(self, x):
        return self.Nodes[x].GetFrom()

    def getPC(self, x):
        return self.Nodes[x].GetTo().union(self.Nodes[x].GetFrom())
    
    def topoSort(self):
        self.topo_order = []
        curr = [i for i in self.NodeIDs if len(self.getFrom(i)) == 0]

        while len(curr) != 0:
            next = set()
            for node in curr:
                self.topo_order.append(node)
                for child in self.getTo(node):
                    if child not in self.topo_order:
                        next.add(child)
            curr = list(next)
    
    def GetSkeleton(self):
        skeleton = MixedGraph(numberOfNodes=len(self.NodeIDs))
        for edge in self.DirectedEdges:
            skeleton.add_undi_edge(edge[0], edge[1])
        return skeleton
    
    def GetBN(self):
        bn = BayesianNetwork()
        bn.add_nodes_from(map(str, self.NodeIDs))
        bn.add_edges_from([(str(e[0]), str(e[1])) for e in self.DirectedEdges])
        return bn
        
class MoralGraph(object):
    def __init__(self, nodeIDs):
        self.NodeIDs = nodeIDs
        self.Nodes = {i:Node() for i in self.NodeIDs}
        self.Edges = set()
    
    def addEdge(self, fromnode, tonode):
        less, more = (fromnode, tonode) if (fromnode < tonode) else (tonode, fromnode)
        if not self.has_edge(less, more):
            self.Edges.add((less, more))
            self.Nodes[less].AddNeighbor(more)
            self.Nodes[more].AddNeighbor(less)

    def has_edge(self, fromnode, tonode):
        less, more = (fromnode, tonode) if (fromnode < tonode) else (tonode, fromnode)
        return (less, more) in self.Edges
    
    def minimal_d_sep(self, node1: int, node2: int, initial_separators: set) -> int:
        if self.has_edge(node1, node2): return -1
        marks = set()
        visited = set()
        paths = [[node1]]
        while len(paths) > 0:
            curr_path = paths.pop(0)
            curr_node = curr_path[-1]
            if curr_node in visited: continue
            else: visited.add(curr_node)
            if curr_node in initial_separators:
                marks.add(curr_node)
            else:
                for neighbor in self.Nodes[curr_node].GetNeighbor():
                    if neighbor not in visited:
                        new_path = copy.copy(curr_path)
                        new_path.append(neighbor)
                        paths.append(new_path)
        
        visited = set()
        minimal_separator = set()
        paths = [[node2]]
        while len(paths) > 0:
            curr_path = paths.pop(0)
            curr_node = curr_path[-1]
            if curr_node in visited: continue
            else: visited.add(curr_node)
            if curr_node in marks:
                minimal_separator.add(curr_node)
            else:
                for neighbor in self.Nodes[curr_node].GetNeighbor():
                    if neighbor not in visited:
                        new_path = copy.copy(curr_path)
                        new_path.append(neighbor)
                        paths.append(new_path)
        
        return len(minimal_separator)


    def is_reachable(self, node1, node2, separators) -> bool:
        visited = set()
        queue = [node1]

        while len(queue) != 0:
            curr = queue.pop(0)
            if curr in visited: continue
            else: visited.add(curr)
            for neighbor in self.Nodes[curr].GetNeighbor():
                if neighbor == node2: return True
                if neighbor not in visited and neighbor not in separators:
                    queue.append(neighbor)
        
        return False


class MixedGraph(object):
    def __init__(self, numberOfNodes=0, nodeIDs=None, graphtxtpath=None):
        if not graphtxtpath:
            self.NodeIDs = list(range(numberOfNodes)) if not nodeIDs else nodeIDs # ordered list
            self.Nodes = {i: Node() for i in self.NodeIDs}
            self.DirectedEdges = set()
            self.UndirectedEdges = set()
        else:
            adjmat = np.loadtxt(graphtxtpath, dtype=np.int16)
            self.DirectedEdges = set()
            self.UndirectedEdges = set()
            self.NodeIDs = list(range(len(adjmat)))  # ordered list
            self.Nodes = {i: Node() for i in self.NodeIDs}
            for i in self.NodeIDs:
                for j in self.NodeIDs:
                    if adjmat[i, j]:
                        if not adjmat[j, i]:
                            self.add_di_edge(i, j)
                        else:
                            self.add_undi_edge(i, j)

            tforks_pth = graphtxtpath.replace('_graph', '_TForks')
            vstrucs_pth = graphtxtpath.replace('_graph', '_VStrucs')
            if os.path.exists(tforks_pth) and os.path.exists(vstrucs_pth):
                self.vstrucs = set(map(tuple, np.loadtxt(vstrucs_pth, dtype=int).reshape((-1, 3))))# warning when txt is empty, eg. sachs
                self.tforks = set(map(tuple, np.loadtxt(tforks_pth, dtype=int).reshape((-1, 3))))
            else:
                mixed_adjmat = self.getAdjacencyMatrix()
                undi_adjmat = mixed_adjmat + mixed_adjmat.T
                undi_adjmat[undi_adjmat == 2] = 1
                di_adjmat = np.copy(mixed_adjmat)
                di_adjmat[mixed_adjmat == mixed_adjmat.T] = 0

                i_j_adjacent = undi_adjmat[:, :, None].astype(bool)
                k_j_adjacent = undi_adjmat[:, None, :].astype(bool)
                i_k_not_adjacent_and_i_less_than_k = np.triu(1 - undi_adjmat, k=1)[None, :, :].astype(bool)
                cube_tforks = i_j_adjacent * k_j_adjacent * i_k_not_adjacent_and_i_less_than_k
                self.tforks = set(map(tuple, np.argwhere(cube_tforks)))

                i_point_to_j = di_adjmat.T[:, :, None].astype(bool)
                k_point_to_j = di_adjmat.T[:, None, :].astype(bool)
                i_k_not_adjacent_and_i_less_than_k = np.triu(1 - undi_adjmat, k=1)[None, :, :].astype(bool)
                cube_vstrucs = i_point_to_j * k_point_to_j * i_k_not_adjacent_and_i_less_than_k
                self.vstrucs = set(map(tuple, np.argwhere(cube_vstrucs)))

                ###################### equiv. for loop version of the code above ######################
                ###################### to find all tforks and vstrucs from graph ######################
                # for j in self.NodeIDs:
                #     for i in self.NodeIDs:
                #         for k in self.NodeIDs:
                #             if k <= i or j == i or j == k: continue
                #             if self.adjacent_in_mixed_graph(i, j) and self.adjacent_in_mixed_graph(k, j) and not self.adjacent_in_mixed_graph(i, k):
                #                 self.tforks.add((j, i, k))
                #                 if self.has_di_edge(i, j) and self.has_di_edge(k, j):
                #                     self.vstrucs.add((j, i, k))
                ###################### ###################### ###################### ##################
                ###################### ###################### ###################### ##################

                np.savetxt(vstrucs_pth, np.array(sorted(list(self.vstrucs)), dtype=int), fmt='%i')
                np.savetxt(tforks_pth, np.array(sorted(list(self.tforks)), dtype=int), fmt='%i')

            identifiable_pth = graphtxtpath.replace('_graph', '_Identifiables')
            if os.path.exists(identifiable_pth):
                self.IdentifiableEdges = set(map(tuple, np.loadtxt(identifiable_pth, dtype=int).reshape((-1, 2))))
            else:
                pdag = MixedGraph(nodeIDs=self.NodeIDs)
                for (j, i, k) in self.vstrucs:
                    pdag.add_di_edge(i, j)
                    pdag.add_di_edge(k, j)
                for (fromnode, tonode) in self.DirectedEdges.union(self.UndirectedEdges):
                    if not pdag.has_di_edge(fromnode, tonode):
                        pdag.add_undi_edge(fromnode, tonode)
                pdag.apply_meek_rules()
                self.IdentifiableEdges = pdag.DirectedEdges
                # now self.IdentifiableEdges.issubset(self.DirectedEdges) may be wrong: eg. input is just PDAG
                del pdag  # memory to self.IdentifiableEdges (original tmp_mix.DirectedEdges) still holds
                np.savetxt(identifiable_pth, np.array(sorted(list(self.IdentifiableEdges)), dtype=int), fmt='%i')

    def adjacent_in_mixed_graph(self, x, y):
        return self.has_di_edge(x, y) or self.has_di_edge(y, x) or self.has_undi_edge(x, y)

    def has_di_edge(self, fromnode, tonode):
        return (fromnode, tonode) in self.DirectedEdges

    def add_di_edge(self, fromnode, tonode, force_del=True):
        if not self.has_di_edge(fromnode, tonode):
            if force_del and self.has_di_edge(tonode, fromnode): # to prevent loop, a->b also b->a
                self.del_di_edge(tonode, fromnode)
            self.DirectedEdges.add((fromnode, tonode))
            self.Nodes[fromnode].AddTo(tonode)
            self.Nodes[tonode].AddFrom(fromnode)

    def del_di_edge(self, fromnode, tonode):
        if self.has_di_edge(fromnode, tonode):
            self.DirectedEdges.remove((fromnode, tonode))
            self.Nodes[fromnode].DelTo(tonode)
            self.Nodes[tonode].DelFrom(fromnode)

    def has_undi_edge(self, fromnode, tonode):
        less, more = (fromnode, tonode) if (fromnode < tonode) else (tonode, fromnode)
        return (less, more) in self.UndirectedEdges

    def add_undi_edge(self, fromnode, tonode):
        less, more = (fromnode, tonode) if (fromnode < tonode) else (tonode, fromnode)
        if not self.has_undi_edge(less, more):
            self.UndirectedEdges.add((less, more))
            self.Nodes[less].AddNeighbor(more)
            self.Nodes[more].AddNeighbor(less)

    def del_undi_edge(self, fromnode, tonode):
        less, more = (fromnode, tonode) if (fromnode < tonode) else (tonode, fromnode)
        if self.has_undi_edge(less, more):
            self.UndirectedEdges.remove((less, more))
            self.Nodes[less].DelNeighbor(more)
            self.Nodes[more].DelNeighbor(less)

    def getTo(self, x):
        return self.Nodes[x].GetTo()

    def getFrom(self, x):
        return self.Nodes[x].GetFrom()

    def getNeighbor(self, x):
        return self.Nodes[x].GetNeighbor()
    
    def reachable(self, x, y, l):
        # l-hop reachability
        visited = set()
        paths = [[x]]
        while len(paths) > 0:
            curr_path = paths.pop(0)
            curr_node = curr_path[-1]
            if curr_node in visited: continue
            else: visited.add(curr_node)
            for neighbor in self.getNeighbor(curr_node):
                if neighbor == y: return len(curr_path)
                if len(curr_path) <= l and neighbor not in visited:
                    new_path = copy.copy(curr_path)
                    new_path.append(neighbor)
                    paths.append(new_path)
        return -1

    def getCPDAG(self):
        pdag = MixedGraph(nodeIDs=self.NodeIDs)
        for (fromnode, tonode) in self.DirectedEdges:
            if (fromnode, tonode) in self.IdentifiableEdges:
                pdag.add_di_edge(fromnode, tonode)
            else:
                pdag.add_undi_edge(fromnode, tonode)
        for (fromnode, tonode) in self.UndirectedEdges:
            if (fromnode, tonode) in self.IdentifiableEdges:
                pdag.add_di_edge(fromnode, tonode)
            elif (tonode, fromnode) in self.IdentifiableEdges:
                pdag.add_di_edge(tonode, fromnode)
            else:
                pdag.add_undi_edge(fromnode, tonode)
        return pdag

    def getAdjacencyMatrix(self):
        adjmat = np.zeros((len(self.NodeIDs), len(self.NodeIDs)), dtype=int)
        if self.DirectedEdges:
            di_inds = tuple(np.array(list(self.DirectedEdges)).T)
            adjmat[di_inds] = 1
        if self.UndirectedEdges:
            undi_inds = tuple(np.array(list(self.UndirectedEdges)).T)
            adjmat[undi_inds] = 1
            adjmat[(undi_inds[1], undi_inds[0])] = 1
        return adjmat
    
    def Compare(self, truth_skeleton):
        truth_skeleton: MixedGraph
        precision = len(truth_skeleton.UndirectedEdges.intersection(self.UndirectedEdges)) / max(len(self.UndirectedEdges), 1)
        recall = len(truth_skeleton.UndirectedEdges.intersection(self.UndirectedEdges)) / max(len(truth_skeleton.UndirectedEdges), 1)
        f1 = 2*precision*recall/(precision+recall) if precision+recall!=0 else 0

        return {"F1": f1, "Precision": precision, "Recall": recall}

    def apply_meek_rules(self):
        def _apply_rule(y, meek_rule_id):
            ''' nested function, changes enclosing variables stack and pdag
            :param y: int, the popped nodeID
            :param meek_rule_id:
            '''
            y_neighbors = self.getNeighbor(y) # deepcopy of the neighbors set NOW. edit further won't change it
            if len(y_neighbors) == 0: return
            meek_rule = [_meek_rule_1, _meek_rule_2, _meek_rule_3, _meek_rule_4][meek_rule_id]
            for x in y_neighbors:
                if meek_rule(x, y):
                    self.del_undi_edge(x, y)
                    self.add_di_edge(x, y)
                    stack.append(y)
                elif meek_rule(y, x):
                    self.del_undi_edge(x, y)
                    self.add_di_edge(y, x)
                    stack.append(x)

        def _meek_rule_1(x, y):
            ''' return bool
            /// meek rule 1:
            /// look for chain z --> x --- y, where !adj(y, z). If so,
            /// orient x --> y
            '''
            for z in self.getFrom(x):
                if not self.adjacent_in_mixed_graph(y, z):
                    return True
            return False

        def _meek_rule_2(x, y):
            '''
            /// meek rule 2: x --- y, look for z such that, z --> y, x --> z. If
            /// so, orient x --> y to prevent a cycle.
            '''
            for z in self.getTo(x):
                if self.has_di_edge(z, y):
                    return True
            return False

        def _meek_rule_3(x, y):
            '''
            /// meek rule 3: orient x --- y into x --> y when there exists
            /// chains x --- u --> y and x --- v --> y, with
            /// u and v not adjacent
            '''
            for u in self.getFrom(y):
                if self.has_undi_edge(x, u):
                    for v in self.getFrom(y):
                        if u != v and self.has_undi_edge(x, v) and \
                                not self.adjacent_in_mixed_graph(u, v):
                            return True
            return False

        def _meek_rule_4(x, y):
            '''
            /// meek rule4: orient x --- y into x --> y when there exists
            /// chains u --> v --> y, u --- x, with x and v adjacent (no need to be x --- v
            /// you can prove it by assuming x --- v, x --> v, x <-- v)
            /// and u, y not adjacent
            '''
            for v in self.getFrom(y):
                if self.adjacent_in_mixed_graph(v, x):
                    for u in self.getFrom(v):
                        if self.has_undi_edge(u, x) and not self.adjacent_in_mixed_graph(u, y):
                            return True
            return False

        stack = []
        for nodeID in self.NodeIDs:
            if len(self.getNeighbor(nodeID)) > 0:
                stack.append(nodeID)
        while (stack):
            node = stack.pop()
            _apply_rule(node, 0)
            _apply_rule(node, 1)
            _apply_rule(node, 2)
            _apply_rule(node, 3)

    def write_graph(self, graph_path):
        node_num = len(self.Nodes)
        mat = np.zeros((node_num, node_num), dtype=int)

        for i in self.NodeIDs:
            for j in self.NodeIDs:
                if self.adjacent_in_mixed_graph(i, j): mat[i,j] = 1
        np.savetxt(graph_path, mat, fmt='%i')