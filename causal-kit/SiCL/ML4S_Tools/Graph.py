# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env python
#-*- coding: utf-8 -*-
import copy, os
import numpy as np
import warnings

class Node(object):
    def __init__(self):
        self.To = set()
        self.From = set()
        self.Neighbor = set() # specific for undirected edges
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

        tforks_pth = graphtxtpath.replace('.txt', '_TForks.txt')
        vstrucs_pth = graphtxtpath.replace('.txt', '_VStrucs.txt')
        if os.path.exists(tforks_pth) and os.path.exists(vstrucs_pth):
            # consider when txt is empty, eg. sachs, or only one line.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.vstrucs = set(map(tuple, np.loadtxt(vstrucs_pth, dtype=int).reshape((-1, 3))))
                self.tforks = set(map(tuple, np.loadtxt(tforks_pth, dtype=int).reshape((-1, 3))))
        else:
            # returns in, and cube indexed in format (j, i, k), where i--j--k, and i < k
            di_adjmat = self.getAdjacencyMatrix()
            undi_adjmat = di_adjmat + di_adjmat.T # the skeleton, will not have value 2
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
            #             if self.is_adjacent(i, j) and self.is_adjacent(k, j) and not self.is_adjacent(i, k):
            #                 self.tforks.add((j, i, k))
            #                 if self.has_di_edge(i, j) and self.has_di_edge(k, j):
            #                     self.vstrucs.add((j, i, k))
            ###################### ###################### ###################### ##################
            ###################### ###################### ###################### ##################

            np.savetxt(vstrucs_pth, np.array(sorted(list(self.vstrucs)), dtype=int), fmt='%i')
            np.savetxt(tforks_pth, np.array(sorted(list(self.tforks)), dtype=int), fmt='%i')

        identifiable_pth = graphtxtpath.replace('.txt', '_Identifiables.txt')
        if os.path.exists(identifiable_pth):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.IdentifiableEdges = set(map(tuple, np.loadtxt(identifiable_pth, dtype=int).reshape((-1, 2))))
        else:
            pdag = MixedGraph(nodeIDs=self.NodeIDs)
            for (j, i, k) in self.vstrucs:
                pdag.add_di_edge(i, j)
                pdag.add_di_edge(k, j)
            for (fromnode, tonode) in self.DirectedEdges:
                if not pdag.has_di_edge(fromnode, tonode):
                    pdag.add_undi_edge(fromnode, tonode)
            pdag.apply_meek_rules()
            self.IdentifiableEdges = pdag.DirectedEdges
            assert self.IdentifiableEdges.issubset(self.DirectedEdges)
            del pdag  # memory to self.IdentifiableEdges (original tmp_mix.DirectedEdges) still holds
            np.savetxt(identifiable_pth, np.array(sorted(list(self.IdentifiableEdges)), dtype=int), fmt='%i')

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

    def getTo(self, x):
        return self.Nodes[x].GetTo()

    def getFrom(self, x):
        return self.Nodes[x].GetFrom()

    def getPC(self, x):
        return self.Nodes[x].GetTo().union(self.Nodes[x].GetFrom())

class MixedGraph(object):
    def __init__(self, numberOfNodes=0, nodeIDs=None, graphtxtpath=None):
        if graphtxtpath is None:
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

            tforks_pth = graphtxtpath.replace('.txt', '_TForks.txt')
            vstrucs_pth = graphtxtpath.replace('.txt', '_VStrucs.txt')

            if os.path.exists(tforks_pth) and os.path.exists(vstrucs_pth):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore") # warning when txt is empty, eg. sachs
                    self.tforks = set(map(tuple, np.loadtxt(tforks_pth, dtype=int).reshape((-1, 3))))
                    self.vstrucs = set(map(tuple, np.loadtxt(vstrucs_pth, dtype=int).reshape((-1, 3))))

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

            identifiable_pth = graphtxtpath.replace('.txt', '_Identifiables.txt')
            if os.path.exists(identifiable_pth):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
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

    def getPC(self, x):
        return self.Nodes[x].GetTo().union(self.Nodes[x].GetFrom()).union(self.Nodes[x].GetNeighbor())

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

