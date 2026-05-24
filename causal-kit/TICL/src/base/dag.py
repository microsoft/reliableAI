import os
import copy
import warnings
import numpy as np
from itertools import combinations
from pgmpy.models import BayesianNetwork

from base.node import Node
from base.pdag import MixedGraph
from base.skeleton import MoralGraph


class DiGraph(object):
    def __init__(self, graph_path):
        """This function is used for 

        Args:
            graph_path (_type_): _description_
            load_type (str, optional): _description_. Defaults to "full", others are "system" / "environment".

        Raises:
            NotImplementedError: can't find suitable load type
        """
        # load_type: a string defines what part of the graph is loaded, option: full, system, environment
        self.graph_path = graph_path
        adjmat = np.loadtxt(graph_path, dtype=np.int16)
        
        self.DirectedEdges = set()
        self.NodeIDs = list(range(len(adjmat)))
        self.Nodes = {i: Node() for i in self.NodeIDs}

        for i in self.NodeIDs:
            for j in self.NodeIDs:
                if adjmat[i, j]:
                    self.add_di_edge(i, j)
        
        self._load_extra_information(graph_path)
        self.topoSort()
    
    def _load_extra_information(self, graph_path):
        
        def get_tforks_vstrucs(di_adjmat):
            undi_adjmat = di_adjmat + di_adjmat.T # the skeleton, will not have value 2
            i_j_adjacent = undi_adjmat[:, :, None].astype(bool)
            k_j_adjacent = undi_adjmat[:, None, :].astype(bool)
            i_k_not_adjacent_and_i_less_than_k = np.triu(1 - undi_adjmat, k=1)[None, :, :].astype(bool)
            cube_tforks = i_j_adjacent * k_j_adjacent * i_k_not_adjacent_and_i_less_than_k

            i_point_to_j = di_adjmat.T[:, :, None].astype(bool)
            k_point_to_j = di_adjmat.T[:, None, :].astype(bool)
            i_k_not_adjacent_and_i_less_than_k = np.triu(1 - undi_adjmat, k=1)[None, :, :].astype(bool)
            cube_vstrucs = i_point_to_j * k_point_to_j * i_k_not_adjacent_and_i_less_than_k

            return set(map(tuple, np.argwhere(cube_tforks))), set(map(tuple, np.argwhere(cube_vstrucs)))
        
        tforks_pth = graph_path.replace('.txt', '_TForks.txt')
        vstrucs_pth = graph_path.replace('.txt', '_VStrucs.txt')
        if os.path.exists(tforks_pth) and os.path.exists(vstrucs_pth):
            # consider when txt is empty, eg. sachs, or only one line.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tforks = set(map(tuple, np.loadtxt(tforks_pth, dtype=int).reshape((-1, 3))))
                self.vstrucs = set(map(tuple, np.loadtxt(vstrucs_pth, dtype=int).reshape((-1, 3))))
        else:
            # self.tforks, self.vstrucs = get_tforks_vstrucs(self.getAdjacencyMatrix())
            self.tforks, self.vstrucs = set(), set()
            for j in self.NodeIDs:
                for i in self.NodeIDs:
                    for k in self.NodeIDs:
                        if k <= i or j == i or j == k: continue
                        if self.is_adjacent(i, j) and self.is_adjacent(k, j) and not self.is_adjacent(i, k):
                            self.tforks.add((j, i, k))
                            if self.has_di_edge(i, j) and self.has_di_edge(k, j):
                                self.vstrucs.add((j, i, k))

            np.savetxt(tforks_pth, np.array(sorted(list(self.tforks)), dtype=int), fmt='%i')
            np.savetxt(vstrucs_pth, np.array(sorted(list(self.vstrucs)), dtype=int), fmt='%i')
        
        identifiable_pth = graph_path.replace('.txt', '_Identifiables.txt')
        if os.path.exists(identifiable_pth):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.IdentifiableEdges = set(map(tuple, np.loadtxt(identifiable_pth, dtype=int).reshape((-1, 2))))
        else:
            pdag = MixedGraph(nodeIDs=self.NodeIDs) # 初始化部分有向无环图(pdag)类
            for (j, i, k) in self.vstrucs: # 通过vstrucs确定部分有向边
                pdag.add_di_edge(i, j)
                pdag.add_di_edge(k, j)
            for (fromnode, tonode) in self.DirectedEdges:  # 通过v-strucs确定不了的剩余的无向边（真实图所有方向 - vstrucs确定的部分有向边）
                if not pdag.has_di_edge(fromnode, tonode):
                    pdag.add_undi_edge(fromnode, tonode)
            pdag.apply_meek_rules()  # 应用4条规则，确定无向边中可进一步定向的边
            self.IdentifiableEdges = pdag.DirectedEdges  # 可识别的边就是两阶段(vstruct + 4 meek rules)后pdag中的有向边
            assert self.IdentifiableEdges.issubset(self.DirectedEdges)  # 可识别的边的集合属于dag有向边的子集
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

    def withinThreeHop(self, x, y):
        oneHopPc = self.getPC(x)
        if y in self.getPC(x): return 1
        twoHopPC = set()
        for neighbor in oneHopPc:
            twoHopPC = twoHopPC.union(self.getPC(neighbor))
        if y in twoHopPC: return 1 / 2
        threeHopPC = set()
        for neighbor in threeHopPC:
            threeHopPC = threeHopPC.union(self.getPC(neighbor))
        if y in threeHopPC: return 1 / 3
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
            if y in twoHopPC: return 1 / 2
            threeHopPC = set()
            for neighbor in threeHopPC:
                threeHopPC = threeHopPC.union(self.getTo(neighbor))
            if y in threeHopPC: return 1 / 3
            return 0

        return xToY(x, y)

    def localityWeightOneFaithful(self, x, y):
        def xToY(x, y):
            oneHopPc = self.getTo(x)
            if y in self.getTo(x): return 1
            twoHopPC = set()
            for neighbor in oneHopPc:
                twoHopPC = twoHopPC.union(self.getTo(neighbor))
            if y in twoHopPC: return 1 / 2
            threeHopPC = set()
            for neighbor in threeHopPC:
                threeHopPC = threeHopPC.union(self.getTo(neighbor))
            if y in threeHopPC: return 1 / 3
            return 0

        xy = xToY(x, y)
        if xy != 0:
            return xy
        else:
            return xToY(y, x)

    def getWeight(self, x, y, l, k):
        hops = self.reachable(x, y, l)
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
            if curr_node in visited:
                continue
            else:
                visited.add(curr_node)
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
        if l == 1:
            return self.getFrom(x)
        else:
            ancestors = set()
            for ancestor in self.getFrom(x):
                ancestors = ancestors.union(self.getAncestor(ancestor, l - 1))
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

    def getSkeleton(self):
        skeleton = MixedGraph(numberOfNodes=len(self.NodeIDs))
        for edge in self.DirectedEdges:
            skeleton.add_undi_edge(edge[0], edge[1])
        return skeleton
    
    def getBN(self):
        bn = BayesianNetwork()
        bn.add_nodes_from(map(str, self.NodeIDs))
        bn.add_edges_from([(str(e[0]), str(e[1])) for e in self.DirectedEdges])
        return bn

