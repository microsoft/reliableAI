# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
from enum import Enum
from typing import Iterable, List, Set, Dict
from copy import copy

class CausalSemanticModel:

    maximal_search_distance = 4

    class SemanticType(Enum):
        # adjacent cases
        DirectCause=1
        PossiblyDirectCause=2
        DirectEffect=3
        PossiblyDirectEffect = 4
        DirectRelevance = 5

        # non-adjacent cases
        IndirectCause=6
        PossiblyIndirectCause=7
        IndirectEffect=8
        PossiblyIndirectEffect = 9
        Unexplainable = 10

    def __init__(self, col_names: List[str], edges: Set[Edge]) -> None:
        self.col_names = col_names
        self.edges = {}
        self.edges: Dict[str, Set[Edge]]
        for edge in edges:
            self.edges.setdefault(edge.start, set()).add(edge)
            self.edges.setdefault(edge.end, set()).add(edge)
    
    def add_edge(self, edge: Edge):
        self.edges[edge.start].add(edge)
        self.edges[edge.end].add(edge)

    def check_semantic(self, target_col: str, explanation_col: str, condition_set: Iterable[str]):
        if target_col == explanation_col: return CausalSemanticModel.SemanticType.Unexplainable 
        adjacent_edges = self.edges[target_col]
        for edge in adjacent_edges:
            edge: Edge
            if edge.start == explanation_col:
                if edge.edge_type == Edge.EdgeType.Direct:
                    return CausalSemanticModel.SemanticType.DirectCause
                elif edge.edge_type == Edge.EdgeType.SemiDirect:
                    return CausalSemanticModel.SemanticType.PossiblyDirectCause
                elif edge.edge_type == Edge.EdgeType.BiDirect:
                    return CausalSemanticModel.SemanticType.DirectRelevance
            if edge.end == explanation_col:
                if edge.edge_type == Edge.EdgeType.Direct:
                    return CausalSemanticModel.SemanticType.DirectEffect
                elif edge.edge_type == Edge.EdgeType.SemiDirect:
                    return CausalSemanticModel.SemanticType.PossiblyDirectEffect
                elif edge.edge_type == Edge.EdgeType.BiDirect:
                    return CausalSemanticModel.SemanticType.DirectRelevance
        
        direct_descendant = set()
        semidirect_decendant = set()
        direct_ancestors = set()
        semidirect_ancestors = set()
        paths = []
        for edge in self.edges[target_col]:
            if (edge.edge_type == Edge.EdgeType.Direct or edge.edge_type == Edge.EdgeType.SemiDirect) and edge.start not in condition_set and edge.end not in condition_set:
                paths.append(Path.initial_path(target_col, edge, condition_set))
        while len(paths) != 0:
            path = paths.pop(0)
            path: Path
            
            if path.size() < CausalSemanticModel.maximal_search_distance:
                paths += path.fork_paths(self.edges[path.current])
            else:
                if path.path_type == Path.PathType.Forward and path.is_all_direct():
                    direct_descendant = direct_descendant.union(path.get_all())
                elif path.path_type == Path.PathType.Forward and not path.is_all_direct():
                    semidirect_decendant = semidirect_decendant.union(path.get_all())
                if path.path_type == Path.PathType.Backward and path.is_all_direct():
                    direct_ancestors = direct_ancestors.union(path.get_all())
                elif path.path_type == Path.PathType.Backward and not path.is_all_direct():
                    semidirect_ancestors = semidirect_ancestors.union(path.get_all())
                    
        if explanation_col in direct_ancestors: return CausalSemanticModel.SemanticType.IndirectCause
        elif explanation_col in direct_descendant: return CausalSemanticModel.SemanticType.IndirectEffect
        elif explanation_col in semidirect_ancestors: return CausalSemanticModel.SemanticType.PossiblyIndirectCause 
        elif explanation_col in semidirect_decendant: return CausalSemanticModel.SemanticType.PossiblyIndirectEffect

        return CausalSemanticModel.SemanticType.Unexplainable 
    
    def __str__(self) -> str:
        output_str = ""
        for node in self.edges:
            for edge in self.edges[node]:
                if node == edge.start:
                    output_str += f"{edge}\n"
        return output_str
    
    def graphviz(self) -> str:
        output_str = "digraph G {\n"
        for node in self.edges:
            for edge in self.edges[node]:
                if node == edge.start:
                    if edge.edge_type == Edge.EdgeType.Direct:
                        output_str += f"\"{edge.start}\" -> \"{edge.end}\" [dir=forward];\n"
                    elif edge.edge_type == Edge.EdgeType.SemiDirect:
                        output_str += f"\"{edge.start}\" -> \"{edge.end}\" [dir=both, arrowtail = dot];\n"
                    elif edge.edge_type == Edge.EdgeType.BiDirect:
                        output_str += f"\"{edge.start}\" -> \"{edge.end}\" [dir=both];\n"
                    elif edge.edge_type == Edge.EdgeType.NonDirect:
                        output_str += f"\"{edge.start}\" -> \"{edge.end}\" [dir=both, arrowhead = dot, arrowtail=dot];\n"
        output_str += "}"
        return output_str

class Edge:

    class EdgeType(Enum):
        Direct = 1
        SemiDirect = 2
        BiDirect = 3
        NonDirect = 4

    def __init__(self, start: str, end: str, edge_type: Edge.EdgeType) -> None:
        self.edge_type = edge_type
        if self.edge_type == Edge.EdgeType.BiDirect or self.edge_type == Edge.EdgeType.NonDirect:
            if start < end:
                self.start = start
                self.end = end
            else:
                self.start = end
                self.end = start
        else:
            self.start = start
            self.end = end

    def __str__(self) -> str:
        if self.edge_type == Edge.EdgeType.Direct: edge_str = "-->"
        elif self.edge_type == Edge.EdgeType.SemiDirect: edge_str = "o->"
        elif self.edge_type == Edge.EdgeType.BiDirect: edge_str = "<->"
        else: edge_str = "o-o"
        return f"{self.start} {edge_str} {self.end}"
    
    @staticmethod
    def full_match(target:Edge, other: Edge) -> bool:
        return str(target) == str(other)
        # if (self.edge_type == Edge.EdgeType.BiDirect or self.edge_type == Edge.EdgeType.NonDirect) and (self.start == other.end and self.end == other.start): return 1

    @staticmethod
    def partial_match(target: Edge, other: Edge) -> bool:
        if Edge.full_match(target, other): return True
        # a o-> b vs. a --> b
        if target.edge_type == Edge.EdgeType.SemiDirect and other.edge_type == Edge.EdgeType.Direct \
            and target.start == other.start and target.end == other.end: return True
        if other.edge_type == Edge.EdgeType.SemiDirect and target.edge_type == Edge.EdgeType.Direct \
            and target.start == other.start and target.end == other.end: return True
        # a o-> b vs. a <-> b
        if target.edge_type == Edge.EdgeType.SemiDirect and other.edge_type == Edge.EdgeType.BiDirect \
            and target.start == other.start and target.end == other.end: return True
        if other.edge_type == Edge.EdgeType.SemiDirect and target.edge_type == Edge.EdgeType.BiDirect \
            and target.start == other.start and target.end == other.end: return True
        # a o-> b vs. b <-> a
        if target.edge_type == Edge.EdgeType.SemiDirect and other.edge_type == Edge.EdgeType.BiDirect \
            and target.end == other.start and target.start == other.end: return True
        if other.edge_type == Edge.EdgeType.SemiDirect and target.edge_type == Edge.EdgeType.BiDirect \
            and target.end == other.start and target.start == other.end: return True
        # a o-> b vs a o-o b
        if target.edge_type == Edge.EdgeType.SemiDirect and other.edge_type == Edge.EdgeType.NonDirect \
            and target.start == other.start and target.end == other.end: return True
        if other.edge_type == Edge.EdgeType.SemiDirect and target.edge_type == Edge.EdgeType.NonDirect \
            and target.start == other.start and target.end == other.end: return True
        # a o-> b vs. b o-o a
        if target.edge_type == Edge.EdgeType.SemiDirect and other.edge_type == Edge.EdgeType.NonDirect \
            and target.end == other.start and target.start == other.end: return True
        if other.edge_type == Edge.EdgeType.SemiDirect and target.edge_type == Edge.EdgeType.NonDirect \
            and target.end == other.start and target.start == other.end: return True
        # a o-o b vs a *-* b
        if target.edge_type == Edge.EdgeType.NonDirect and ((target.start == other.start and target.end == other.end) \
            or (target.start == other.end and target.end == other.start)): return True
        if other.edge_type == Edge.EdgeType.NonDirect and ((target.start == other.start and target.end == other.end) \
            or (target.start == other.end and target.end == other.start)): return True
        
        return False
    
    @staticmethod
    def edge_match(target: Edge, other: Edge) -> bool:
        return (target.start == other.start and target.end == other.end) or (target.start == other.end and target.end == other.start)



class Path:
    class PathType(Enum):
        Forward = 1
        Backward = 2

    def __init__(self, start: str, current: str, edges: List[Edge], path_type: Path.PathType, condition_set: Iterable[str]) -> None:
        self.start = start
        self.current = current
        self.edges = edges
        self.path_type = path_type
        self.condition_set = condition_set
    
    @staticmethod
    def initial_path(target_col: str, initial_edge:Edge, condition_set: Iterable[str]) -> Path:
        start = target_col
        if target_col == initial_edge.start:
            current = initial_edge.end
            path_type = Path.PathType.Forward
        else:
            current = initial_edge.start
            path_type = Path.PathType.Backward
        edges = [initial_edge]
        return Path(start, current, edges, path_type, condition_set)

    def fork_paths(self, incoming_edges: List[Edge]):
        forked_paths = []
        for incoming_edge in incoming_edges:
            if incoming_edge in self.edges: continue
            if incoming_edge.edge_type == Edge.EdgeType.BiDirect or incoming_edge.edge_type == Edge.EdgeType.NonDirect: continue
            if self.path_type == Path.PathType.Forward and self.current == incoming_edge.start:
                current = incoming_edge.end
            elif self.path_type == Path.PathType.Backward and self.current == incoming_edge.end:
                current = incoming_edge.start
            else: continue
            if current not in self.condition_set:
                forked_paths.append(Path(self.start, current, copy(self.edges)+[incoming_edge], self.path_type, self.condition_set))
        return forked_paths
    
    def is_all_direct(self):
        return len([edge for edge in self.edges if edge.edge_type == Edge.EdgeType.SemiDirect]) == 0
    
    def get_all(self):
        nodes = set()
        for edge in self.edges:
            nodes.add(edge.start)
            nodes.add(edge.end)
        return nodes
    
    def size(self):
        return len(self.edges)
    
    def __str__(self) -> str:
        output_str = f"{self.path_type}\n"
        for edge in self.edges:
            output_str += f"{edge}\n"
        return output_str