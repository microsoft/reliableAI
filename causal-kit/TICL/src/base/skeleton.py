
import copy
from base.node import Node


class MoralGraph(object):
    def __init__(self, nodeIDs):
        self.NodeIDs = nodeIDs
        self.Nodes = {i: Node() for i in self.NodeIDs}
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
            if curr_node in visited:
                continue
            else:
                visited.add(curr_node)
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
            if curr_node in visited:
                continue
            else:
                visited.add(curr_node)
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
            if curr in visited:
                continue
            else:
                visited.add(curr)
            for neighbor in self.Nodes[curr].GetNeighbor():
                if neighbor == node2: return True
                if neighbor not in visited and neighbor not in separators:
                    queue.append(neighbor)

        return False

