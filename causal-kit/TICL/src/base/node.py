import copy

class Node(object):
    def __init__(self):
        self.To = set()
        self.From = set()
        self.Neighbor = set()  # for undirected edges

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