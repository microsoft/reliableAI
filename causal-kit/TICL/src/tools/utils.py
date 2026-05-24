import datetime
import logging
import sklearn.preprocessing
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import loggamma, digamma, polygamma



def flip_coin(p=0.5):
    """ This function is simulate flip coin, randomly generate a probability, if < p, return True, otherwise, return False.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        bool: binary result
    """
    return True if np.random.random() < p else False


def dirichlet_alpha_estimation(D, tol=1e-3, maxiter=10000):
    """Estimation of Dirichlet's parameter alpha using maximum likelihood estimation.

    Args:
        D (np.array): CPT of node(variable)
        tol (_type_, optional): _description_. Defaults to 1e-3.
        maxiter (int, optional): _description_. Defaults to 10000.

    Returns:
        float: parameter alpha
    """
    D[D == 0] = 1e-8
    D = D / D.sum(axis=1)[:, None]
    N, K = D.shape
    logp = np.mean(np.log(D), axis=0)
    euler = -1 * digamma(1)

    def _likelihood(alpha):
        obj = N * loggamma(np.sum(alpha)) - N * np.sum(loggamma(alpha)) + N * np.sum((alpha - 1) * logp)
        return obj

    def _inverse_digamme(y, tol=1e-5, maxiter=100):
        def _trigamma(x):
            return polygamma(1, x)
        
        x0 = np.piecewise(y, [y >= -2.22, y < -2.22], [(lambda x: np.exp(x) + 0.5), (lambda x: -1 / (x + euler))])
        for _ in range(maxiter):
            x1 = x0 - (digamma(x0) - y) / _trigamma(x0)
            if np.linalg.norm(x1 - x0) < tol:
                return x1
            else:
                x0 = x1
        return x0

    a0 = np.ones(K)

    for _ in range(maxiter):
        a1 = _inverse_digamme(digamma(np.sum(a0)) + logp)
        if abs(_likelihood(a0) - _likelihood(a1)) < tol:
            return a1
        a0 = a1
    a1 = _inverse_digamme(digamma(np.sum(a0)) + logp)
    return a0


def Erf(x):
    """ 
    error function. handle either positive or negative x.
    because error function is negatively symmetric of x
    reference: https://en.wikipedia.org/wiki/Error_function
    """
    a = 0.140012
    b = x ** 2
    item = -b * (4 / np.pi + a * b) / (1 + a * b)
    return np.abs(np.sqrt(1 - np.exp(item)))


def GaussianSignificance(x, u, sigma):
    '''
    calculate the statistical significance for a gaussian distribution.
    :param x: the observed x value
    :param u: mean value
    :param sigma: the standard deviation
    :return:
    '''
    x1 = np.abs(x - u)
    return Erf(x1 / sigma / 1.414213562373095)
    # cdf = 0.5 + 0.5 * Erf(x1 / sigma / 1.414213562373095)
    # return 2 * cdf - 1


def meanstdmaxmin(a):
    return [np.mean(a), np.std(a), np.max(a), np.min(a)] if len(a) else [-1., -1., -1., -1.]


def percentileEmbedding(arr, k=11):
    if len(arr) == 0: return [-1.] * k
    offset = 100 / (k - 1)
    percentiles = [i * offset for i in range(k)]
    return np.percentile(arr, percentiles)


class Kernel_Embedding(object):
    # kernel embedding from https://github.com/lopezpaz/causation_learning_theory
    def __init__(self, k=5, s=None, d=1):  # in RCC k=100
        if not s: s = [0.15, 1.5, 15]
        self.k, self.s, self.d = k, s, d
        self.w = np.hstack((  # w in shape 2*15
            np.vstack([si * np.random.randn(k, d) for si in s]),
            # shape 15*1, first 5 rows~N(0, 0.15), then ~N(0, 1.5), ~N(0, 15)
            2 * np.pi * np.random.rand(k * len(s), 1)  # shape 15*1, ~N(0, 2pi)
        )).T
    
    def get_empirical_embedding(self, a):
        # param: a (list) is the same as that in percentile(a, q): samples P_S to a distribution P
        if len(a) == 0: return [-1.] * self.k * len(self.s)  # np.ones((self.k * len(self.s))) * -1.
        arr = sklearn.preprocessing.scale(a)[:, None]  # arr = np.array(a)[:, None]
        return np.cos(np.dot(np.hstack((arr, np.ones((arr.shape[0], 1)))), self.w)).mean(axis=0).tolist()



def getLogger(exp_number):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 移除默认的 consoleHandler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    current_time_info = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fileHandler = logging.FileHandler(filename=f'./log_info/{exp_number}_{current_time_info}.log', mode='w')
    consoleHandler.setLevel(logging.INFO)
    
    consoleformatter = logging.Formatter("%(message)s")
    fileformatter = logging.Formatter("%(message)s")

    consoleHandler.setFormatter(consoleformatter)
    fileHandler.setFormatter(fileformatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger



def draw_graph(adj_matrix, graph_name=None):
    """
    This function takes an adjacency matrix as input and draws a graph using the NetworkX library.
    
    Input:
    - adj_matrix: a numpy array with 1s and 0s representing the presence or absence of edges between nodes.
    
    Output:
    - A graph with nodes and edges drawn using matplotlib.
    
    Functionality:
    - The function creates two graphs: one directed and one undirected.
    - If there is an edge between two nodes in both directions, it is added to the undirected graph.
    - Otherwise, it will be added to the directed graph when there is only one direction between two nodes.
    - The function then uses the circular layout to position the nodes and edges in the directed graph.
    - Finally, it draws the nodes and edges of both graphs using matplotlib.
    - It supports directed graphs, undirected graphs, and mixed graphs!
    """
    directed_G = nx.DiGraph()
    undirected_G = nx.Graph()
    for i in range(len(adj_matrix)):
        directed_G.add_node(i)
        undirected_G.add_node(i)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] == 1:
                if adj_matrix[j][i] == 1:
                    undirected_G.add_edge(i, j)
                else:
                    directed_G.add_edge(i, j)
    pos = nx.circular_layout(directed_G)
    nx.draw_networkx_nodes(directed_G, pos)
    nx.draw_networkx_labels(directed_G, pos)
    nx.draw_networkx_edges(directed_G, pos, edge_color='red')
    nx.draw_networkx_edges(undirected_G, pos, edge_color='green')
    plt.title(graph_name)
    plt.show()


def is_skeleton(adj_matrix):
    """
    Determine whether an adjacency matrix is an undirected graph (skeleton).
    
    Args:
    - adj_matrix: A two-dimensional array containing only 0s and 1s, indicating whether there is an edge between nodes.
    
    Returns:
    - True if the adjacency matrix is an undirected graph; False otherwise.
    
    Functionality:
    - This function traverses the upper triangular part of the adjacency matrix (excluding the diagonal) and determines whether there are symmetric edges.
    """
    for i in range(len(adj_matrix)):
        for j in range(i+1, len(adj_matrix)):
            if adj_matrix[i][j] == 1 and adj_matrix[j][i] == 1:
                return True
    
    return False



def is_dag(adj_matrix):
    """
    Determine whether an adjacency matrix represents a directed acyclic graph (DAG).
    
    Args:
    - adj_matrix: a 2D array of 0s and 1s representing the presence or absence of edges between nodes.
    
    Returns:
    - True if the adjacency matrix represents a DAG; False otherwise.
    
    Functionality:
    - This function uses topological sorting to determine whether an adjacency matrix represents a DAG.
    """

    in_degree = [0] * len(adj_matrix)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] == 1:
                in_degree[j] += 1
    
    queue = []
    for i in range(len(in_degree)):
        if in_degree[i] == 0:
            queue.append(i)
    
    count = 0
    while queue:
        node = queue.pop(0)
        count += 1
        for i in range(len(adj_matrix)):
            if adj_matrix[node][i] == 1:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)
    
    return count == len(adj_matrix)