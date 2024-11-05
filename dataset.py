import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler


# TODO: define dataset for unweighted graphs (I cannot think of too application and therefore have not written one)
# TODO: add plotting functions when specification is clear
# TODO: add code for using graph dataset once the main notebook is updated


"""
Dataset where each point is a position in space
"""
class NumericalData:
    def __init__(self,
                 X=None,
                 y=None,
                 max_samples=None,
                 to_float32=True,
                 standardize=True,
                 seed=None):
        """
        If applicable, convert to float32 --> standardize X --> subsample
        X and y are expected to be numpy arrays of shape (n_samples, n_features) and (n_samples, ) respectively
        """
        self.X=X
        self.y=y
        self.seed=seed
        
        if to_float32:
            self.convert_to_float32()
            
        if standardize:
            self.standardize_X()
        
        if max_samples:
            self.truncate(max_samples, seed)
            
    def convert_to_float32(self):
        self.X = self.X.astype("float32")
        
    def standardize_X(self):
        self.X = StandardScaler().fit_transform(self.X)
        
    def truncate(self, max_samples, seed):
        """subsample if needed"""
        if max_samples and max_samples < self.X.shape[0]:
            if seed:
                np.random.seed(seed)
            idx = np.random.choice(self.X.shape[0], max_samples, replace=False)
            self.X = self.X[idx]
            self.y = self.y[idx]
            
    def n_samples(self):
        return self.X.shape[0]
        

"""
Undirected Graph Dataset
"""
class UndirectedWeightedGraphData:
    def __init__(self, G=None, label=None):
        self.G = G
        self.label = label
    def n_nodes(self):
        return self.G.number_of_nodes()
    def n_edges(self):
        return self.G.number_of_edges()
    """average degree per node"""
    def avg_deg(self):
        return float(sum([val[1] for val in self.G.degree()])) / float(self.G.number_of_nodes()) / 2
    """average weight per edge"""
    def avg_weight(self):
        total_weight = 0
        for node_adjacency in self.G.edges(data=True):
            total_weight += node_adjacency[2]["weight"]
        avg_weight = total_weight / float(self.G.number_of_edges()) / 2
        return avg_weight
    def knn(self, k):
        for node in self.G:
            edges = sorted(self.G.edges(node, data=True), key=lambda edge: edge[2]["weight"], reverse=True)
            deg = len(edges)
            for i in range(deg):
                if i >= k:
                    self.G.remove_edge(edges[i][0], edges[i][1])          
    def eps_nn(self, eps):
        for node in self.G:
            edges = sorted(self.G.edges(node, data=True), key=lambda edge: edge[2]["weight"], reverse=True)
            deg = len(edges)
            for i in range(deg):
                if edges[i][2]["weight"] < eps:
                    self.G.remove_edge(edges[i][0], edges[i][1])       
    def k_and_eps_nn(self, k, eps):
        for node in self.G:
            edges = sorted(self.G.edges(node, data=True), key=lambda edge: edge[2]["weight"], reverse=True)
            deg = len(edges)
            for i in range(deg):
                if i >= k and edges[i][2]["weight"] < eps:
                    self.G.remove_edge(edges[i][0], edges[i][1])
    def generate_spring_layout(self):
        return nx.spring_layout(self.G)
        
def get_Xy_from_csv(source, label_name, delim=","):
    df = pd.read_csv(source, sep=delim)
    X = df.drop(columns=[label_name]).to_numpy()
    y = np.array(df[label_name].tolist(), dtype="string")
    return X, y
    
def get_weighted_graph_from_file(source, delim=None, nodetype=int):
    graph = nx.read_weighted_edgelist(source, delimiter=delim, nodetype=nodetype)
    return graph
