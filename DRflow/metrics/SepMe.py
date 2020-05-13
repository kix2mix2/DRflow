import networkx as nx
import numpy as np
import nglpy
from scipy.spatial.distance import euclidean


def get_cbsg(df, beta=0):
    graph = nx.Graph()
    graph.add_nodes_from(df.iterrows())

    point_set = np.array(df[["x", "y"]])
    point_set = point_set + 0.0001

    # aGraph = nglpy.Graph(point_set, "beta skeleton", 9, beta)
    aGraph = nglpy.EmptyRegionGraph(max_neighbors=9, relaxed=False, beta=beta)
    aGraph.build(point_set)
    d = aGraph.neighbors()

    for key, value in d.items():
        for v in value:
            graph.add_edge(
                key, v, weight=euclidean(df.loc[key, ["x", "y"]], df.loc[v, ["x", "y"]])
            )

    return graph