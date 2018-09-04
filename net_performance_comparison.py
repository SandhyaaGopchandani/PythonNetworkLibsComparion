import itertools
import numpy as np
from timeit import default_timer as timer
from graph_tool.all import *
import pickle
import networkx as nx
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
from igraph import *


def nodes_edges(num_nodes):
    """ this function takes number of nodes and returns nodes and edge list"""
    nodes = list(range(num_nodes))
    edges = list(itertools.combinations(nodes, 2))
    return nodes, edges

def create_graph_graphtool(node_num, edges):
    """ this function creates graph object of graphtool library"""
    g = Graph(directed=False)
    vlist = g.add_vertex(node_num)
    g.add_edge_list(edges)
    return g

def create_graph_igraph(nodes, edges):
    """ this function creates graph object of igraph library"""
    g = Graph(directed=False)
    g.add_vertices(nodes)
    g.add_edges(edges)
    return g

def create_graph_networkx(nodes, edges):
    """ this function creates graph object of networkx library"""
    g = nx.Graph(directed=False)
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def get_edges(complete_edge_list, threshold=0.5):
    """ this function randomnly picks the edges in graph based on probability. 0.5 means we want to include only 50% of random 
    edges of the total edges in the graph"""
    edge_list = []
    for key in complete_edge_list:
        if np.random.random() < threshold:
            edge_list.append(key)

    return edge_list


def multiple_graph(complete_edge_list, nodes, probs, netlib='networkx'):
    """this function times the various centrality measures calculated using three different network libararies.
    The function computes various graph based on given probability of edges, computes the degree, closeness and betweenness
    centrality measure and time those. At the end, it returns the list of timestamp for each cenrality. """
    print("total possible edges:", len(complete_edge_list))
    time_deg_central = []
    time_closeness_central = []
    time_between_central = []
    num_edges = []
    for prob in probs:
        edges = get_edges(complete_edge_list, prob)
        if netlib == 'graph-tool':
            num_nodes = len(nodes)
            graph = create_graph_graphtool(num_nodes, edges)
            print(prob, len(graph.get_vertices()), len(graph.get_edges()))
            num_edges.append(len(graph.get_edges()))

            start = timer()
            doc_degree_centralities = graph.get_out_degrees(nodes)
            end = timer()
            time_deg_central.append(end - start)

            start = timer()
            vertex_betweenness, edge_betweenness = graph_tool.centrality.betweenness(graph)
            end = timer()
            time_between_central.append(end - start)

            start = timer()
            vertex_closeness = graph_tool.centrality.closeness(graph)
            end = timer()
            time_closeness_central.append(end - start)

        if netlib == 'networkx':
            graph = create_graph_networkx(nodes, edges)
            print(prob, len(graph.nodes()), len(graph.edges()))
            num_edges.append(len(graph.edges()))

            start = timer()
            doc_degree_centralities = nx.algorithms.centrality.degree_centrality(graph)
            end = timer()
            time_deg_central.append(end - start)

            start = timer()
            vertex_betweenness = nx.algorithms.centrality.betweenness_centrality(graph)
            end = timer()
            time_between_central.append(end - start)

            start = timer()
            vertex_closeness = nx.algorithms.centrality.closeness_centrality(graph)
            end = timer()
            time_closeness_central.append(end - start)

        if netlib == 'igraph':
            graph = create_graph_igraph(nodes, edges)
            print(prob, graph.vcount(), graph.ecount())
            num_edges.append(graph.ecount())

            start = timer()
            doc_degree_centralities = np.array(graph.degree(nodes), dtype='f') / (graph.vcount() - 1)
            end = timer()
            time_deg_central.append(end - start)

            start = timer()
            normalization_factor = 2 / (float(graph.vcount() - 1) * float(graph.vcount() - 2))
            vertex_betweenness = np.array(graph.betweenness(), dtype='f') * normalization_factor
            end = timer()
            time_between_central.append(end - start)

            start = timer()
            vertex_closeness = graph.closeness()
            end = timer()
            time_closeness_central.append(end - start)

    return num_edges, time_deg_central, time_closeness_central, time_between_central


def plot_result(num_nodes, x, y1, y2, y3):
    """This function plots the timestamp for three centralities as a function of number of edges."""
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.legend(['degree centrality', 'closeness centrality','betweenness centrality'], loc='upper left')
    plt.xticks(x)
    plt.title('with network of nodes '+str(num_nodes))
    plt.xticks(rotation=90)
    plt.xlabel('number of edges')
    plt.ylabel('time (in seconds)')
    plt.show()


if __name__ == '__main__':
    
    num_nodes = 500  # number of nodes
    nodes, complete_edge_list = nodes_edges(num_nodes)
    threshold = [0.05, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    num_edges, time_deg_central, time_closeness_central, time_between_central = multiple_graph(complete_edge_list,
                                                                                               nodes, threshold,
                                                                                               netlib='igraph')
    print(num_edges, time_deg_central, time_closeness_central, time_between_central)
    plot_result(num_nodes, num_edges, time_deg_central, time_closeness_central, time_between_central)