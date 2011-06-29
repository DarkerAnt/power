#!/usr/bin/python

from lpsolve55 import *
import networkx as nx
import matplotlib.pyplot as plt

def display_graph(graph):
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    edgecapwidth = [d['capacity'] for (u,v,d) in edges]
    edgeloadwidth = [d['load'] for (u,v,d) in edges]
    labels = [str(d['load']) + '/' + str(d['capacity']) for u,v,d in edges]
    e_labels=dict(zip(graph.edges(), labels))
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=100)
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=edgecapwidth,edge_color = 'b')
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=edgeloadwidth,edge_color = 'r')
    labels=dict(zip(graph.edges(),[d for u,v,d in graph.edges(data=True)]))
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=e_labels)
    plt.show()
    
def init_graph():
    graph = nx.generators.classic.grid_2d_graph(2, 2)
    #graph = graph.to_directed()
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    
    for (name,data) in nodes:
        data['load'] = 0
    name,data = nodes[0]
    data['load'] = 5.0
    name,data = nodes[3]
    data['load'] = -5.0
        
    for u,v,data in edges:
        data['capacity'] = 3.0
        data['load'] = 0.0
    u,v,data = edges[0]
    data['load'] = 1.5
    return graph

if __name__ == "__main__":
   
    graph = init_graph()
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    
    #lp = lpsolve('make_lp')
    #lpsolve('set_verboxe', lp, IMPORTANT)
    #lpsolve('set_minim', lp)
    #lpsolve('add_constraint', lp,
    #name,data = nodes[0]
    #print graph.neighbors(name)
    #print graph.node[name]
    #print graph.node

    #for name,data in graph.node.items():
    #    for cname in graph.neighbors(name):
    #        cname,cdata = graph.node[cname]
     
    #for u,v,data in edges:
    #    lpsolve('add_constraint', lp, , LE, -
    display_graph(graph)

