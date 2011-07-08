#!/usr/bin/python

from lpsolve55 import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def display_graph(graph):
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    edgecapwidth = [d['capacity'] for (u,v,d) in edges]
    edgeloadwidth = [d['load'] for (u,v,d) in edges]
    labels = [str(d['load']) + '/' + str(d['capacity']) for u,v,d in edges]
    e_labels=dict(zip(graph.edges(), labels))
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=200)
    nx.draw_networkx_labels(graph,pos)
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=edgecapwidth,edge_color = 'b')
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=edgeloadwidth,edge_color = 'r')
    labels=dict(zip(graph.edges(),[d for u,v,d in graph.edges(data=True)]))
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=e_labels)
    plt.show()
    
def init_graph():
    #graph = nx.generators.classic.grid_2d_graph(2, 2)
    #graph = graph.to_directed()
    graph = nx.DiGraph()
    #graph.add_nodes_from(['A','B','C','D'])
    graph.add_edges_from([('A','B'),('B','C'),('C','D')])
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    
    for (name,data) in nodes:
        data['load'] = 0
    name,data = nodes[0]
    data['load'] = 5.0
    name,data = nodes[3]
    data['load'] = -5.0
        
    for u,v,data in edges:
        data['capacity'] = 6.0
        data['load'] = 0.0
    u,v,data = edges[0]
    data['load'] = 1.5
    return graph

if __name__ == "__main__":
   
    graph = init_graph()
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    lp = lpsolve('make_lp', 0, len(edges))

    objfn = np.zeros(len(edges))
    # use adj list
    edge_index = dict(zip(graph.edges(), range(len(edges))))
    node_index = dict(zip(graph.nodes(), range(len(nodes))))

    con_matrix = np.zeros((len(nodes), len(edges)))
    for u,v,d in edges:
        j = edge_index[(u,v)]
        iu = node_index[u]
        iv = node_index[v]
        con_matrix[iu,j] = 1
        con_matrix[iv,j] = -1
    
    # source/sink constraints
    for name,data in nodes:
        i = node_index[name]
        load = data['load']
        print name, con_matrix[i]
        if load > 0: # source
            lpsolve('add_constraint', lp, con_matrix[i], LE, load)
            print "constraint:", con_matrix[i], "<=", load
            lpsolve('add_constraint', lp, con_matrix[i], GE, 0)
            print "constraint:", con_matrix[i], ">=", 0
        elif load < 0: # sink
            lpsolve('add_constraint', lp, con_matrix[i], GE, load)
            print "constraint:", con_matrix[i], ">=", load
            lpsolve('add_constraint', lp, con_matrix[i], LE, 0)
            print "constraint:", con_matrix[i], "<=", 0
        else:
            lpsolve('add_constraint', lp, con_matrix[i], EQ, 0)
            print "constraint:", con_matrix[i], "=", load

    # objective function
    source_weight = 1
    sink_weight = -100
    objfn = np.zeros(len(edges))
    for u,v,d in edges:
        i = edge_index[(u,v)]
        data = graph.node[v]
        load = data['load']
        if load > 0: # source
            objfn[i] = source_weight
        elif load < 0:
            objfn[i] = sink_weight
    
    print "objective func:", objfn
    lpsolve('set_verbose', lp, IMPORTANT)
    lpsolve('set_minim', lp)
    lpsolve('set_obj_fn', lp, objfn)
    
    # set var upper and lower bounds
    bounds = np.zeros(len(edges))
    for u,v,d in edges:
        i = edge_index[(u,v)]
        bounds[i] = d['capacity']
    lpsolve('set_upbo', lp, bounds)
    lpsolve('set_lowbo', lp, -bounds)
    #lpsolve('set_mat', lp, adjmatrix)
    #lpsolve('set_rh_vec', lp, init_loads)
    lpsolve('set_outputfile', lp, "")
    lpsolve('print_lp',lp)
    result = lpsolve('solve',lp)

    solution = lpsolve('get_variables', lp)[0]
    for u,v,d in edges:
        i = edge_index[(u,v)]
        d['load'] = solution[i]
    lpsolve('delete_lp',lp)
    display_graph(graph)
