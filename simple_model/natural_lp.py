#!/usr/bin/python

from lpsolve55 import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

node_index = []
edge_index = []

def display_graph(graph, edge_width=2):
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    edgecapwidth = [d['capacity'] for (u,v,d) in edges]
    edgeloadwidth = [d['load'] for (u,v,d) in edges]
    max_cap = 1.0
    
    for i in range(len(edgeloadwidth)): # reverse edges for negative load
        if edgeloadwidth[i] < 0:
            u,v,d = edges[i]
            edges[i] = (v,u,d) 
    for cap in edgecapwidth: # get max cap to scale edge drawing
        if cap > max_cap:
            max_cap = cap
    width_mod = edge_width/(1.0*max_cap)
    for i in range(len(edgecapwidth)):
        edgecapwidth[i] *= width_mod
        edgeloadwidth[i]*= width_mod
        
    #labels = [str(d['load']) + '/' + str(d['capacity']) for u,v,d in edges]
    labels = [str(abs(d['load'])) + '/' + str(d['capacity']) for u,v,d in edges]
    e_labels=dict(zip(graph.edges(), labels))
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=300)
    nx.draw_networkx_labels(graph,pos)
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=edgecapwidth,edge_color = 'b')
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=edgeloadwidth,edge_color = 'r', alpha = 0.9)
    labels=dict(zip(graph.edges(),[d for u,v,d in graph.edges(data=True)]))
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=e_labels)
    plt.show()
    
def init_graph():
    global node_index, edge_index
    #graph = nx.generators.classic.grid_2d_graph(2, 2)
    #graph = graph.to_directed()
    graph = nx.DiGraph()
    #graph.add_edges_from([('A','B'),('B','C'),('C','A')])
    #graph.add_edges_from([('A','B'),('B','C'),('C','D'),('D','A')])
    graph.add_edges_from([('A','B'),('A','C'),('B','D'),('C','E'),('D','E'),('D','F'),('E','G')])
    #graph.add_edges_from([('1','2'),('2','4'),('1','3'),('3','5')])
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    
    graph.node['A']['load'] = -10
    graph.node['B']['load'] = 75
    graph.node['C']['load'] = 35
    graph.node['D']['load'] = 0
    graph.node['E']['load'] = 0
    graph.node['F']['load'] = -50
    graph.node['G']['load'] = -50
    
    graph.edge['A']['B']['capacity'] = 5
    graph.edge['A']['C']['capacity'] = 5
    graph.edge['B']['D']['capacity'] = 100
    graph.edge['C']['E']['capacity'] = 100
    graph.edge['D']['E']['capacity'] = 100
    graph.edge['D']['F']['capacity'] = 100
    graph.edge['E']['G']['capacity'] = 100

    #graph.node['1']['load'] = 30
    #graph.node['2']['load'] = 0
    #graph.node['3']['load'] = 0
    #graph.node['4']['load'] = -20
    #graph.node['5']['load'] = -10

    #graph.edge['1']['2']['capacity'] = 20
    #graph.edge['1']['3']['capacity'] = 10
    #graph.edge['2']['4']['capacity'] = 20
    #graph.edge['3']['5']['capacity'] = 10

    for u,v,d in edges:
        d['load'] = 0
    
    node_index = dict(zip(graph.nodes(), range(len(nodes))))
    edge_index = dict(zip(graph.edges(), range(len(edges))))

    #for (name,data) in nodes:
    #    data['load'] = 0
    #name,data = nodes[0]
    #print "supply:", name
    #data['load'] = 5.0
    #name,data = nodes[1]
    #print "sink:", name
    #data['load'] = -5.0
    #    
    #for u,v,data in edges:
    #    data['capacity'] = 2.0
    #    data['load'] = 0.0
    #u,v,data = edges[0]
    #data['load'] = 1.5
    return graph

def solve_network_flow(graph):
    global edge_index, node_index
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    print "edge index:", edge_index
    
    num_source_sink = 0
    for name,data in nodes:
        if data['load'] != 0:
            num_source_sink += 1
        
    con_matrix = np.zeros((len(nodes)+len(edges), len(edges)+num_source_sink+len(nodes)))
    for u,v,d in edges:
        j = edge_index[(u,v)]
        iu = node_index[u]
        iv = node_index[v]
        con_matrix[iu,j] = 1
        con_matrix[iv,j] = -1    
    
    # line capacities (upper and lower bounds)
    source_weight = 5
    sink_weight = -100
    #bus_weight = 0
    line_resistance = 1
    lp = lpsolve('make_lp', 0, len(edges)+num_source_sink+len(nodes))
    objfn = np.zeros(len(edges)+num_source_sink+len(nodes))
    upbounds = np.zeros(len(edges)+num_source_sink+len(nodes))
    lobounds = np.zeros(len(edges)+num_source_sink+len(nodes))
    j = len(edges)
    for name,data in nodes:
        i = node_index[name]
        load = data['load']
        if load > 0:
            con_matrix[i,j] = -1
            upbounds[j] = load
            lobounds[j] = 0
            objfn[j] = source_weight
            edge_index[(None,name)] = j
            j += 1
        elif load < 0:
            con_matrix[i,j] = 1
            upbounds[j] = -load
            lobounds[j] = 0
            objfn[j] = sink_weight
            edge_index[(name, None)] = j
            j += 1
    # set up power angle constraints
    pow_angle_offset = j
    i = len(nodes)
    for u,v,d in edges:
        ju = node_index[u] + pow_angle_offset
        jv = node_index[v] + pow_angle_offset
        je = edge_index[(u,v)]
        con_matrix[i,ju] = 1
        con_matrix[i,jv] = -1
        con_matrix[i,je] = -line_resistance
        i += 1
    # set bounds to be line capacities
    for u,v,d in edges:
        i = edge_index[(u,v)]
        cap = d['capacity']
        upbounds[i] = cap
        lobounds[i] = -cap
    for i in range(len(edges)+num_source_sink,len(upbounds)):
        upbounds[i] = Infinite
        lobounds[i] = -Infinite
      
    # node constraints
    for name,data in nodes:
        i = node_index[name]
        print name
        lpsolve('add_constraint', lp, con_matrix[i], EQ, 0)
        print "constraint:", con_matrix[i], "=", 0
    # power angle constraints
    for i in range(len(nodes), len(nodes)+len(edges)):
        lpsolve('add_constraint', lp, con_matrix[i], EQ, 0)
        print "constraint:", con_matrix[i], "=", 0
        
    print "objective func:", objfn
    lpsolve('set_verbose', lp, IMPORTANT)
    lpsolve('set_minim', lp)
    lpsolve('set_obj_fn', lp, objfn)
    lpsolve('set_upbo', lp, upbounds)
    lpsolve('set_lowbo', lp, lobounds)
    lpsolve('set_outputfile', lp, "")
    lpsolve('print_lp',lp)
    result = lpsolve('solve',lp)

    solution = lpsolve('get_variables', lp)[0]
    print "solution:", solution
    for u,v,d in edges:
        i = edge_index[(u,v)]
        d['load'] = solution[i]
        #if len(edges) > 1:
        #    d['load'] = solution[i]
        #else:
        #    d['load'] = solution
    #lpsolve('delete_lp',lp)
    return lp


# checks: theta_i - theta_j = x_ij * p_ij
# for all power angles theta, resistance x, and power p
# analogous to V=RI
def test_feasibility(graph):
    global node_index, edge_index
    edges = graph.edges(data=True)
    a = np.zeros((len(graph.edge), len(graph.node)))
    b = np.zeros(len(graph.edge))
    for u,v,d in edges:
        i = edge_index[(u,v)]
        ju = node_index[u]
        jv =  node_index[v]
        a[i,ju] = 1
        a[i,jv] = -1
        b[i] = d['load']
    
    x = np.linalg.lstsq(a,b)
    feasible = True
    dot_prod = np.dot(a,x[0])
    for u,v,d in edges:
        i = edge_index[(u,v)]
        if round(dot_prod[i],3) != round(b[i],3):
            feasible = False
        print u, '==>', v, dot_prod[i], b[i] 
    return feasible
if __name__ == "__main__":
    graph = init_graph()
    lp = solve_network_flow(graph)
    feasible = test_feasibility(graph)
    if feasible:
        print "Solution Feasible"
    else:
        print "Solution Infeasible"
    display_graph(graph)
