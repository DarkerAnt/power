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
    labels = [str(abs(round(d['load'],3))) + '/' + str(d['capacity']) for u,v,d in edges]
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
    #graph.add_edges_from([('1','3'),('1','4'),('2','3'),('2','4')])
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

    #graph.node['1']['load'] = 100
    #graph.node['2']['load'] = 50
    #graph.node['3']['load'] = -50
    #graph.node['4']['load'] = -50

    #graph.edge['1']['3']['capacity'] = 50
    #graph.edge['1']['4']['capacity'] = 30
    #graph.edge['2']['3']['capacity'] = 25
    #graph.edge['2']['4']['capacity'] = 10

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
    source_cost = 5
    sink_cost = -100
    #line_weight = 0
    cap_increase_cost = 90
    cap_increase = 5
    line_resistance = 1 # i bet i can fix this by throwing it into a capacity form instead

    global edge_index, node_index
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    print "edge index:", edge_index
    
    num_source_sink = 0
    for name,data in nodes:
        if data['load'] != 0:
            num_source_sink += 1

    # con_matrix format: con_matrix[node, var]
    #                    len(edges)  + num_source_sink +     len(edges)          +       len(edges)        +  len(nodes)
    # |             | power line use | source/sink use | cap increase(pos going) | cap increase(neg going) | power angles |
    # | node1       |        1               1                   i                           i                    0          = 0
    # | node2       |        1               1                   i                           i                    0          = 0
    # | node3       |        1               1                   i                           i                    0          = 0
    # | node_angle1 |        1               1                   i                           i                   -r          = 0
    # | node_angle2 |        1               1                   i                           i                   -r          = 0
    # | node_angle3 |        1               1                   i                           i                   -r          = 0
    cap_increase_start = len(edges) + num_source_sink
    var_len = len(edges) + num_source_sink + len(edges) + len(edges) + len(nodes)
    lp = lpsolve('make_lp', 0, var_len)
    con_matrix = np.zeros((len(nodes)+len(edges), var_len))
    objfn = np.zeros(var_len)
    upbounds = np.zeros(var_len)
    lobounds = np.zeros(var_len)
    
    start = 0
    stop = len(edges) 
    line_power_vars = con_matrix[:,start:stop]
    line_power_objfn = objfn[start:stop]
    line_power_upbounds = upbounds[start:stop]
    line_power_lobounds = lobounds[start:stop]

    start = stop
    stop = start + num_source_sink
    source_sink_vars = con_matrix[:,start:stop]
    source_sink_objfn = objfn[start:stop]
    source_sink_upbounds = upbounds[start:stop]
    source_sink_lobounds = lobounds[start:stop]

    start = stop
    stop = start + len(edges)
    cap_increase_pos_vars = con_matrix[:,start:stop]
    cap_increase_pos_objfn = objfn[start:stop]
    cap_increase_pos_upbounds = upbounds[start:stop]
    cap_increase_pos_lobounds = lobounds[start:stop]

    start = stop
    stop = start + len(edges)
    cap_increase_neg_vars = con_matrix[:,start:stop]
    cap_increase_neg_objfn = objfn[start:stop]
    cap_increase_neg_upbounds = upbounds[start:stop]
    cap_increase_neg_lobounds = lobounds[start:stop]

    start = stop
    stop = start + len(nodes)
    pow_angle_vars = con_matrix[:,start:stop]
    pow_angle_objfn = objfn[start:stop]
    pow_angle_upbounds = upbounds[start:stop]
    pow_angle_lobounds = lobounds[start:stop]
    
    # line_power and cap_increase
    for u,v,d in edges:
        cap = d['capacity']
        j = edge_index[(u,v)]
        iu = node_index[u]
        iv = node_index[v]
        line_power_vars[iu,j] = 1
        line_power_vars[iv,j] = -1
        line_power_upbounds[j] = cap
        line_power_lobounds[j] = -cap
        
        cap_increase_pos_vars[iu,j] = cap_increase
        cap_increase_pos_vars[iv,j] = -cap_increase
        cap_increase_pos_objfn[j] = cap_increase_cost
        cap_increase_pos_upbounds[j] = Infinite
        cap_increase_pos_lobounds[j] = 0

        cap_increase_neg_vars[iu,j] = -cap_increase
        cap_increase_neg_vars[iv,j] = cap_increase
        cap_increase_neg_objfn[j] = cap_increase_cost
        cap_increase_neg_upbounds[j] = Infinite
        cap_increase_neg_lobounds[j] = 0
    # set cap increase vars to type int
    cap_increase_stop = cap_increase_start + len(cap_increase_pos_objfn) + len(cap_increase_neg_objfn)
    for i in range(cap_increase_start+1, cap_increase_stop+1): # + 1 because vars are stored starting at 1
        lpsolve('set_int', lp, i, True)

    # source/sink (i should really just hold the list of nodes when i initially searched for this stuff)
    j = 0
    for name,data in nodes:
        i = node_index[name]
        load = data['load']
        if load > 0:
            source_sink_vars[i,j] = -1
            source_sink_objfn[j] = source_cost
            source_sink_upbounds[j] = load
            source_sink_lobounds[j] = 0
            j += 1
        elif load < 0:
            source_sink_vars[i,j] = 1
            source_sink_objfn[j] = sink_cost
            source_sink_upbounds[j] = -load
            source_sink_lobounds[j] = 0
            j += 1

    # power angles
    i = len(nodes)
    print pow_angle_vars.shape
    print "edge len:", len(edges)
    print edge_index
    print node_index
    for u,v,d in edges:
        ju = node_index[u]
        jv = node_index[v]
        je = edge_index[(u,v)]
        print ju,jv,je
        pow_angle_vars[i,ju] = 1
        pow_angle_vars[i,jv] = -1
        line_power_vars[i,je] = -line_resistance
        cap_increase_pos_vars[i,je] = -line_resistance
        cap_increase_neg_vars[i,je] = line_resistance
        i += 1
    for i in range(len(pow_angle_upbounds)):
        pow_angle_upbounds[i] = Infinite
        pow_angle_lobounds[i] = -Infinite
      
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
    lpsolve('set_outputfile', lp, "lp.txt")
    lpsolve('print_lp',lp)
    result = lpsolve('solve',lp)

    solution = lpsolve('get_variables', lp)[0]
    print "solution:", solution

    line_cap_increase = solution[cap_increase_start:len(cap_increase_pos_objfn)]
    line_cap_increase += solution[cap_increase_start+len(cap_increase_pos_objfn):cap_increase_stop]
    for i in range(len(line_cap_increase)):
        line_cap_increase[i] *= cap_increase

    for u,v,d in edges:
        i = edge_index[(u,v)]
        d['load'] = solution[i]
        d['load'] += solution[cap_increase_start + i] # pos going additional capacity
        d['load'] -= solution[cap_increase_start+len(cap_increase_pos_objfn) + i] # neg going add cap
        #if len(edges) > 1:
        #    d['load'] = solution[i]
        #else:
        #    d['load'] = solution
    #lpsolve('delete_lp',lp)
    return (lp, line_cap_increase)

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
    lp,cap_increase = solve_network_flow(graph)
    print "cap increase:", cap_increase
    feasible = test_feasibility(graph)
    if feasible:
        print "Solution Feasible"
    else:
        print "Solution Infeasible"
    display_graph(graph)
