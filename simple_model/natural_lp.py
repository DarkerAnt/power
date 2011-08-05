#!/usr/bin/python

from lpsolve55 import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


source_cost = 29.5 # cost of coal / MW
sink_cost = -100
#line_weight = 0
cap_increase_cost = 90
cap_increase = 5
line_resistance = 1 # i bet i can fix this by throwing it into a capacity form instead


#source_sink_nodes = None

class LPVars:
    def __init__(self):
        self.vars = None
        self.objfn = None
        self.upbounds = None
        self.lobounds = None

class LPCont:
    def __init__(self):
        self.lp = None
        self.con_matrix = None
        self.objfn = None
        self.upbounds = None
        self.lobounds = None
        self.line_power = LPVars()
        self.source_sink = LPVars()
        self.pos_cap_increase = LPVars()
        self.neg_cap_increase = LPVars()
        self.pow_angle = LPVars()
        self.source_nodes = []
        self.sink_nodes = []
        self.node_index = {}
        self.edge_index = {}

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
    labels = [str(abs(round(d['load'],3))) + '/' + str(round(d['capacity'],3)) for u,v,d in edges]
    e_labels=dict(zip(graph.edges(), labels))
    pos = nx.spring_layout(graph, iterations=100)
    nx.draw_networkx_nodes(graph, pos, node_size=300)
    nx.draw_networkx_labels(graph,pos)
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=edgecapwidth,edge_color = 'b')
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=edgeloadwidth,edge_color = 'r', alpha = 0.9)
    labels=dict(zip(graph.edges(),[d for u,v,d in graph.edges(data=True)]))
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=e_labels)
    plt.show()
    
def init_graph():
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
    
    #node_index = dict(zip(graph.nodes(), range(len(nodes))))
    #edge_index = dict(zip(graph.edges(), range(len(edges))))

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



def build_lp(graph):
    global source_cost, sink_cost, cap_increase_cost, cap_increase, line_resistance
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    
    lpc = LPCont()
    lpc.node_index = dict(zip(graph.nodes(), range(len(nodes))))
    lpc.edge_index = dict(zip(graph.edges(), range(len(edges))))

    for name,data in nodes:
        load = data['load']
        if load > 0:
            lpc.source_nodes.append(name)
        elif load < 0:
            lpc.sink_nodes.append(name)

    # con_matrix format: con_matrix[node, var]
    #                    len(edges)  + len(source_sink_nodes) + len(edges)          +       len(edges)        +  len(nodes)
    # |             | power line use | source/sink use | cap increase(pos going) | cap increase(neg going) | power angles |
    # | node1       |        1               1                   i                           i                    0          = 0
    # | node2       |        1               1                   i                           i                    0          = 0
    # | node3       |        1               1                   i                           i                    0          = 0
    # | node_angle1 |        1               1                   i                           i                   -r          = 0
    # | node_angle2 |        1               1                   i                           i                   -r          = 0
    # | node_angle3 |        1               1                   i                           i                   -r          = 0
    
    cap_increase_start = len(edges) + len(lpc.source_nodes) + len(lpc.sink_nodes)
    var_len = len(edges) + len(lpc.source_nodes) + len(lpc.sink_nodes) + len(edges) + len(edges) + len(nodes)

    lp = lpsolve('make_lp', 0, var_len)
    con_matrix = np.zeros((len(nodes)+len(edges), var_len))
    objfn = np.zeros(var_len)
    upbounds = np.zeros(var_len)
    lobounds = np.zeros(var_len)
    lpc.lp = lp
    lpc.con_matrix = con_matrix
    lpc.objfn = objfn
    lpc.upbounds = upbounds
    lpc.lobounds = lobounds
    
    start = 0
    stop = len(edges) 
    lpc.line_power.vars = con_matrix[:,start:stop]
    lpc.line_power.objfn = objfn[start:stop]
    lpc.line_power.upbounds = upbounds[start:stop]
    lpc.line_power.lobounds = lobounds[start:stop]

    start = stop
    stop = start + len(lpc.source_nodes) + len(lpc.sink_nodes)
    lpc.source_sink.vars = con_matrix[:,start:stop]
    lpc.source_sink.objfn = objfn[start:stop]
    lpc.source_sink.upbounds = upbounds[start:stop]
    lpc.source_sink.lobounds = lobounds[start:stop]

    start = stop
    stop = start + len(edges)
    lpc.pos_cap_increase.vars = con_matrix[:,start:stop]
    lpc.pos_cap_increase.objfn = objfn[start:stop]
    lpc.pos_cap_increase.upbounds = upbounds[start:stop]
    lpc.pos_cap_increase.lobounds = lobounds[start:stop]
    
    start = stop
    stop = start + len(edges)
    lpc.neg_cap_increase.vars = con_matrix[:,start:stop]
    lpc.neg_cap_increase.objfn = objfn[start:stop]
    lpc.neg_cap_increase.upbounds = upbounds[start:stop]
    lpc.neg_cap_increase.lobounds = lobounds[start:stop]

    start = stop
    stop = start + len(nodes)
    lpc.pow_angle.vars = con_matrix[:,start:stop]
    lpc.pow_angle.objfn = objfn[start:stop]
    lpc.pow_angle.upbounds = upbounds[start:stop]
    lpc.pow_angle.lobounds = lobounds[start:stop]
    
    # line_power and cap_increase
    for u,v,d in edges:
        cap = d['capacity']
        j = lpc.edge_index[(u,v)]
        iu = lpc.node_index[u]
        iv = lpc.node_index[v]
        lpc.line_power.vars[iu,j] = 1
        lpc.line_power.vars[iv,j] = -1
        lpc.line_power.upbounds[j] = cap
        lpc.line_power.lobounds[j] = -cap
        
    
        lpc.pos_cap_increase.vars[iu,j] = cap_increase
        lpc.pos_cap_increase.vars[iv,j] = -cap_increase
        lpc.pos_cap_increase.objfn[j] = cap_increase_cost
        lpc.pos_cap_increase.upbounds[j] = Infinite
        lpc.pos_cap_increase.lobounds[j] = 0

        lpc.neg_cap_increase.vars[iu,j] = -cap_increase
        lpc.neg_cap_increase.vars[iv,j] = cap_increase
        lpc.neg_cap_increase.objfn[j] = cap_increase_cost
        lpc.neg_cap_increase.upbounds[j] = Infinite
        lpc.neg_cap_increase.lobounds[j] = 0
    # set cap increase vars to type int   
        cap_increase_stop = cap_increase_start + len(lpc.pos_cap_increase.objfn) + len(lpc.neg_cap_increase.objfn)
        for i in range(cap_increase_start+1, cap_increase_stop+1): # + 1 because vars are stored starting at 1
            lpsolve('set_int', lp, i, True)

    # source/sink
    j = 0
    for name in lpc.source_nodes:
        i = lpc.node_index[name]
        load = graph.node[name]['load']
        lpc.source_sink.vars[i,j] = -1
        lpc.source_sink.objfn[j] = source_cost
        lpc.source_sink.upbounds[j] = load
        lpc.source_sink.lobounds[j] = 0
        j += 1
    for name in lpc.sink_nodes:
        i = lpc.node_index[name]
        load = graph.node[name]['load']
        lpc.source_sink.vars[i,j] = 1
        lpc.source_sink.objfn[j] = sink_cost
        lpc.source_sink.upbounds[j] = -load
        lpc.source_sink.lobounds[j] = 0
        j += 1

    #for j in range(len(lpc.source_nodes) + len(lpc.sink_nodes)):
    #    name = source_sink_nodes[j]
    #    i = lpc.node_index[name]
    #    load = graph.node[name]['load']
    #    if load > 0: # source
    #        lpc.source_sink.vars[i,j] = -1
    #        lpc.source_sink.objfn[j] = source_cost
    #        lpc.source_sink.upbounds[j] = load
    #        lpc.source_sink.lobounds[j] = 0
    #        lpc.source_nodes.append(name)
    #    else: # sink
    #        lpc.source_sink.vars[i,j] = 1
    #        lpc.source_sink.objfn[j] = sink_cost
    #        lpc.source_sink.upbounds[j] = -load
    #        lpc.source_sink.lobounds[j] = 0
    #        lpc.sink_nodes.append(name)

    # power angles
    i = len(nodes)
    for u,v,d in edges:
        ju = lpc.node_index[u]
        jv = lpc.node_index[v]
        je = lpc.edge_index[(u,v)]
        
        lpc.pow_angle.vars[i,ju] = 1
        lpc.pow_angle.vars[i,jv] = -1
        lpc.line_power.vars[i,je] = -line_resistance
        lpc.pos_cap_increase.vars[i,je] = -line_resistance * cap_increase
        lpc.neg_cap_increase.vars[i,je] = line_resistance * cap_increase
        i += 1
    for i in range(len(lpc.pow_angle.upbounds)):
        lpc.pow_angle.upbounds[i] = Infinite
        lpc.pow_angle.lobounds[i] = -Infinite
      
    # node constraints
    for name,data in nodes:
        i = lpc.node_index[name]
        
        lpsolve('add_constraint', lp, con_matrix[i], EQ, 0)
        #print name, "constraint:", con_matrix[i], "=", 0
        lpsolve('set_row_name', lp, i, str(name))
    # power angle constraints
    for i in range(len(nodes), len(nodes)+len(edges)):
        lpsolve('add_constraint', lp, con_matrix[i], EQ, 0)
        #print "constraint:", con_matrix[i], "=", 0
        
    #print "objective func:", objfn
    lpsolve('set_verbose', lp, IMPORTANT)
    lpsolve('set_minim', lp)
    lpsolve('set_obj_fn', lp, objfn)
    lpsolve('set_upbo', lp, upbounds)
    lpsolve('set_lowbo', lp, lobounds)

    return lpc

def solve_network_flow(graph, lpc = None):
    global cap_increase
    

    if lpc == None:
        lpc = build_lp(graph)
    lp = lpc.lp
    cap_increase_start = len(lpc.edge_index) + len(lpc.source_nodes) + len(lpc.sink_nodes)
    lpsolve('set_outputfile', lp, "lp.txt")
    lpsolve('print_lp',lp)
    result = lpsolve('solve',lp)

    solution = lpsolve('get_variables', lp)[0]
    #print "solution:", solution
    
    cap_start = len(lpc.edge_index)  + len(lpc.source_nodes) + len(lpc.sink_nodes) + 1
    if lpsolve('get_upbo', lp, cap_start) == Infinite:
        build_capacity = True
    else:
        build_capacity = False

    if build_capacity:
        line_cap_increase = solution[cap_increase_start:cap_increase_start+len(lpc.edge_index)]
        line_cap_increase += solution[cap_increase_start+len(lpc.edge_index):cap_increase_start+2*len(lpc.edge_index)]
        for i in range(len(line_cap_increase)):
            line_cap_increase[i] *= cap_increase
    else:
        line_cap_increase = np.zeros(len(solution))
    
    for u,v,d in graph.edges(data=True):
        i = lpc.edge_index[(u,v)]
        d['load'] = solution[i]
        if build_capacity:
            d['load'] += solution[cap_increase_start + i] * cap_increase  # pos going additional capacity
            d['load'] -= solution[cap_increase_start+len(lpc.edge_index) + i] * cap_increase # neg going add cap
        #if len(edges) > 1:
        #    d['load'] = solution[i]
        #else:
        #    d['load'] = solution
    #lpsolve('delete_lp',lp)

    # get net power at all nodes
    node_power = np.zeros(len(lpc.node_index))
    j = 0
    for name in lpc.source_nodes:
        i = lpc.node_index[name]
        node_power[i] = solution[len(lpc.edge_index) + j]
        j += 1
    for name in lpc.sink_nodes:
        i = lpc.node_index[name]
        node_power[i] = solution[len(lpc.edge_index) + j]
        j += 1

    #node_power = np.zeros(len(lpc.node_index))
    #for j in range(len(source_sink_nodes)):
    #    name = source_sink_nodes[j]
    #    i = lpc.node_index[name]
    #    power = solution[len(lpc.edge_index) + j]
    #    node_power[i] = power

    return (node_power, line_cap_increase)

def toggle_capacity_expansion(lpc, enable):
    if enable:
        upbo = Infinite
    else:
        upbo = 0

    for i in range(len(lpc.pos_cap_increase.upbounds)):
        lpc.pos_cap_increase.upbounds[i] = upbo
        lpc.neg_cap_increase.upbounds[i] = upbo

    lpsolve('set_upbo', lpc.lp, lpc.upbounds)

        
# checks: theta_i - theta_j = x_ij * p_ij
# for all power angles theta, resistance x, and power p
# analogous to V=RI
def test_feasibility(graph, lpc):
    edges = graph.edges(data=True)
    a = np.zeros((len(edges), len(graph.node)))
    b = np.zeros(len(edges))
    for u,v,d in edges:
        i = lpc.edge_index[(u,v)]
        ju = lpc.node_index[u]
        jv =  lpc.node_index[v]
        a[i,ju] = 1
        a[i,jv] = -1
        b[i] = d['load']
    
    x = np.linalg.lstsq(a,b)
    feasible = True
    dot_prod = np.dot(a,x[0])
    for u,v,d in edges:
        i = lpc.edge_index[(u,v)]
        if round(dot_prod[i],3) != round(b[i],3):
            feasible = False
        #print u, '==>', v, dot_prod[i], b[i] 
    return feasible

def get_delta_power(graph, net_node_power):
    nodes = graph.nodes(data=True)
    delta_power = np.zeros(len(nodes))
    total_load_shed = 0
    total_excess_power = 0
    for i in range(len(nodes)):
        name,data = nodes[i]
        load = data['load']
        if load < 0:
            delta_power[i] = load + net_node_power[i]
            total_load_shed += delta_power[i]
        else:
            delta_power[i] = load - net_node_power[i]
            total_excess_power += delta_power[i]
    return (delta_power, total_load_shed, total_excess_power)

def increase_load(graph, num_nodes, total_increase, lpc=None):
    load_increase = total_increase / num_nodes
    selected_nodes = random.sample(graph.node, num_nodes)
    for name in selected_nodes:
        graph.node[name]['load'] -= load_increase
        if lpc != None:
            i = node_index[name]
            lpc.source_sink.upbounds[i] -= load_increase

def find_optimal_weight(graph):
    global source_cost, sink_cost, cap_increase_cost, cap_increase, line_resistance
    
    x = []
    y = []
    lpc = build_lp(graph)
    for i in range(1, -sink_cost-1):
        lp2 = lpsolve('copy_lp', lpc.lp)
        for name in source_sink_nodes:
            if graph.node[name]['load'] > 0:
                lpsolve('set_obj', lp2, len(lpc.edge_index) + lpc.node_index[name], i)
        lpsolve('solve', lp2)
        y.append(lpsolve('get_total_iter', lp2))
        lpsolve('delete_lp',lp2)
        x.append(i)
    plt.subplot(1,2,1)
    plt.plot(x,y, label="source cost")
    plt.legend()
    x = []
    y = []
    for i in range(-500,-(source_cost+1)):
        lp2 = lpsolve('copy_lp', lpc.lp)
        for name in source_sink_nodes:
            if graph.node[name]['load'] < 0:
                lpsolve('set_obj', lp2, len(edge_index) + node_index[name], i)
        lpsolve('solve', lp2)
        y.append(lpsolve('get_total_iter', lp2))
        lpsolve('delete_lp',lp2)
        x.append(i)
    
    plt.subplot(1,2,2)
    plt.plot(x,y, label="sink cost")
    plt.legend()

    plt.show()
        
def speed_test():
    for run in range(100):
        for i in range(10,211,10):
            graph = nx.random_graphs.powerlaw_cluster_graph(i, 3, 0.1)
            nodes = graph.nodes(data=True)
            for name,data in nodes:
                data['load'] = random.uniform(0,-20)
            gens = random.sample(nodes, 5)
            for name,data in gens:
                data['load'] = 350#random.uniform(100, 800)
            for u,v,d in graph.edges(data=True):
                d['capacity'] = random.uniform(5, 100)
                d['load'] = 0
            lpc = build_lp(graph)
            solve_network_flow(graph,lpc)
            print i, lpsolve('time_elapsed', lpc.lp)
            lpsolve('delete_lp', lpc.lp)
        

def test_load_volatility():
    n = 100
    num_increase = 5
    total_increase = 40
    graph = nx.random_graphs.powerlaw_cluster_graph(n, 3, 0.1)
    lpc = build_lp(graph)
    for i in range(100):
        node_power,cap_increase = solve_network_flow(graph,lp)
        increase_load(graph, num_increase, total_increase,lpc)

if __name__ == "__main__":
    #speed_test()
    graph = init_graph()
    
    #find_optimal_weight(graph)
    lpc = build_lp(graph)
    #toggle_capacity_expansion(lpc, enable=False)
    node_power,cap_increase = solve_network_flow(graph, lpc)
    print "cap increase:", cap_increase
    print "net node power:", node_power
    feasible = test_feasibility(graph,lpc)
    if feasible:
        print "Solution Feasible"
    else:
        print "Solution Infeasible"
    delta_power,load_shed,excess_power = get_delta_power(graph, node_power)
    print "Delta Power:", delta_power
    print "Load Shed:", load_shed
    print "Excess Power:", excess_power
    display_graph(graph)
