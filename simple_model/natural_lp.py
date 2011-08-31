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
        self.start = 0
        self.end = 0
        self.vars = None
        self.objfn = None
        self.upbounds = None
        self.lobounds = None
    def set(self, start, stop, con_matrix, objfn, upbounds, lobounds):
        self.start = start
        self.end = stop
        self.vars = con_matrix[:,start:stop]
        self.objfn = objfn[start:stop]
        self.upbounds = upbounds[start:stop]
        self.lobounds = lobounds[start:stop]

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

    def update_edge_power(self, graph):
        global cap_increase
        solution = lpsolve('get_variables', self.lp)[0]
        start = self.line_power.start
        stop = self.line_power.end
        line_pow = solution[start:stop]
        start = self.pos_cap_increase.start
        stop = self.pos_cap_increase.end
        pos_cap = solution[start:stop]
        start = self.neg_cap_increase.start
        stop = self.neg_cap_increase.end
        neg_cap = solution[start:stop]
        
        edges = graph.edges(data=True)
        for u,v,d in edges:
            i = self.edge_index[(u,v)]
            d['load'] = line_pow[i] + (pos_cap[i] - neg_cap[i]) * cap_increase

    def transfer_cap_increase(self, graph):
        global cap_increase
        solution = lpsolve('get_variables', self.lp)[0]
        start = self.pos_cap_increase.start
        stop = self.pos_cap_increase.end
        pos_cap = solution[start:stop]
        start = self.neg_cap_increase.start
        stop = self.neg_cap_increase.end
        neg_cap = solution[start:stop]

        edges = graph.edges(data=True)
        for u,v,d in edges:
            i = self.edge_index[(u,v)]
            cap = d['capacity'] + (pos_cap[i] + neg_cap[i]) * cap_increase
            d['capacity'] = cap
            self.line_power.upbounds[i] = cap
            self.line_power.lobounds[i] = -cap
            # this doesn't actually work, maybe use guess basis?
            pos_cap[i] = 0
            neg_cap[i] = 0
        lpsolve('set_upbo', self.lp, self.upbounds)
        lpsolve('set_lowbo', self.lp, self.lobounds)
        
    def update_node_power(self, graph):
        node_power = self.get_net_node_power()
        for name in graph.node:
            i = self.node_index[name]
            graph.node[name]['power'] = node_power[i]

        node_throughput = self.get_node_throughput(graph)
        print "update node throughput:", node_throughput
        for name in self.source_nodes:
            i = self.node_index[name]
            graph.node[name]['throughput'] = node_throughput[i]
        for name in self.sink_nodes:
            i = self.node_index[name]
            graph.node[name]['throughput'] = node_throughput[i]

    def get_line_cap_increase(self):
        global cap_increase
        if self.lp == None:
            print "ERROR: LP not initialized"
            return -1
        solution = lpsolve('get_variables', self.lp)[0]
        start = self.pos_cap_increase.start
        stop = self.pos_cap_increase.end
        line_cap_increase = np.copy(solution[start:stop])
        start = self.neg_cap_increase.start
        stop = self.neg_cap_increase.end
        line_cap_increase += solution[start:stop]
        return line_cap_increase*cap_increase
    
    def get_line_power(self):
        if self.lp == None:
            print "ERROR: LP not initialized"
            return -1
        solution = lpsolve('get_variables', self.lp)[0]
        start = self.line_power.start
        stop = self.line_power.end
        return np.copy(solution[start:stop]) + self.get_line_cap_increase()
    
    # uses injecting and consuming edges to find node power
    # used to get final power unit placement
    def get_net_node_power(self):
        if self.lp == None:
            print "ERROR: LP not initialized"
            return -1
        solution = lpsolve('get_variables', self.lp)[0]
        node_power = np.zeros(len(self.node_index))
        start = self.source_sink.start
        stop = self.source_sink.end
        net_power = solution[start:stop]
        j = 0
        for name in self.source_nodes:
            i = self.node_index[name]
            node_power[i] = net_power[j]
            j += 1
        for name in self.sink_nodes:
            i = self.node_index[name]
            node_power[i] = net_power[j]
            j += 1
        return node_power
    
    # uses edge direction to determine power through nodes
    # used to find power routed through nodes
    def get_node_throughput(self, graph):
        global cap_increase
        solution = lpsolve('get_variables', self.lp)[0]
        start = self.line_power.start
        stop = self.line_power.end
        line_pow = solution[start:stop]
        start = self.pos_cap_increase.start
        stop = self.pos_cap_increase.end
        pos_cap_pow = solution[start:stop]
        start = self.neg_cap_increase.start
        stop = self.neg_cap_increase.end
        neg_cap_pow = solution[start:stop]
        start = self.source_sink.start
        stop = self.source_sink.end
        gen_power = solution[start:stop]
                
        edges = graph.edges()
        node_power = np.zeros(len(edges))
        for u,v in edges:
            iu = self.node_index[u]
            iv = self.node_index[v]
            j = self.edge_index[(u,v)]
            power = line_pow[j] + cap_increase * (pos_cap_pow[j] - neg_cap_pow[j])
            if power >= 0:
                node_power[iv] += power
            else:
                node_power[iu] -= power

        j = 0
        for name in self.source_nodes:
            i = self.node_index[name]
            node_power[i] += gen_power[j]
            j += 1
        return node_power

    def update(self, graph):
        self.update_edge_power(graph)
        self.transfer_cap_increase(graph)
        self.update_node_power(graph)

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
    graph.node['C']['load'] = 40#35
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

    for u,d in nodes:
        d['power'] = 0
        d['throughput'] = 0
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
        else:
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
    lpc.line_power.set(start, stop, con_matrix, objfn, upbounds, lobounds)

    start = stop
    stop = start + len(lpc.source_nodes) + len(lpc.sink_nodes)
    lpc.source_sink.set(start, stop, con_matrix, objfn, upbounds, lobounds)
    
    start = stop
    stop = start + len(edges)
    lpc.pos_cap_increase.set(start, stop, con_matrix, objfn, upbounds, lobounds)
        
    start = stop
    stop = start + len(edges)
    lpc.neg_cap_increase.set(start, stop, con_matrix, objfn, upbounds, lobounds)
    
    start = stop
    stop = start + len(nodes)
    lpc.pow_angle.set(start, stop, con_matrix, objfn, upbounds, lobounds)
    
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

    # this j nonsense scares me
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
    line_cap_increase = lpc.get_line_cap_increase()
    node_power = lpc.get_net_node_power()
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
    print a,b
    x = np.linalg.lstsq(a,b)
    feasible = True
    dot_prod = np.dot(a,x[0])
    for u,v,d in edges:
        i = lpc.edge_index[(u,v)]
        if round(dot_prod[i],3) != round(b[i],3):
            feasible = False
        #print u, '==>', v, dot_prod[i], b[i] 
    return feasible

#def get_delta_power(graph, node_power):
#    nodes = graph.nodes(data=True)
#    delta_power = np.zeros(len(nodes))
#    total_load_shed = 0
#    total_excess_power = 0
#    for i in range(len(nodes)):
#        name,data = nodes[i]
#        load = data['power']
#        print load
#        if load < 0:
#            delta_power[i] = load + node_power[i]
#            total_load_shed += delta_power[i]
#        else:
#            delta_power[i] = load - node_power[i]
#            total_excess_power += delta_power[i]
#    return (delta_power, total_load_shed, total_excess_power)

def get_delta_power(graph, lpc):
    node_power = lpc.get_net_node_power()
    print node_power
    delta_power = np.zeros(len(graph.node))
    total_load_shed = 0
    total_excess_power = 0

    for name in graph.node:
        i = lpc.node_index[name]
        delta_power[i] = node_power[i] - graph.node[name]['power']
        load = graph.node[name]['load']
        if load <= 0:
            total_load_shed -= load + node_power[i]
        else:
            print "excess:", name, load, '-', node_power[i]
            total_excess_power += load - node_power[i]
        
    return (delta_power, total_load_shed, total_excess_power)

def get_delta_throughput(graph, lpc):
    throughput = lpc.get_node_throughput(graph)
    delta_throughput = np.zeros(len(graph.node))
    for name in graph.node:
        i = lpc.node_index[name]
        delta_throughput[i] = throughput[i] - graph.node[name]['throughput']
    return delta_throughput

def sync_graph_to_lpc(graph, lpc):
    line_cap_increase = lpc.get_line_cap_increase()
    for u,v,d in graph.edges(data=True):
        i = lpc.edge_index[(u,v)]
        d['capacity'] += cap_increase[i]

def increase_gens(num_nodes, total_increase, graph, lpc):
    gen_increase = total_increase / num_nodes
    selected_nodes = random.sample(lpc.source_nodes, num_nodes)
    for name in selected_nodes:
        graph.node[name]['load'] += gen_increase
        i = lpc.node_index[name]
        lpc.source_sink.upbounds[i] += gen_increase
    lpsolve('set_upbo', lpc.lp, lpc.upbounds)

def increase_load(num_nodes, total_increase, graph, lpc):
    load_increase = total_increase / num_nodes
    selected_nodes = random.sample(lpc.sink_nodes, num_nodes)
    for name in selected_nodes:
        graph.node[name]['load'] -= load_increase
        i = lpc.node_index[name]
        lpc.source_sink.upbounds[i] += load_increase
    lpsolve('set_upbo', lpc.lp, lpc.upbounds)
        

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
        

def remove_edges(names, lpc):
    for name in names:
        i = lpc.edge_index[name]
        lpc.line_power.upbounds[i] = 0
        lpc.line_power.lobounds[i] = 0
    lpsolve('set_upbo', lpc.lp, lpc.upbounds)
    lpsolve('set_lowbo', lpc.lp, lpc.lobounds)

def restore_edges(names, graph, lpc):
    for name in names:
        i = lpc.edge_index[name]
        u,v = name
        cap = graph.edge[u][v]['capacity']
        lpc.line_power.upbounds[i] = cap
        lpc.line_power.lobounds[i] = -cap
    lpsolve('set_upbo', lpc.lp, lpc.upbounds)
    lpsolve('set_lowbo', lpc.lp, lpc.lobounds)
    
def test_load_volatility():
    fout = open("load_volatility_run.txt", 'w')
    num_nodes = 80
    num_increase = 5
    #total_increase = 5
    orig_graph = nx.random_graphs.powerlaw_cluster_graph(num_nodes, 3, 0.1)
    nodes = orig_graph.nodes()
    for name in nodes:
        orig_graph.node[name]['load'] = -10
    gens = random.sample(nodes, 5)
    for name in gens:
        orig_graph.node[name]['load'] = 200
    edges = orig_graph.edges(data=True)
    for u,v,d in edges:
        d['capacity'] = 20
        d['load'] = 0
    #display_graph(graph)
    for total_increase in range(5,35,5):
        print "Starting run:", total_increase
        fout.write(str(total_increase))
        graph = orig_graph.copy()
        lpc = build_lp(graph)
    # initially solve an easy LP and then tighten the bounds via increased demand
        for i in range(20):
            node_power,cap_increase = solve_network_flow(graph,lpc)
            print "time elapsed:", lpsolve('time_elapsed', lpc.lp), "cap increase:", np.sum(cap_increase)
            #fout.write(", "+ str(np.sum(cap_increase)))
            delta_power,load_shed,excess_power = get_delta_power(graph, node_power)
            fout.write(", " + str(sum(delta_power)))
            lpc.update(graph)
            increase_load(num_increase, total_increase, graph, lpc)

    # start expanding the gens
        for i in range(100):
            node_power,cap_increase = solve_network_flow(graph,lpc)
            print "time elapsed:", lpsolve('time_elapsed', lpc.lp), "cap increase:", np.sum(cap_increase)
            #fout.write(", " + str(np.sum(cap_increase)))
            delta_power,load_shed,excess_power = get_delta_power(graph, node_power)
            fout.write(", " + str(sum(delta_power)))
            lpc.update(graph)
            increase_load(num_increase, total_increase, graph, lpc)
            for name in gens:
                graph.node[name]['load'] += total_increase/len(gens)
        fout.write('\n')

def normal_test():
    graph = init_graph()
    lpc = build_lp(graph)
    #toggle_capacity_expansion(lpc, enable=False)
    node_power,cap_increase = solve_network_flow(graph, lpc)
    #print "cap increase:", cap_increase
    print "node power:", node_power
    delta_power,load_shed,excess_power = get_delta_power(graph,lpc)
    print "delta power:", delta_power
    lpc.update(graph)
    feasible = test_feasibility(graph,lpc)
    if feasible:
        print "Solution Feasible"
    else:
        print "Solution Infeasible"
    
    #print "Delta Power:", delta_power
    #print "Load Shed:", load_shed
    #print "Excess Power:", excess_power
    print lpc.node_index
    display_graph(graph)
    return graph,lpc

def extended_normal_test():
    num_increase = 1
    total_increase = 7.5
    graph,lpc = normal_test()
    gens = lpc.source_nodes
    for i in range(6):
        node_power,cap_increase = solve_network_flow(graph,lpc)
        print "node_power", node_power
        #print "time elapsed:", lpsolve('time_elapsed', lpc.lp), "cap increase:", np.sum(cap_increase)
        delta_power,load_shed,excess_power = get_delta_power(graph, lpc)
        print "delta power:", delta_power
        print "load shed:", load_shed
        print "excess power:", excess_power
        lpc.update(graph)
        display_graph(graph)
        increase_load(num_increase, total_increase, graph, lpc)

    # start expanding the gens
    print "Now expanding generators"
    for i in range(1):
        node_power,cap_increase = solve_network_flow(graph,lpc)
        #print "time elapsed:", lpsolve('time_elapsed', lpc.lp), "cap increase:", np.sum(cap_increase)
        #fout.write(", " + str(np.sum(cap_increase)))
        delta_power,load_shed,excess_power = get_delta_power(graph, lpc)
        print "delta power:", delta_power
        lpc.update(graph)
        increase_load(num_increase, total_increase, graph, lpc)
        increase_gens(len(lpc.source_nodes), total_increase, graph, lpc)
    
if __name__ == "__main__":
    #speed_test()
    #test_load_volatility()
    #normal_test()
    extended_normal_test()

    
