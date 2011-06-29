#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy
import pylon
# map networkx to pylon

NUM_BUS_INCREMENT = 5
BUS_INCREMENT = (0,2)
EDGE_INCREMENT = (0, 5)
NUM_GENERATING_INCREMENT = 1
GENERATING_MARGIN = 0.10
GENERATING_INCREMENT = (0, 8)

total_demand = 0.0
total_capacity = 0.0
total_generation = 0.0
buses = []
generators = []
branches = []

def set_demand(bus, demand):
    global total_demand
    name,data = bus
    

#def add_demand(bus, demand):
#def set_generating(gen, power):
#def add_generating(gen, power):
#def set_capacity(branch, capacity):
#def add_capacity(branch, capacity):
def update_buses(buses):
    selected_buses = random.sample(buses, NUM_BUS_INCREMENT)
    for bus in selected_buses:
        bus.p_demand += random.uniform(BUS_INCREMENT[0], BUS_INCREMENT[1])

def update_branches(branches):
    selected_branches = random.sample(branches, NUM_BRANCH_INCREMENT)
    for branch in selected_branches:
        increase = random.uniform(BRANCH_INCREMENT[0],BRANCH_INCREMENT[1])
        branch.rate_a += increase 
        branch.rate_b += increase
        branch.rate_c += increase

def run_sim(graph, iterations):
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    failed_branches = []
    
    case = pylon.Case(buses=buses, branches=branches, generators=generators)
    for i in range(iterations):
        update_buses(buses)
        update_branches(branches)
        update_generators(generators)
        compute_flow(case)
        failed_branches = find_failed_branches(branches)
        
if __name__ == "__main__":
    #global buses
    #global generators
    #global branches
    dims = (5,5)
    generator_supply = 10
    init_demand = 60.0
    init_generation = 200.0
    init_capacity = 100.0
    iterations = 1

    
    random.seed()
    graph = nx.generators.classic.grid_2d_graph(dims[0], dims[1])
    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)
    gen_nodes = [nodes[dims[0]*dims[1]/2]]
    bus_nodes = nodes[:dims[0]*dims[1]/2]
    bus_nodes.extend(nodes[dims[0]*dims[1]/2+1:])

    for node in bus_nodes:
        name,data = node
        data['type'] = 'bus'
        data['bus'] = pylon.Bus(name=str(name), p_demand=init_demand, q_demand=0.0)
        buses.append(data['bus'])
    for gen in gen_nodes:
        name,data = gen
        data['type'] = 'generator'
        if len(generators) == 0:
            data['bus'] = pylon.Bus(name=str(name), type=pylon.REFERENCE)
        else:
            data['bus'] = pylon.Bus(name=str(name))
        buses.append(data['bus'])
        data['generator'] = pylon.Generator(data['bus'], p_max=init_generation)
        generators.append(data['generator'])
    dict_nodes = dict(nodes)
    for edge in edges:
        name1,name2,data = edge
        data['type'] = 'branch'
        node1 = dict_nodes[name1]
        node2 = dict_nodes[name2]
        data['branch'] = pylon.Branch(node1['bus'],node2['bus'], rate_a=init_capacity, rate_b=init_capacity, rate_c=init_capacity)
        branches.append(data['branch'])
 
    run_sim(graph, iterations)
