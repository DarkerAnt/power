#!/usr/bin/python

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import numpy as np


def create_cluster(size, mu, sigma):
    x,y = mu
    return [(random.gauss(x,sigma), random.gauss(y,sigma)) for i in range(size)]
    
def init_graph():
    clusters = 10
    cluster_size = 10
    p = []
    for i in range(clusters):
        x = random.uniform(-10,10)
        y = random.uniform(-10,10)
        p.extend(create_cluster(cluster_size, (x,y), 2))
    p = dict(zip(range(len(p)), p))
    graph = nx.random_geometric_graph(len(p), 5, pos=p)
    graph = graph.to_directed()

    for u,d in graph.nodes(data=True):
        d['alive'] = True
        d['anger'] = 0
        d['total_friend'] = 0
        d['num_deaths'] = 0
        d['death_priority'] = 0

    for u,v,d in graph.edges(data=True):
        d['friend'] = random.gauss(5,3)
        graph.node[u]['total_friend'] += d['friend']
        
    cluster_edges = graph.edges(data=True)
    
    for u in graph.node:
        for v in graph.node:
            if u != v:
                if v not in graph.edge[u]:
                    graph.add_edge(u,v, friend=np.random.rayleigh(0.15))
                    #print graph.edge[u][v]['friend']
                    graph.node[u]['total_friend'] += graph.edge[u][v]['friend']
    return graph, cluster_edges

def kill_dwarf(graph, name, deaths):
    if graph.node[name]['alive'] == False:
        print "oops"
    graph.node[name]['alive'] = False
    graph.node[name]['num_deaths'] += 1
    graph.node[name]['death_priority'] += len(graph.node)-len(deaths)
    for n in graph.edge[name]:
        graph.node[n]['anger'] += graph.edge[n][name]['friend']
        graph.node[n]['total_friend'] -= graph.edge[n][name]['friend']
        graph.remove_edge(n,name)
    deaths.append(name)

def tantrum(graph, name, size, deaths):
    for murder in range(size):
        lottery = random.uniform(0,graph.node[name]['total_friend'])
        for victim in graph.edge[name]:
            lottery -= graph.edge[name][victim]['friend']
            if lottery <= 0:
                kill_dwarf(graph, victim, deaths)
                break
    if graph.node[name]['alive'] == True: # it isn't one of the initial deaths
        kill_dwarf(graph, name, deaths)

def prevent_kills(graph, node_names):
    for n1 in node_names:
        for n2 in node_names:
            if n1 != n2:
                graph.node[n1]['total_friend'] -= graph.edge[n1][n2]['friend']
                graph.remove_edge(n1,n2)

def run_sim(graph, calamities=2, kill=5, tantrum_threshold=50):
    deaths = []
    initial_deaths = random.sample(graph.node, calamities)
    prevent_kills(graph, initial_deaths)

    for name in initial_deaths:
        tantrum(graph, name, kill, deaths)

    tantrums = True
    while(tantrums):
        tantrums = False
        new_tantrums = []
        for name in graph.node:
            if graph.node[name]['alive'] == True:
                if graph.node[name]['anger'] >= tantrum_threshold:
                    tantrums = True
                    new_tantrums.append(name)
        prevent_kills(graph, new_tantrums)
        for name in new_tantrums:
            tantrum(graph, name, kill, deaths)
            
    print "deaths:", len(deaths)
    return (graph, deaths)

def plot_graph(graph, node_weights, edges=None):
    pos=nx.get_node_attributes(graph,'pos')
    dmin=1
    ncenter=0
    for n in pos:
        x,y=pos[n]
        d=(x-0.5)**2+(y-0.5)**2
        if d<dmin:
            ncenter=n
            dmin=d
    #p=nx.single_source_shortest_path_length(graph,ncenter)
    plt.figure(figsize=(8,8))
    nx.draw_networkx_edges(graph,pos,edgelist=edges,alpha=0.4)
    nx.draw_networkx_nodes(graph,pos,nodelist=node_weights.keys(),
                           node_size=80,
                           node_color=node_weights.values(),
                           cmap=plt.cm.Reds)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    random.seed()
    graph, cluster_edges = init_graph()
    
    death_priority = dict(zip(graph.node, np.zeros(len(graph.node))))
    for i in range(100):
        sim_graph,deaths = run_sim(graph.copy())
        while(len(deaths) < len(graph.node)/3):
            sim_graph,deaths = run_sim(graph.copy())
        for node in death_priority:
            death_priority[node] += sim_graph.node[node]['death_priority']
    print death_priority
    plot_graph(graph, death_priority, cluster_edges)
