#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy

Z_CRIT = 1.0 # crit point of node
NF = 0.0004 # initial incremented load on node of the network
D_LOAD = 0.0001 # load dumped onto neighbors when node goes crit
PDF_BINS = 0 
LOAD_PDF = []
CRIT_TIMES = [] # holds the (node,time) points when nodes go crit


def scatterPlot(points):
    if(len(points) == 0):
        print "NO CRITS, NO GRAPH"
        return None
    
    x_arr = []
    y_arr = []
    for x,y in points:
        x_arr.append(x)
        y_arr.append(y)
    plt.scatter(x_arr,y_arr, s=1, marker='^')
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
        
# assumes a ring graph is already created
# K must be even
def setK(K, graph): # K = number of connections between each node
    K = K/2+1
    nodes = graph.nodes()
    for i in range(len(nodes)):
        for k in range(2,K):
            if((i+k) < len(nodes)):
                graph.add_edge(nodes[i],nodes[i+k])
            else:
                graph.add_edge(nodes[i],nodes[i+k-len(nodes)])
            
            if((i-k) >= 0):
                graph.add_edge(nodes[i],nodes[i-k])
            else:
                graph.add_edge(nodes[i],nodes[(i-k) + len(nodes)])

def setUniformWeights(min_load, max_load, graph):
    nodes = graph.nodes(data=True)
    for name,data in nodes:
        data['load'] = random.uniform(min_load, max_load-0.000001)
        data['running'] = True
    
def listToPDF(list):
    arr = list[:]
    sum = 0
    for i in range(len(arr)):
        sum += list[i]
    for i in range(len(arr)):
        arr[i] = arr[i] /(1.0 * sum)
    return arr

def addLoadToPDF(name,graph):
    node = findNode(name,graph)
    name,data=node
    LOAD_PDF[data['load']] +=1

# this might actually be faster than grabbing the index
def findNode(name,graph):
    for node in graph.nodes_iter(data=True):
        n,data = node
        if(n == name):
            return (n,data)
    return None

def getRandomNode(nodes):
    return random.choice(nodes)

def getRandomNeighbor(name,graph):
    i = random.choice(graph.neighbors(name))
    #if(graph.nodes()[name] != name):
    #    print "Bad node hit: Was looking for name", name, "got", graph.nodes()[name]
    return graph.nodes(data=True)[i]
    #    return findNode(i,graph)
    
def transferLoad(n1, n2):
    name,data = n1
    if(data['load'] > 0):
        data['load'] -= 1
        name,data = n2
        data['load'] += 1   

# Note: arg nodes must have data=True!
def transferEvent(nodes,graph):
    n1 = getRandomNode(nodes)
    n2 = getRandomNeighbor(n1[0],graph)
    transferLoad(n1,n2)
    return n2

def addLoad(node):
    name,data = node
    data['load'] += D_LOAD

def findAllCrits(nodes):
    crit_nodes = []
    for name,data in nodes:
        if(data['load'] > Z_CRIT):
            crit_nodes.append((name,data))
    if(len(crit_nodes) > 0):
        return crit_nodes
    return None

def findRunningCrits(nodes):
    crit_nodes = []
    for name,data in nodes:
        if(data['load'] > Z_CRIT and  data['running'] == True):
            crit_nodes.append((name,data))
    if(len(crit_nodes) > 0):
        return crit_nodes
    return None

def testCrit(graph):
    for name,data in graph.nodes_iter(data=True):
        if(data['load'] > Z_CRIT):
            return (name,data)
    return None

def unloadNode(crit_node, graph):
    name,data = crit_node
    nodes = graph.nodes(data=True)
    neighbors = graph.neighbors(name)
    for n in neighbors:
        name,data = nodes[n]
        data['load'] += D_LOAD    

def processCrits(time, nodes, graph):
    for node in nodes:
        #CRIT_TIMES.append((node[0],time))
        unloadNode(node,graph)
        name,data = node
        data['running'] = False
    
def handleCrit(time, node, graph):
    CRIT_TIMES.append((node[0],time))
    unloadNode(node,graph)

def handleSystemCrit(time, graph):
    while(True):
        crit_node = testCrit(graph)
        if(crit_node == None):
            break
        CRIT_TIMES.append((crit_node[0],time))
        unloadNode(crit_node,graph)

def resetSim(min_load, max_load, graph):
    setUniformWeights(min_load, max_load, graph)
    LOAD_PDF = numpy.zeros(200)
    CRIT_TIMES = []
        
# assumes graph topology does not change
def runSim(graph, iterations=100000):
    crit_events = 0
    crit_size = 0
    fout = open("temp_run.txt", 'a')
    minute_mode = True # changed from false as we add to all nodes initially
    nodes = graph.nodes(data=True)
    for n in range(iterations):
        for node in nodes:
            addLoad(node)               
        while(minute_mode):
            crit_nodes = findRunningCrits(nodes)
            if(crit_nodes == None):
                minute_mode = False
            else:
                crit_events += len(crit_nodes)
                fout.write(str(n) + " " + str(crit_events) + '\n')
                processCrits(n, crit_nodes, graph)
                if(crit_events >= PDF_BINS):
                    return PDF_BINS-1 
    return crit_events

if __name__ == "__main__":
    random.seed()
    N = int(sys.argv[1])
    K = int(sys.argv[2])
    Load = float(sys.argv[3])
    PDF_BINS = N+1
    LOAD_PDF = numpy.zeros(PDF_BINS)
    graph = nx.cycle_graph(N)
    setK(K,graph)
   
    crit_pdf = numpy.zeros(PDF_BINS)
    fout = open("crit_pdf.txt", 'w')
    for i in range(100000):
        resetSim(Load, Z_CRIT, graph)
        #for node in graph.nodes_iter(data=True):
        #    print node
        #runSim(graph,1)
        crit_pdf[runSim(graph,1)] += 1
    for i in range(PDF_BINS):
        fout.write(str(i) + " " + str(crit_pdf[i]) + '\n')
    fout.close()

    #for k in range(2,20):
    #    graph = nx.cycle_graph(N)
    #    setK(k,graph)
    #    setWeights(Load,graph)
    #    events = runSim(graph)
    #    print k, events

    #pos = nx.spring_layout(graph,iterations=200)
    #nx.draw(graph)
    
    #print "Load at Node A:", LOAD_PDF
    #plt.plot(listToPDF(LOAD_PDF))
    
    #scatterPlot(CRIT_TIMES)

    #if(len(sys.argv) > 4):
    #    plt.savefig(sys.argv[4])

    #plt.show()
