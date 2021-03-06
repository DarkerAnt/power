#!/usr/bin/python
import sys
import os
import numpy

PROGRAM = "./disturbed.py"
NUM_NODES = 100
NODE_CONNECTIONS = 4
CRIT_PDF = numpy.zeros(NUM_NODES)

def myRange(start, stop, step):
    r = []
    num = start
    while(num  < stop):
        r.append(num)
        num += step
    return r

if __name__ == "__main__":
    fout = open("temp_crits.txt",'w')
    i = 0
    loads = myRange(0.9955, 0.998, 0.0005)#[9.965] #range(0,9,2)
    #loads = [0.0,0.5,0.8,0.9955,0.9965,0.998,0.3,0.4,0.6,0.7]
    for min_load in loads:
        print "min_load:", min_load
        #os.system(PROGRAM + " " + str(NUM_NODES) + " " + str(NODE_CONNECTIONS) + " " + str(min_load))
        os.system(PROGRAM + " " + str(min_load))
        os.system("mv " + " crit_pdf.txt " + 'Lmin_' + str(min_load) + "_crit_PDF.txt")
        i+=1
        print "Processed ", i, "of", len(loads)
