#!/usr/bin/python

import numpy
import sys

#def findLargest(

if __name__ == "__main__":
    filename = sys.argv[1]
    fout = open(filename[:-8] + "pdf.txt", 'w')
    fin = open(filename, 'r')
    
    sum = 0.0
    for line in fin:
        line = line.split()
        sum += int(line[1])

    fin.close()
    fin = open(filename, 'r')
    
    for line in fin:
        line = line.split()
        fout.write(line[0] + " " + str(int(line[1])/sum) +'\n')

