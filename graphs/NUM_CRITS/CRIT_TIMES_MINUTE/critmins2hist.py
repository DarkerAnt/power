#!/usr/bin/python

import numpy
import sys

def array2file(arr,file):
    stop = 0
    for i in range(len(arr)-1,0,-1):
        if(arr[i] == 0):
            stop = i
        else:
            break
    for i in range(stop):
        file.write(str(i) + " " + str(int(arr[i])) + '\n')

if __name__ == "__main__":
    filename = sys.argv[1]
    fout = open(filename[:-11] + "hist.txt", 'w')
    fin = open(filename, 'r')
    hist = numpy.zeros(100000)

    for line in fin:
        line = line.split()
        hist[int(line[1])] += 1

    array2file(hist, fout)
    
