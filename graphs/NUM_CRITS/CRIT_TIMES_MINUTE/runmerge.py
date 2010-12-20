#!/usr/bin/python
import sys
import os
import numpy

PROGRAM = "./disturbed.py"
ITERS = 10

def addfile(file, arr):
    file = open(file, 'r')
    for time,crits in file:
        arr[int(time)] += int(crits)
    #for line in file:
    #    line = line.split()
    #    time = int(line[0])
    #    crits = int(line[1])
    #    arr[time] += crits
        
if __name__ == "__main__":
    
    crits = numpy.zeros(100000)
    for n in range(ITERS):
        os.system(PROGRAM + " " + str(200) + " " + str(10) + " " + str(102))
        print n, 'of', ITERS
        addfile("temp_run.txt", crits)
        print "finished processing", n
    fout = fopen('crits_at_time.txt', 'w')
    for i in range(len(crits)):
        fout.write(str(i) + ' ' + str(crits[i]) + '\n')
