#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy
import sys

def autocorrelate(x):
    result = numpy.correlate(x, x, mode='full')
    return result[result.size/2:]

for data_file in sys.argv[1:]:
    data = numpy.genfromtxt(data_file)
    fout = open("auto_" + data_file, 'w')
    autocorr = autocorrelate(data[0:,1])
    print data
    for i in range(len(data)):
        fout.write(str(data[i,0]) + ' ' + str(autocorr[i]/autocorr[0]) + '\n')
    fout.close()
