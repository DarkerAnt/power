#!/usr/bin/python

import sys, numpy

def autocorrelate(x):
    result = numpy.correlate(x,x, mode='full')
    return result[result.size/2:]

data = numpy.genfromtxt(sys.argv[1])
data[:,1] = data[:,1]-data[:,1].mean()
autocorr = autocorrelate(data[:,1])
autocorr = autocorr/autocorr[0]
for i in range(len(data)):
    print data[i,0], autocorr[i]
