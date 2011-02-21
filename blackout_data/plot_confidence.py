#!/usr/bin/python

import sys, numpy, math
import matplotlib.pyplot as plt

data = numpy.genfromtxt(sys.argv[1])
confidence_95 = numpy.linspace(1,1,len(data))
confidence_95 *= 1.96/math.sqrt(len(data))
plt.loglog(data[:,0],numpy.abs(data[:,1]), '.-')
#plt.loglog(data[:,0],confidence_95, '-')
plt.show()
