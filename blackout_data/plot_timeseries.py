#!/usr/bin/python

import sys, numpy
import matplotlib.pyplot as plt

data = numpy.genfromtxt(sys.argv[1])
plt.plot(data[:,0],data[:,1], '.-')
plt.show()
