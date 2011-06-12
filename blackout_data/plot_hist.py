#!/usr/bin/python

import numpy, sys
import matplotlib.pyplot as plt

data = numpy.genfromtxt(sys.argv[1])
points = int(sys.argv[2])
(n, bins, patches) =  plt.hist(data,points)
print "Bin Size:", bins[1]-bins[0]
plt.show()
