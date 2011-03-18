#!/usr/bin/python

import sys, numpy
import matplotlib.pyplot as plt

data = numpy.genfromtxt(sys.argv[1])
data[:,0] = numpy.log10(numpy.abs(data[:,0]))
data[:,1] = numpy.log10(numpy.abs(data[:,1]))
coefs = numpy.lib.polyfit(data[1:11,0],data[1:11,1], 1)
best_fit = numpy.lib.polyval(coefs, data[:,0])
p1 = plt.plot(data[:,0], data[:,1], '+-')
p2 = plt.plot(data[:,0], best_fit, '--')
plt.legend((p1[0], p2[0]), ('loglog ', 'slope: ' + str(coefs[0])))
print "slope:", coefs[0]
plt.show()
