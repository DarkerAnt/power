#!/usr/bin/python
import numpy, sys
import matplotlib.pyplot as plt

data = numpy.genfromtxt(sys.argv[1])
plt.title(sys.argv[1])
plt.plot(data[:,0], data[:,1])
plt.show()
