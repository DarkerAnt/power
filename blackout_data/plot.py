#!/usr/bin/python
import numpy, sys
import matplotlib.pyplot as plt

data = numpy.genfromtxt(sys.argv[1])
plt.title(sys.argv[1])
if data.ndim > 1:
    plt.plot(data[:,0], data[:,1])
else:
    plt.plot(data)
plt.show()
