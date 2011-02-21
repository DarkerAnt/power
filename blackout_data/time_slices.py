#!/usr/bin/python

import numpy, sys
import matplotlib.pyplot as plt

data = numpy.genfromtxt(sys.argv[1])

bin_size = []
bin = []
mean = []
variance = []


steps = range(1,100,1)
for step in steps:
    bin = []
    bin_size = range(0, len(data)+step, step)
    for i in range(1,len(bin_size)):
        bin.append(data[bin_size[i-1]:bin_size[i]])
        bin[-1] = bin[-1].sum()/len(bin[-1])
    arr = numpy.array(bin)
    mean.append(arr.mean())
    variance.append(arr.var())
 
plt.subplot(211)    
plt.plot(steps, variance, 'b.-')
plt.subplot(212)
plt.plot(steps, mean, 'r.-')
plt.show()
