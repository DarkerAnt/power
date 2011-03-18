#!/usr/bin/python

import numpy, sys
import matplotlib.pyplot as plt

N = 100
steps = range(2,N/2)

data = numpy.genfromtxt(sys.argv[1])
bin_size = []
bin = []
mean = []
variance = []


for step in steps:
    bin = []
    bin_size = range(0, len(data)+step, step)
    for i in range(1,len(bin_size)):
        bin.append(data[bin_size[i-1]:bin_size[i]])
        bin[-1] = bin[-1].sum()/len(bin[-1])
    arr = numpy.array(bin)
    mean.append(arr.mean())
    variance.append(arr.var())

log_steps = numpy.log(steps)
log_variance = numpy.log(variance)
log_mean = numpy.log(mean)

slope,intercept = numpy.lib.polyfit(log_steps, log_variance, 1)
hurst = (slope + 2.0)/2.0

plt.subplot(211)    
plt.title("Hurst: " + str(hurst))
plt.xlabel("loglog Variance")
plt.plot(log_steps, log_variance, 'b.-') 
#plt.plot(steps, variance, 'b.-')
plt.subplot(212)
plt.xlabel("loglog Mean")
plt.plot(log_steps, log_mean, 'r.-') 
#plt.plot(steps, mean, 'r.-')

print "Hurst:", hurst
plt.show()

