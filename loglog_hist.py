#!/usr/bin/python

import matplotlib.pyplot as plt
import sys, os
import numpy
from scipy import stats
    

colors = ['b','g','r','c','m','y','k']

data_file = sys.argv[1]
num_bins = 100
if len(sys.argv) > 2:
    num_bins = int(sys.argv[2])

data = numpy.genfromtxt(data_file)
#temp_bins = [0,20000,50000,70000,100000,130000,150000,200000,300000,400000,500000,2000000]
time_between_bins = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
hist,bin_edges = numpy.histogram(data, bins=time_between_bins,normed=True)

slope, intercept, r_value, p_value, std_err = stats.linregress(bin_edges[:-1], hist)
print hist
print bin_edges
print slope
plt.loglog(bin_edges[:-1], hist,'.', label=data_file)
plt.legend()
plt.show()
