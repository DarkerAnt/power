#!/usr/bin/python

import matplotlib.pyplot as plt
import sys, os
import numpy

def myplot(ax):
    

colors = ['b','g','r','c','m','y','k']
c_iter = 0
plt.hold(True)
for data_file in sys.argv[1:]:
    data = numpy.genfromtxt(data_file)
    plt.bar(data[:,0], data[:,1], color=colors[c_iter], log=True)
    c_iter+=1
    if(c_iter >= len(colors)):
        c_iter = 0
        
plt.legend(sys.argv[1:], colors)
plt.show()
