#!/usr/bin/python

import matplotlib.pyplot as plt
import sys, os
import numpy

colors = ['b','g','r','c','m','y','k']
c_iter = 0

bar_width_dec = 0.5 / len(sys.argv[1:])
bar_width = 0.5 + bar_width_dec
for data_file in sys.argv[1:]:
    data = numpy.genfromtxt(data_file)
    plt.bar(data[:,0], data[:,1], color=colors[c_iter], label=data_file, width=bar_width, log=True)
    bar_width -= bar_width_dec
    c_iter+=1
    if(c_iter >= len(colors)):
        c_iter = 0
        
plt.legend()
plt.show()
