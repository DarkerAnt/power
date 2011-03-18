#!/usr/bin/python

import numpy, sys

x = numpy.arange(0,100000)/1000.0
A = 1.0
f = 80.0
omega = 2*numpy.pi*f
theta = 0#numpy.pi
y = A * numpy.sin(omega*x + theta)

for i in range(len(y)):
    print str(x[i]) + ' ' + str(y[i])
