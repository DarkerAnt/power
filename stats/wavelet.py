#!/usr/bin/python

import numpy
import matplotlib.pyplot as plt

x = numpy.linspace(0, 2*numpy.pi, 100)
sinwave = numpy.sin(x)

lowpass = numpy.sinc(x)
lowpass[len(lowpass)/2:] = numpy.sinc(-x[len(x)/2:])
lowpass = numpy.fft.fft(lowpass)

highpass = numpy.sinc(x*numpy.pi)
highpass[len(highpass)/2:] = numpy.sinc(-x[len(x)/2:])
for i in range(len(highpass)):
    highpass *= numpy.exp(complex(0.0, numpy.pi * i))
highpass = numpy.fft.fft(highpass)

ha = lowpass.copy()
ga = highpass.copy()
for i in range(len(lowpass)/2):
    ha[i] = lowpass[i+len(lowpass)/2].real
    ga[i] = highpass[i+len(highpass)/2].real
    ha[i+len(lowpass)/2] = lowpass[i].real
    ga[i+len(highpass)/2] = highpass[i].real
    
hs = ha.copy()
gs = ga.copy()
for i in range(len(lowpass)):
    hs[i] = ha[-i]
    gs[i] = ga[-1]


plt.plot(ha)
plt.plot(ga)
plt.show()
