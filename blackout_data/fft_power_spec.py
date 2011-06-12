#!/usr/bin/python

import numpy, sys
import matplotlib.pyplot as plt

#data = numpy.sin(1.0/4.0*numpy.pi + numpy.arange(0,10000)/100.0*3.0/4.0*numpy.pi)
#x = numpy.arange(-data.size/2, data.size/2)/100.0

data = numpy.genfromtxt(sys.argv[1])
sample_spacing = data[1,0] - data[0,0]
data = data[:,1]
# take out the dc component
data = data - numpy.mean(data)
fft = numpy.fft.fft(data)
sfreq = numpy.fft.fftfreq(data.size, d=sample_spacing)

plt.title(sys.argv[1])
plt.xlabel('Frequency (Hz)')
max_loc = numpy.argmax(numpy.abs(fft)**2)
print "MAX freq:", abs(sfreq[max_loc]), 'period:', 1.0/(abs(sfreq[max_loc]))
plt.plot(sfreq, abs(fft)**2, 'r')
plt.show()
