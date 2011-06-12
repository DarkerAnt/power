#!/usr/bin/python

import numpy, sys

spacing = 365

def GetEnsamble(spacing):
    data = numpy.genfromtxt(sys.argv[1])
    ensamble = []
    for i in range(len(data)/spacing):
        ensamble.append(data[i*spacing:(i+1)*spacing, 1])
    return data[:spacing,0], ensamble

def AverageEnsamble(ensamble):
    average = ensamble[0]
    for i in range(1, len(ensamble)):
        average += ensamble[i]
    return average / len(ensamble)

def FFTPowerSpec(indices, ensamble):
    sample_spacing = indices[1]-indices[0]
    for i in range(len(ensamble)):
        ensamble[i] -= numpy.mean(ensamble[i]) #remove dc component
        ensamble[i] = numpy.abs(numpy.fft.fft(ensamble[i]))**2
    freq = numpy.fft.fftfreq(ensamble[0].size, d=sample_spacing)
    return freq, ensamble

def AutoCorr(ensamble):
    #remove average
    for i in range(len(ensamble)):
        ensamble[i] -= numpy.mean(ensamble[i])
        ensamble[i] = autocorrelate(ensamble[i])
        stupid = ensamble[i]
        ensamble[i] = ensamble[i]/(1.0 * stupid[0])
    return ensamble

def autocorrelate(x):
    result = numpy.correlate(x,x, mode='full')
    return result[result.size/2:]

def PrintAverage(x, ensamble):
    ensamble = AverageEnsamble(ensamble)
    for i in range(len(ensamble)):
        print x[i], ensamble[i]


indices, ensamble = GetEnsamble(spacing)
freq, ensamble = FFTPowerSpec(indices, ensamble)
PrintAverage(freq, ensamble)
#ensamble = AutoCorr(ensamble)
#PrintAverage(range(0,ensamble[0].size), ensamble)
