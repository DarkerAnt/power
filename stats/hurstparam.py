#!/usr/bin/python

import numpy, sys

# 1. For an integer m between 2 and N/2, divide the series into blocks of
#    length m and compute the sample average over each k-th block.
#    mean(X_k(m)) = 1/m * sum(X_t, t=(k-1)*m+1, km), k=1,2...,[N/m]
#
# 2. For each m, compute the sample variance of X_k(m) across the blocks.
#    s^2 = 1/([N/m]-1)* sum((mean(X_k(m)) - mean(X))^2, k=1, [N/m]) 
#
# 3. Return log(s^2_m) against log(m)
# Throw out the first points and the last points as they are unreliable
# Method is not robust to departures from standard Gaussian assumptions

def AggregatedVariance(X,cut=1):
    mean = []
    variance = []
    m = range(2, len(X)/2)

    for k in m:
        bin = []
        bin_size = range(0, len(X)+k, k)
        for i in range(1, len(bin_size)):
            bin.append(X[bin_size[i-1]:bin_size[i]])
            bin[-1] = bin[-1].sum()/len(bin[-1])
        arr = numpy.array(bin)
        mean.append(arr.mean())
        variance.append(arr.var())

    log_s2 = numpy.log(variance[cut:-cut])
    log_m = numpy.log(m[cut:-cut])

    slope,intercept = numpy.lib.polyfit(log_m, log_variance, 1)
    hurst = (slope + 2.0)/2.0
    return hurst

# Assumes spectral density f(lambda) can be approximated as:
#             f_c,H(lambda)=c(lambda)^(1-2H)
# for frequencies lamda in a neighborhood of the origin
    
def LocalWhittle(X):
