#!/usr/bin/python

import sys,os,math
import scipy

def phi(x):
    if x < 0:
        return 0
    elif (x <= 1):
        return x
    else:
        return 0

def critProb(n,r,d,p):
    return scipy.misc.comb(n,r)*phi(d)*(d+r*p)**(r-1)*(phi(1.0-d-r*p)**(n-r))

if __name__ == "__main__":
    n = int(sys.argv[1])
    r = int(sys.argv[2])
    D = float(sys.argv[3])
    P = float(sys.argv[4])
    load_min = float(sys.argv[5])
    load_max = float(sys.argv[6])
    load_fail = load_max
    p = P/(load_max - load_min)
    d = (D+load_max-load_fail)/(load_max-load_min)
    print critProb(n,r,d,p)
