#!/usr/bin/python

import csv_reader

import sys, os
import datetime
import numpy

reader = csv_reader.Reader(sys.argv[1])
data = reader.get_columns(['year','month','day','Northeast'])
#data = reader.exclude_on(data,[3])
days_between = []
for i in range(len(data)-1):
    event = datetime.date(int(data[i,0]),int(data[i,1]),int(data[i,2]))
    next_event = datetime.date(int(data[i+1,0]),int(data[i+1,1]),int(data[i+1,2]))
    days_between.append((next_event - event).days)

numpy.savetxt(sys.argv[2],days_between)
