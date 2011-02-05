#!/usr/bin/python

import numpy

class Reader:
    def __init__(self,filename):
        self.names = []
        self.data = []
        self.filename = filename
        self.load_file()
    def load_file(self):
        fin = open(self.filename,'r')
        self.names = fin.readline()
        self.names = self.names.split(',')
        fin.close()
        self.data = numpy.genfromtxt(self.filename,delimiter=',',names=True)
    def get_columns(self,column_names):
        column_nums = []
        for cname in column_names:
            for i in range(len(self.names)+1):
                if i == (len(self.names) + 1):
                    print "Error:", cname, "not found in column names."
                elif cname == self.names[i]:
                    column_nums.append(i)
                    break
        return self.data[:,column_nums]
    def exclude_on(self, data, excluding_cols):
        survivors = []
        for i in range(len(data)):
            keep = True
            for j in excluding_cols:
                if data[i,j] == 0.0:
                    keep = False
            if keep == True:
                survivors.append(i)
        return data[survivors,:]
