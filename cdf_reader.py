#!/usr/bin/python

class Reader:
    def __init__(self,filename):
        self.buses = []
        self.branches = []
        self.loadFile(filename)
    def loadFile(self,filename):
        fin = open(filename,'r')
        read_states = {'header': 1, 'bus': 2, 'branch': 3, 'footer': 4}
        state = read_states['header']
        for line in fin:
            if(state == read_states['header']):
                loc = line.find("BUS DATA FOLLOWS")
                if(loc != -1):
                    state = read_states['bus']
            elif(state == read_states['bus']):
                loc = line.find("BRANCH DATA FOLLOWS")
                if(loc != -1):
                    state = read_states['branch']
                else:
                    data = line.split()
                    if(data[0] != '-999'):
                        self.buses.append(int(data[0]))
            elif(state == read_states['branch']):
                loc = line.find("-999")
                if(loc != -1):
                    state = read_states['footer']
                else:
                    data = line.split()
                    self.branches.append((int(data[0]), int(data[1])))
        fin.close()
    def startBusAtZero(self):
        shift = self.buses[0]
        for i in range(len(self.buses)):
            self.buses[i] -= shift
        for i in range(len(self.branches)):
            b1,b2 = self.branches[i]
            self.branches[i] = (b1-shift, b2-shift)
        
