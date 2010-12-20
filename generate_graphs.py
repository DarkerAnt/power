#!/usr/bin/python
import sys
import os

PROGRAM = "./disturbed.py"
SAVEFIG_PATH = "./graphs/PDF_LOAD/"
TYPE = ".png"
if __name__ == "__main__":
    N = int(sys.argv[1])
    K = int(sys.argv[2])
    #choose correct directory
    dir = "N" + sys.argv[1] + "K" + sys.argv[2] + "/"
    for i in [2,5,8]:
        for j in [1,2]:
            filename = "load_all" + str(i) + "_" + str(j)
            print "generating " + SAVEFIG_PATH + dir + filename + TYPE
            os.system(PROGRAM + " " + sys.argv[1] + " " + sys.argv[2] + " " + str(i) + " " + SAVEFIG_PATH + dir + filename + TYPE)
