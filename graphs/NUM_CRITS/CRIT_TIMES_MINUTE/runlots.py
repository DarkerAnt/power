#!/usr/bin/python
import sys
import os

PROGRAM = "./disturbed.py"
if __name__ == "__main__":
    i = 0
    for K in [2,6,10]:
        #for i in range(10):
        i+=1
        os.system(PROGRAM + " " + str(200) + " " + str(K) + " " + str(102))
        print i, "of", 3
        os.system("mv " + "temp_run.txt " + 'K' + str(K) + "_crits_at_time.txt")
