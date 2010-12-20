#!/usr/bin/python
import sys
import os

PROGRAM = "./disturbed.py"
if __name__ == "__main__":
    for K in [10]:
        for i in range(10):
            os.system(PROGRAM + " " + str(200) + " " + str(K) + " " + str(102))
            print i, "of", 10
    #os.system("mv " + "temp_run.txt " + 'K' + str(K) + "_Total_crits_at_timeme.txt")
