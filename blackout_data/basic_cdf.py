#!/usr/bin/python

import sys
import csv_reader

filename = sys.argv[1]
columns = []
exclude = []
for arg in sys.argv[2:]:
    try:
        exclude.append(int(arg))
    except ValueError:
        columns.append(arg)

r = csv_reader.Reader(filename)
print r.exclude_on(r.get_columns(columns), exclude)
