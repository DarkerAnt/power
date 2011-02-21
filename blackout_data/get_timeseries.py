#!/usr/bin/python

import csv_reader
import sys
import datetime

reader = csv_reader.Reader(sys.argv[1])
data = reader.get_columns(['year', 'month', 'day', 'Northeast'])

start = datetime.date(int(data[0,0]), int(data[0,1]), int(data[0,2]))
end = datetime.date(int(data[-1,0]), int(data[-1,1]), int(data[-1,2]))

weeks = range(0,(end-start).days/7+1) 
occur = []
for i in weeks:
    occur.append(0)

for year,month,day,northeast in data:
    current_day = datetime.date(int(year), int(month), int(day))
    if northeast == 1:
        occur[(current_day-start).days/7] += 1

for i in range(len(weeks)):
    print int(weeks[i]), occur[i]
