#!/usr/bin/python

import bibparse
import sys

bibs = bibparse.parse(sys.argv[1])
bibs = sorted(bibs, key=lambda bib : int(bib['year']))

bib = bibs[0]
year = bib['year']

print "### " + '[' + year + ']'
for bib in bibs:
    if year != bib['year']:
        year = bib['year']
        print "\n### " + '[' + year + ']'
    print '*\t' + '**' + bib['title'] + '**'
    print ""
    print '\t* ' + '_' +  bib['author'] + '_'
    
