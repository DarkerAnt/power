#!/usr/bin/python

# very poor mans parse tool, won't work for lots of reasons


def parse(filename):
    refs = []
    fin = open(filename, 'r')
    meta = []
    for line in fin:
        line = line.strip()
        if len(line) > 0:
            if line[0] == '@':
                meta = []
            elif line[0] == '}':
                refs.append(dict(meta))
                #foo = refs[-1]
                #print foo['title']
                #print foo['year']
            else:
                id = line[:line.find('=')]
                val = line[line.find('{')+1:line.rfind('}')]
                meta.append((id,val))
    return refs
