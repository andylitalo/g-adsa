# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:38:48 2019

@author: Andy
"""
import numpy as np

# parameters
data_file = 'TrdGraph.TRD'
max_ct = 10000
char_len = 4


def bytes_from_file(filename, chunksize=8192):
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                for b in chunk:
                    yield b
            else:
                break
            
binary_str = bytes_from_file(data_file)

c = 0
bits = ''
for b in binary_str:
    if c == 1000:
        break
    result = np.unpackbits(np.array([b], dtype='uint8'))
    for r in result:
        bits += str(r)
    c += 1
#        
## encode binary string  
#ct = 0
#sub_ct = 0
#s = ''
#d = {}
#d['0000'] = ' '
#char_ct = 60
#num_str = ''
#for b in binary_str:
#    num_str += str(b)
#    if sub_ct == char_len-1:
#        if num_str not in d:
#            d[num_str] = chr(char_ct)
#            char_ct += 1
#        s += d[num_str]
#        sub_ct -= char_len
#        num_str = ''
#    ct += 1
#    sub_ct += 1
#    
#    if ct > max_ct:
#        break
#    
#print(s)