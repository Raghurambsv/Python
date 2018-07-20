#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 15:00:09 2018

@author: raghuram.b
"""

import time

#### speed of program in secs by adding onebyone item into list
start=time.time()
a=range(100000000)
b=[]
for i in a:
    b.append(i*2)
    
end=time.time()
print("Time taken with appending into list",end-start)

##speed of program by doing with list comprehensions

start=time.time()
a=range(100000000)
b=[i *2 for i in a]
   
end=time.time()
print("Time taken with list comprehension",end-start)