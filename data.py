# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 12:25:49 2016

@author: Lima
"""

from json import load


def load_json(path):
    with open(path,'r') as f:
        data = load(f)
    return data
    
data = load_json('Data/amsterdam (2).json')
for i in range(0,3):
    print(data[i]['_id'])
