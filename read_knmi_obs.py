# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:24:26 2016

@author: Lima
"""

import pandas as pd


def read_obs_file(file_path):
    column_names = [
        'valid_date', 'element', 'value', ' '
    ]
    df = pd.read_csv(
        file_path,
        header=None,
        names=column_names,
        usecols=[0, 1, 2],
        na_values=['', '              '],
        #parse_dates={'valid_datetime': [1, 2]},
        skipinitialspace=True
    )
    return df


df = read_obs_file('Data/new_data.txt')