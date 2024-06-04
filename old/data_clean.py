#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:02:50 2024

@author: danodoms
"""
import pandas as pd

# Sample DataFrame with numerical column
data = {'age': [20, 35, 50, 65, 30, 45, 55, 70],
        'salary': [40000, 60000, 80000, 50000, 70000, 30000, 90000, 55000]}

df = pd.DataFrame(data)

# Convert 'age' column to categorical
bins = [0, 30, 40, 50, 60, float('inf')]
labels = ['<30', '30-39', '40-49', '50-59', '>=60']

df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels)

print(df)

