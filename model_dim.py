from pandas import read_csv, DataFrame
import numpy as np
from collections import Counter

dt_colnames = ['age', 'year', 'nodes', 'class']

dataset = 'BC.csv'
read_dt = read_csv(dataset, header=None, names=dt_colnames)

# Calculate target distribution
targets = read_dt['class'].values
target_counter = Counter(targets)

for key, value in target_counter.items():
    percent = value / len(targets) * 100
    print(f'Class={key}, Count={value}, Percent={round(percent, 4)}%')