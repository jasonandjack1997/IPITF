import sys
import pickle
import numpy as np
import argparse
from pitf import PITF

def read_data(filepath):
    return np.genfromtxt(filepath, delimiter=' ', dtype=int)

data = read_data("sample.allData")
data_shape = data[0]
data = data[1:]

dataList = list()
for a, b, c in data:
    dataList.append([a, b, c])
    
dataList.sort()


np.savetxt("reorder.txt",dataList, fmt='%.0f')

