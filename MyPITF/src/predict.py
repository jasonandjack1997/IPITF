import sys
import pickle
import argparse
import numpy as np
from pitf import PITF

def read_data(filepath):
    return np.genfromtxt(filepath, delimiter=' ', dtype=int)

filePathNPrefix = "..//data//base1000_step1000_n1"

p = argparse.ArgumentParser()
p.add_argument("-it", "--in_test_file", help="input test file", type=str, default = filePathNPrefix +".test")
p.add_argument("-m", "--modelfile", help="input model file", type=str, default = filePathNPrefix + ".incModel")
p.add_argument("-op", "--out_predicted_file", help="output predicted file (default=STDOUT)", type=str, nargs='?', default=sys.stdout)
args = p.parse_args()

N = 3 #the top n estimated tags

data = read_data(args.in_test_file)


model = pickle.load(open(args.modelfile, "r"))

predicted = model.predict_topN(data, N)

#np.savetxt(args.out_predicted_file, predicted, fmt='%.0f')

answer = data[:, N - 1]

print (answer==predicted[:, 2]).mean()

