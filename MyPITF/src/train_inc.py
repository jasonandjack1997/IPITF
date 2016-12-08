import sys
import pickle
import numpy as np
import argparse
from pitf import PITF

def read_data(filepath):
    return np.genfromtxt(filepath, delimiter=' ', dtype=int)

p = argparse.ArgumentParser()
'''
p.add_argument("-i", "--infile", help="input tensor file", type=argparse.FileType('r'), required=True)
p.add_argument("-o", "--outfile", help="output model file", type=argparse.FileType('w'), required=True)
p.add_argument("-bm", "--basemodelfile", help="base model file", type=argparse.FileType('r'), required=True)
p.add_argument("-a", "--alpha", help="alpha (default=0.05)", type=float, nargs='?', default=0.05)
p.add_argument("-l", "--lamb", help="lambda (default=0.00005)", type=float, nargs='?', default=0.00005)
p.add_argument("-k", help="k (default=10)", type=int, nargs='?', default=10)
p.add_argument("-m", "--max_iter", help="max_iter (default=100)", type=int, nargs='?', default=100)
p.add_argument("-v", "--verbose", help="verbosity", action='store_true')
'''
args = p.parse_args()

trainingFilePathNPrefix = "..//data//taobao100"
baseModelPath = "..//data//base1000_step1000_n2.incModel"

INC_TRAIN = False
args.infile = trainingFilePathNPrefix +".train"

if INC_TRAIN == True:
    args.outfile = open(trainingFilePathNPrefix + ".incModel", "w")
else:
    args.outfile = open(trainingFilePathNPrefix + ".retrainModel", "w")
    
if INC_TRAIN == True:
    args.basemodelfile = open(baseModelPath, "r")

args.alpha = 0.01
args.lamb = 0.005
args.k = 10
args.max_iter = 500
args.verbose = True


data = read_data(args.infile)
data_shape = data[0]
data = data[1:]

if INC_TRAIN:
    basemodel = pickle.load(args.basemodelfile)
    base_latent_vectors = basemodel.latent_vector_
    
model = PITF(args.alpha, args.lamb, args.k, args.max_iter, data_shape, args.verbose) #a bigger cubic

if INC_TRAIN:
    model.fit_inc(data, base_latent_vectors, data) #incremental fit
else:
    model.fit(data, data)

pickle.dump(model, args.outfile)

args.outfile.close()

if INC_TRAIN == True:
    args.basemodelfile.close()
