import sys
import numpy as np
import argparse

p = argparse.ArgumentParser()
p.add_argument("-i", "--infile", help="input data file", type=str, default = "sample_ordered.fullData")
p.add_argument("-o", "--outfile", help="output file prefix", type=str, default = "sample_")
p.add_argument("-t", "--test_ratio", help="test ratio (default=0.3)", type=float, nargs='?', default=0.3)
p.add_argument("-su", "--sizeU", help="size of u", type=int, default=25000)
p.add_argument("-si", "--sizeI", help="size of i", type=int, default=5000)
args = p.parse_args()

BASE_SIZE_U = 5000

data = np.genfromtxt(args.infile, dtype=int)
datasize2 = [args.sizeU, args.sizeI, data[0][2]]
datasize1 = [BASE_SIZE_U, args.sizeI, data[0][2]]
data = data[1:]


dataList2 = list()
dataList1 = list()


for u, i, t in data:
    if u < args.sizeU and i < args.sizeI:
        dataList2.append([u,i,t])
    if u < BASE_SIZE_U and i < BASE_SIZE_U:
        dataList1.append([u,i,t])
        

data2 = np.asarray(dataList2)
data1 = np.asarray(dataList1)
    
test_index2 = np.random.choice(data2.shape[0], size=int(args.test_ratio*data2.shape[0]), replace=False)
train_index2 = np.setdiff1d(np.arange(data2.shape[0]), test_index2)

test_index1 = np.random.choice(data1.shape[0], size=int(args.test_ratio*data1.shape[0]), replace=False)
train_index1 = np.setdiff1d(np.arange(data1.shape[0]), test_index1)



with open(args.outfile+'2.train', 'w') as f:
    np.savetxt(f, np.vstack([datasize2,data2[train_index2]]), fmt='%.0f')
with open(args.outfile+'2.test', 'w') as f:
    np.savetxt(f, data2[test_index2], fmt='%.0f')


        
        
with open(args.outfile+'1.train', 'w') as f:
    np.savetxt(f, np.vstack([datasize1,data1[train_index1]]), fmt='%.0f')
with open(args.outfile+'1.test', 'w') as f:
    np.savetxt(f, data1[test_index1], fmt='%.0f')


print "done"
