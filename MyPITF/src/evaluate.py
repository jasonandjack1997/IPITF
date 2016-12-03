import sys
import argparse
import numpy as np

def read_data(filepath):
    return np.genfromtxt(filepath, delimiter=' ', dtype=int)

p = argparse.ArgumentParser()
p.add_argument("-t", "--testfile", help="answer in test file (ground truth)", type=argparse.FileType('r'), required=True)
p.add_argument("-r", "--resultfile", help="result file", type=argparse.FileType('r'), required=True)
args = p.parse_args()

answer = read_data(args.testfile)
answerList = list()
for u, i, t in answer:
    answerList.append(t)

#answer = np.asarray(answerList)

predicted = read_data(args.resultfile)


print (answer==predicted).mean()
hit = 0
count = 0

for i in range (0, len(answer)):
    if answer[i][2] == predicted[i][2]:
        hit += 1
        print answer[i][2], predicted[i][2]
    count += 1

print hit*1.0/count

