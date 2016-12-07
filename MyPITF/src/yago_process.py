'''
Created on 6Dec.,2016

@author: uqwwan10
'''
import sys
import pickle
import numpy as np
import argparse
from pitf import PITF

def read_data(filepath):
    return np.genfromtxt(filepath, delimiter='\t', dtype=str, comments = "#@", max_rows = None)

#triples = read_data("..//data//yagoFacts.ttl")

subjectSet = set()
objectSet = set()
MAX_SUBJECT_COUNT = 1000
id = 0
triple_list = list()

tripleCount = None
with open("..//data//yagoFacts.ttl", "r") as f:
    for line in f:
        if tripleCount != None:
            if tripleCount < 0:
                break
            tripleCount -= 1       
        if (not str(line).startswith("#@")):
            s, p, o = str(line).split("\t")
            if len(subjectSet) < MAX_SUBJECT_COUNT and not s in subjectSet:
                subjectSet.add(s)
              
            if s in subjectSet:
                triple_list.append([s, p, o])
                objectSet.add(o)
                            

#objectSet = set(np.asarray(triple_list)[:, 2])

print "len(subjectSet), len(objectSet): ", len(subjectSet), len(objectSet)


 
subject_id_dict = dict()
object_id_dict = dict()
relation_id_dict = dict()
 
id = 0
for name in subjectSet:
    subject_id_dict[name] = id
    id += 1

id = 0
for name in objectSet - subjectSet:
    object_id_dict[name] = id
    id += 1
 
id = 0
for name in set(np.asarray(triple_list)[:,1]):
    relation_id_dict[name] = id
    id += 1
 
id_triples = list()

for s, p, o in triple_list:
    s_id = subject_id_dict.get(s)
    o_id = object_id_dict.get(o)
    p_id = relation_id_dict.get(p)
    id_triples.append([s_id, o_id, p_id])
     
dataShape = [len(subject_id_dict), len(object_id_dict), len(relation_id_dict)]
id_triples_ndarray = np.asarray(id_triples, dtype=np.float32)
np.savetxt("..//data//id_triples" + str(MAX_SUBJECT_COUNT/1000) + "K.train", np.vstack([dataShape, id_triples_ndarray]), fmt="%.0f %.0f %.0f")

print "done"