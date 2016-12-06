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

triples = read_data("..//data//yagoFacts.ttl")
subjectSet = set(triples[:, 0])
predicateSet = set(triples[:, 1])
objectSet = set(triples[:, 2])

entitySet = subjectSet | objectSet

entity_id_dict = dict()
relation_id_dict = dict()
id = 0

for name in entitySet:
    entity_id_dict[name] = id
    id += 1

id = 0
for name in predicateSet:
    relation_id_dict[name] = id
    id += 1

#with open("..//data//500K//triples.pickle", "w") as f:
#    pickle.dump(triples, f)
with open("..//data//500K//entity_id_dict.pickle", "w") as f:
    pickle.dump(entity_id_dict, f)
with open("..//data//500K//relation_id_dict.pickle", "w") as f:
    pickle.dump(relation_id_dict, f)

id_triples = list()

for s, p, o in triples:
    s_id = entity_id_dict.get(s)
    o_id = entity_id_dict.get(o)
    p_id = relation_id_dict.get(p)
    id_triples.append([s_id, p_id, o_id])
    

id_triples_ndArray = np.asarray(id_triples)
np.savetxt("..//data//500K//id_triples.txt", id_triples_ndArray, fmt='%.0f')
print "done"