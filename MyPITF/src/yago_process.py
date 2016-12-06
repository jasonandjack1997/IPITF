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

entitySet = set()
MAX_ENTITY_COUNT = 100
id = 0
triple_list = list()

tripleCount = 10000000
with open("..//data//yagoFacts.ttl", "r") as f:
    for line in f:
        if tripleCount < 0:
            break
        tripleCount -= 1       
        if (not str(line).startswith("#@")):
            s, p, o = str(line).split("\t")
            if len(entitySet) <= MAX_ENTITY_COUNT and not s in entitySet:
                entitySet.add(s)
            if s in entitySet:
                triple_list.append([s, p, o])

    
    

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