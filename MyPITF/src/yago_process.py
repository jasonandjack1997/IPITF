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
    return np.genfromtxt(filepath, delimiter='\t', dtype=str, comments="#@", max_rows=None)

# triples = read_data("..//data//yagoFacts.ttl")

def selectN2Id():
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
                                
    
    # objectSet = set(np.asarray(triple_list)[:, 2])
    
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
    for name in set(np.asarray(triple_list)[:, 1]):
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
    np.savetxt("..//data//id_triples" + str(MAX_SUBJECT_COUNT / 1000) + "K.train", np.vstack([dataShape, id_triples_ndarray]), fmt="%.0f %.0f %.0f")
    
    print "done"
    return

def incrementalSelection():
    INIT_SUBJECT_COUNT = 1000
    SUBJECT_INCREMENT = 1000
    INCREMENT_COUNT = 5
    TEST_RATIO = 0.3

    subjectSet = set()
    objectSet = set()
    relationSet = set()
    tripleIdList = list()
    
    subject_id_dict = dict()
    object_id_dict = dict()
    relation_id_dict = dict()
    
    trainingTriples = list()
    testingTriples = list()
    
    
    for step in range (0, INCREMENT_COUNT):
        
        deltaSubjectSet = set()
        deltaObjectSet = set()
        deltaRelationSet = set()
        deltaTripleList = list()
        deltaTripleIdList = list()
        
        tripleCount = None
        
        # build pure incremented subject, object, relation set and triple set.
        # only new subjects, object, relations are added to the delta sets
        with open("..//data//yagoFacts.ttl", "r") as f:
            for line in f:
                if tripleCount != None:
                    if tripleCount < 0:
                        break
                    tripleCount -= 1       
                if (not str(line).startswith("#@")):
                    s, r, o = str(line).split("\t")
                    maxSize = INIT_SUBJECT_COUNT if step == 0 else SUBJECT_INCREMENT
                    if len(deltaSubjectSet) < maxSize and not s in subjectSet:
                        deltaSubjectSet.add(s)
                      
                    if s in deltaSubjectSet:
                        if o not in objectSet:
                            deltaObjectSet.add(o)
                        if r not in relationSet:
                            deltaRelationSet.add(r)
                        deltaTripleList.append([s, r, o])
        
        
        # give new subjects, objects, relations ids and add to the id lookup dict 
        id = len(subjectSet)
        for name in deltaSubjectSet:
            subject_id_dict[name] = id
            id += 1
        
        id = len(objectSet)
        for name in deltaObjectSet:
            object_id_dict[name] = id
            id += 1
        
        id = len(relationSet)
        for name in deltaRelationSet:
            relation_id_dict[name] = id
            id += 1
        
        # check sets building
        if len(subjectSet & deltaSubjectSet) + len(objectSet & deltaObjectSet) + len(relationSet & deltaRelationSet) > 0:
            print "set building error"
            return
        
        # update sets with new elements
        subjectSet |= deltaSubjectSet
        objectSet |= deltaObjectSet
        relationSet |= deltaRelationSet
        
        tensorShape = [len(subjectSet), len(objectSet), len(relationSet)]
        

        # build new triples with ids
        for s, r, o in deltaTripleList:
            s_id = subject_id_dict.get(s)
            o_id = object_id_dict.get(o)
            r_id = relation_id_dict.get(r)
            deltaTripleIdList.append([s_id, o_id, r_id])
        tripleIdList.extend(deltaTripleIdList)
        npDelataTripleIdList = np.asarray(deltaTripleIdList)
        
        
        # split the triples into training and testing
        try:
            deltaTrainingIndex = np.random.choice(npDelataTripleIdList.shape[0],int((1 - TEST_RATIO) * len(deltaTripleIdList)), replace = False)
            deltaTestingIndex = np.setdiff1d(np.arange(npDelataTripleIdList.shape[0]), deltaTrainingIndex)
            deltaTrainingTriples = npDelataTripleIdList[deltaTrainingIndex]
            deltaTestingTriples = npDelataTripleIdList[deltaTestingIndex]
        except ValueError as err:
            print " error", err
        trainingTriples.extend(deltaTrainingTriples)
        testingTriples.extend(deltaTestingTriples)
        
        
        # load based training and testing files, rename, and append new triples to the files, then write file
        if step == 0:
            filePathNPrefix = "..//data//base{}_base".format(INIT_SUBJECT_COUNT)
        else: 
            filePathNPrefix = "..//data//base{}_step{}_n{}".format(INIT_SUBJECT_COUNT, SUBJECT_INCREMENT, step)
        
        np.savetxt(filePathNPrefix + ".train", np.vstack([tensorShape, trainingTriples]), fmt = "%.0f %.0f %.0f")
        np.savetxt(filePathNPrefix + ".test", testingTriples, fmt = "%.0f %.0f %.0f")
        
        print "files saved"
        print "step = {}, len(npDelataTripleIdList): {}, len(trainingTriples):{}".format(step, len(npDelataTripleIdList), len(trainingTriples))
                
    return

incrementalSelection()