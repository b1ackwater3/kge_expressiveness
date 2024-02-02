from collections import defaultdict
import os


with open("/home/ai/yl/data/FB15k/train.txt",'r') as f:
    lines = f.readlines()
    triples  = []
    for line in lines:
        h,r,t = line.strip().split("\t")
        triples.append((h,r,t))

def find_functional(triples):
    relation = set()
    rel2triples = defaultdict(list)
    for h,r,t in triples:
        relation.add(r)
        rel2triples[r].append((h,r,t))
    for r in relation:
        print("Now start ,", r, '  ', len(rel2triples[r]))
        reltriples = rel2triples[r]
        h2t = defaultdict(list)
        isOk = True
        for h,r,t in reltriples:
            h2t[h].append(t)
            if len(h2t[h]) > 1:
                isOk = False
                break
        if isOk:
            print(r)
print(len(triples))
find_functional(triples)