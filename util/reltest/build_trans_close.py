

# 读取当前的传递关系测试
from collections import defaultdict
relations = set()

files1 = "/home/ai/yl/data/YAGO3-10/extend/transitive_pre1.txt"

with open(files1,'r') as f:
    lines = f.readlines()
    for line in lines:
        h,r,t = line.strip().split('\t')
        relations.add(r)

r2triples = defaultdict(list)

files1 = "/home/ai/yl/data/YAGO3-10/train.txt"
with open(files1,'r') as f:
    lines = f.readlines()
    for line in lines:
        h,r,t = line.strip().split('\t')
        if r in relations:
            r2triples[r].append((h,r,t))
files1 = "/home/ai/yl/data/YAGO3-10/valid.txt"

with open(files1,'r') as f:
    lines = f.readlines()
    for line in lines:
        h,r,t = line.strip().split('\t')
        if r in relations:
            r2triples[r].append((h,r,t))

files1 = "/home/ai/yl/data/YAGO3-10/test.txt"
with open(files1,'r') as f:
    lines = f.readlines()
    for line in lines:
        h,r,t = line.strip().split('\t')
        if r in relations:
            r2triples[r].append((h,r,t))


def build_trans_triple(triples):
    hr2triples = defaultdict(set)
    rt2triples = defaultdict(set)
    for h,r,t in triples:
        hr2triples[(h,r)].add((h,r,t))
        rt2triples[(r,t)].add((h,r,t))

    new_build_triples = []
    triples = set(triples)
    print("Trans Caculate begin")
    new_triples = set()
    for h,r,t in triples:
        new_triples.add((h,r,t))
        for triple2 in hr2triples[(t,r)]:
            if (h,r,triple2[2]) not in triples and (h,r,triple2[2]) not in new_triples :
                new_triples.add((h,r,triple2[2]))
                new_build_triples.append((h,r,triple2[2]))
        for triple2 in rt2triples[(r,h)]:
            if (triple2[0],r,t) not in triples and (triple2[0],r,t) not in new_triples :
                new_triples.add((triple2[0],r,t))
                new_build_triples.append((triple2[0],r,t))
    triples = new_triples
    print("Trans Caculate over")
    return new_build_triples

for r in relations:
    r2triples[r].extend(build_trans_triple(r2triples[r]))

files1 = "/home/ai/yl/data/YAGO3-10/transitive_all.txt"
with open(files1,'w') as f:
    for r in relations:
        for h,r,t in r2triples[r]:
            f.write("%s\t%s\t%s\n" % (h,r,t))