from collections import defaultdict

import os

def read_triples(file):
    triples = []
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            h,r,t = line.strip().split("\t")
            triples.append((h,r,t))
    return triples

def r2triples(triples):
    rel2triple = defaultdict(list)
    for h,r,t in triples:
        rel2triple[r].append((h,r,t))
    return rel2triple

def any_rel(triples):
    # 判断一个关系的映射属性
    h2t = defaultdict(list)
    t2h = defaultdict(list)
    for h,r,t in triples:
        h2t[h].append(t)
        t2h[t].append(h)
    
    # 统计平均每个h有多少个t:
    t_count = 0
    for h in h2t.keys():
        t_count += len(h2t[h])
    mean_t_count = float(t_count)/len(h2t.keys())

    h_count = 0
    for h in t2h.keys():
        h_count += len(t2h[h])
    mean_h_count = float(h_count)/len(t2h.keys())

    if mean_h_count < 1.1 and mean_t_count < 1.1:
        # 121
        return 1

    if mean_h_count > 1.1 and mean_t_count < 1.1:
        # n21
        return 2

    if mean_h_count > 1.1 and mean_t_count > 1.1:
        # n2n
        return 4

    if mean_h_count < 1.1 and mean_t_count > 1.1:
        # 12n
        return 3


path = "/home/ai/yl/data/YAGO3-1668k/yago_insnet/"
train = read_triples(os.path.join(path, "train.txt"))
valid = read_triples(os.path.join(path, "valid.txt"))
test = read_triples(os.path.join(path, "test.txt"))

all_triples = train + valid + test
rel2triples = r2triples(all_triples)

one2one = []
one2n = []
n2one = []
n2n = []

for r in rel2triples.keys():
    result = any_rel(rel2triples[r])
    if result == 1:
        one2one.append(r)
    elif result == 2:
        n2one.append(r)
    elif result == 3:
        one2n.append(r)
    else:
        n2n.append(r)

print(one2one)
print(one2n)
print(n2one)
print(n2n)
