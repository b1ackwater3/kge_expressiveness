from collections import defaultdict
import numpy as np
train_file = "/home/ai/yl/data/YAGO3-10/train.txt"
# train_file = "/home/ai/yl/data/FB15k/train.txt"
# train_file = "/home/ai/yl/data/wn18/train.txt"

train_triples = []
with open(train_file,'r') as f:
    lines = f.readlines()
    for line in lines:
        h,r,t = line.strip().split("\t")
        train_triples.append((h,r,t))


r2triples = defaultdict(list)

for h,r,t in train_triples:
    r2triples[r].append((h,r,t))




def test_transitity(triples, threold = 0.9):
    h2t= defaultdict(list)
    for h,r,t in triples:
        h2t[h].append(t)
    triples_set = set(triples)
    # 统计符合传递关系三元组的个数占据整体的比例
    hit_triples = set()
    hit_triples_1 = set()
    for h,r,t in triples:
        for st in h2t[t]:
            if (h,r,st) in triples_set:
                hit_triples.add((h,r,t))
                hit_triples.add((t,r,st))
                hit_triples.add((h,r,st))
    for h,r,t in triples:
        for st in h2t[t]:
            for tt in h2t[st]:
                if (h,r,tt) in triples_set:
                    hit_triples_1.add((h,r,t))
                    hit_triples_1.add((t,r,st))
                    hit_triples_1.add((st,r,tt))
                    hit_triples_1.add((h,r,tt))

    value = float(len(hit_triples))/len(triples_set)
    value2 = float(len(hit_triples_1))/len(triples_set)

    if value >= threold  and len(triples_set) > 50:
       print(r)
       return True
    return False



def test_trainsitity_new(triples, threold = 0.95):
    h2t= defaultdict(list)
    for h,r,t in triples:
        h2t[h].append(t)
    triples_set = set(triples)
    # 统计符合传递关系三元组的个数占据整体的比例
    hit_triples = set()
    hit_triples_1 = set()
    total = 0
    for h,r,t in triples:
        for st in h2t[t]:
            total += 1
            if (h,r,st) in triples_set:
                hit_triples.add((h,r,t))
                hit_triples.add((t,r,st))
                hit_triples.add((h,r,st))

    value = float(len(hit_triples))/len(triples_set)
    if value >= threold  and len(triples_set) > 50:
       print(r)
       return True
    return False

def build_transitivty_test(triples):
    h2t= defaultdict(list)
    for h,r,t in triples:
        h2t[h].append(t)
    triples_set = set(triples)
    # 统计符合传递关系三元组的个数占据整体的比例
    hit_triples = set()
    test = []
    new_triples = []
    for h,r,t in triples:
        for st in h2t[t]:
            if (h,r,st) not in triples_set:
                hit_triples.add((h,r,t))
                hit_triples.add((t,r,st))
                hit_triples.add((h,r,st))
                test.append(((h,r,t),(t,r,st),(h,r,st)))
                new_triples.append((h,r,st))
    return test,new_triples
tests = []
transe_r = []
total_triple_num = 0
for r in r2triples.keys():
    total_triple_num += len(r2triples[r])
    if(test_trainsitity_new(r2triples[r])):
        transe_r.append(r)
        test,new_triple = build_transitivty_test(r2triples[r])
        r2triples[r].append(new_triple)
        tests.extend(test)
print(len(tests))
# if(len(tests)<5000):
#     for r in transe_r:
#         test,new_triple = build_transitivty_test(r2triples[r])
#         r2triples[r].append(new_triple)
#         tests.extend(test)

# np.random.shuffle(tests)
# pre1 = open("/home/ai/yl/data/YAGO3-10/extend/transitive_pre1_new.txt","w")
# pre2 = open("/home/ai/yl/data/YAGO3-10/extend/transitive_pre2_new.txt","w")
# gen = open("/home/ai/yl/data/YAGO3-10/extend/transitive_gen_new.txt","w")

# for i in range(10000):
#     test = tests[i]
#     pre1.write("%s\t%s\t%s\n" % (test[0][0],test[0][1],test[0][2]))
#     pre2.write("%s\t%s\t%s\n" % (test[1][0],test[1][1],test[1][2]))
#     gen.write("%s\t%s\t%s\n" % (test[2][0],test[2][1],test[2][2]))
# pre1.close()
# pre2.close()
# gen.close()