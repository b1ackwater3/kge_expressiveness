import os

# 统计当前测试数据集的
data_path = "/home/ai/yl/data/wn18rr"

sys_file = os.path.join(data_path,'extend','symmetry_pre.txt')
trans_file = os.path.join(data_path,'extend','transitive_gen.txt')

rel2id = os.path.join(data_path,'relations.dict')
rel2idDict = dict()
with open(rel2id, 'r') as f:
    lines = f.readlines()
    for line in lines:
        id, relname = line.strip().split('\t')
        rel2idDict[relname] = id



sys_rel_set = set()
with open(sys_file,'r') as f:
    lines = f.readlines()
    for line in lines:
        h,r,t = line.strip().split("\t")
        sys_rel_set.add(r)

print("Sym: ")
for rel in sys_rel_set:
    print(rel, " : ", rel2idDict[rel])



trans_rel_set = set()
with open(trans_file,'r') as f:
    lines = f.readlines()
    for line in lines:
        h,r,t = line.strip().split("\t")
        trans_rel_set.add(r)


print("Trans: ")
for rel in trans_rel_set:
    print(rel, " : ", rel2idDict[rel])
