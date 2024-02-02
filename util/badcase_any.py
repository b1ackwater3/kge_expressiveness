# 用于对模型的bad case 进行分析

import os
import json
from collections import defaultdict
from traceback import print_tb
import numpy as np


def read_dic(file):
    id2name = {}
    name2id = {}
    with open(file, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            eid,ename = line.strip().split('\t')
            id2name[int(eid)] = ename
            name2id[ename] = int(eid)

    return id2name,name2id

def read_ent2type(file):
    id2name = {}
    with open(file, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            eid,ename = line.strip().split('\t')
            id2name[eid] =int(ename)
    return id2name


def read_triples(file):
    result = []
    with open(file, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            h,r,t = line.strip().split('\t')
            result.append((h,r,t))
    return result
          

def caculate_relation_true():
    pass

def print_badcase():
    pass


def read_badcase():
    pass

base_path = "/home/skl/yl/models"
model_list = ['boxlevel_04','TransE_type_dis_1007_01','Rotate_yago3_base_01']
file_name = 'base_case.json'

entities_file = '/home/skl/yl/data/YAGO3-1668k/yago_new_ontonet/entities.dic'
relation_file = '/home/skl/yl/data/YAGO3-1668k/yago_new_ontonet/relations.dic'
test_file = '/home/skl/yl/data/YAGO3-1668k/yago_new_ontonet/test.txt'


id2en, en2id  = read_dic(entities_file)
id2rel,rel2id = read_dic(relation_file)
test = read_triples(test_file)

print(id2rel)

with open(os.path.join(base_path,model_list[0],file_name),'r',encoding='utf-8') as f:
    line  = f.readline()
    bad_case_list = json.loads(line,encoding='utf-8')


badcase_name_list = []
count = 0
for badcase in bad_case_list:
    newCase = {}
    if count == 0:
        print(badcase)
        count += 1
    newCase['true'] = [id2en[badcase['true'][0]],id2en[badcase['true'][2]]]
    new_top = []
    for value in badcase['top10']:
        new_top.append(id2en[value])
    newCase['top10'] = new_top
    newCase['mode'] = badcase['mode']
    newCase['rank'] = badcase['rank']
    newCase['Topscore'] = badcase['Topscore']
    newCase['trueScore'] = badcase['trueScore']

    badcase_name_list.append(newCase)




def save_by_r(bad_case_list,id2en,id2rel):
    rel2badcase = defaultdict(list)
    for badcase in bad_case_list:
        trans_baseC={}
        trans_baseC['true'] = [id2en[badcase['true'][0]],id2rel[badcase['true'][1]],id2en[badcase['true'][2]]]
        new_top = []
        for value in badcase['top10']:
            new_top.append(id2en[value])
        trans_baseC['top10'] = new_top
        trans_baseC['mode'] = badcase['mode']
        rel2badcase[trans_baseC['true'][1]].append(trans_baseC)

# rel2test = defaultdict(list)

# for h,r,t in test:
#     rel2test[r].append((h,r,t))


# 输出关系的正确率
# for rel in rel2test.keys():
#     print("Test Relation %20s  = %5d / %5d = %.4f" % (rel,len(rel2badcase[rel]),len(rel2test[rel])*2,1-float(len(rel2badcase[rel]))/(len(rel2test[rel])*2)) )


def read_typeDiff():
    typeDiff = []
    for line in open(os.path.join('/home/skl/yl/data/YAGO3-1668k/yago_insnet/','typeDiff.txt'),'r',encoding='utf-8'):
        data  = line.strip().split('\t')
        data = list(map(float, data))
        typeDiff.append(data)
    return np.array(typeDiff)



# typeDiff = read_typeDiff()

# ent2typeId = read_ent2type('/home/skl/yl/data/YAGO3-1668k/yago_insnet/entity2typeid.txt')


def print_bad_case(badList):
    count = 0
    total = 0
    for badCase in badList:
        # if badCase['mode'] == 'hr_t' and count < 20:
        if count < 10 and badCase['mode'] == 'h_rt' and badCase['rank'] < 10:
            count += 1
            print("Test %s, True: (%s,  %s )" % (badCase['mode'],badCase['true'][0],badCase['true'][1]))
            print("Rank: %d" % badCase['rank'])
            print('True Score: %s' % (str(badCase['trueScore'])))

            print("Error predicate: %s" % ", ".join(badCase['top10']))
            print("Error_score: %s" %  str(badCase['Topscore']))
            print("******************")
            print("                             ")
        if badCase['mode'] == 'hr_t':
            total += 1
    
    print(count)
    print(total)
        
    print(len(badList))
    

def print_rel_badCase(badList, typeIdff, en2typeId):
    count = 0
    for badCase in badList:
        if count > 20:
            break
        # print("Test %s, True: (%s, %s, %s )" % (badCase['mode'],badCase['true'][0],badCase['true'][1],badCase['true'][2]))
       
        if badCase['mode'] =='hr_t':
            true_type = en2typeId[badCase['true'][2]]
        elif badCase['mode'] =='h_rt':
            true_type = en2typeId[badCase['true'][0]]
        typeIds = []
        typeNames = []
        for ent in badCase['top10']:
            typeId = en2typeId[ent]

            if typeIdff[true_type][typeId] > 0.3:
                typeIds.append(typeId)
                typeNames.append(ent)
        typeDiff_value = typeIdff[true_type][typeIds]
        if np.sum(typeDiff_value) < 1:
            count += 1
            # print("In a type")
        else:
            count += 1
            print("Test %s, True: (%s, %s, %s )" % (badCase['mode'],badCase['true'][0],badCase['true'][1],badCase['true'][2]))
            print("Error predicate: %s" % ", ".join(typeNames))
            print("Error TypeDif: %s" % (str(typeDiff_value)))
    # print("In a type %d / %d" % (count, len(badList)))




print_bad_case(badcase_name_list)


# print_rel_badCase(rel2badcase['playsFor'],typeDiff,ent2typeId)

# print(typeDiff[6618][673])
# print(typeDiff[673][6618])


# 分析badcase的错误的实体的情况
# 读取模型2type的实例，然后看正确样本和预测错误的模型之间的type 有什么关系

