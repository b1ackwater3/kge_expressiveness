import json
import numpy as np
import os


root_dic = "/home/skl/stw/models"
# model_list = ['TransH_YAGO3-10_transh_rep_new1','TransH_YAGO3-10_0408_1','TransH_YAGO3-10_0408_2'
            #   ,'TransH_YAGO3-10_0409_1','TransH_YAGO3-10_0409_2','TransH_YAGO3-10_transh_17','TransH_YAGO3-10_transh_22']

model_list = ['TransH_YAGO3-10_0408_1','TransH_YAGO3-10_0408_2'
              ,'TransH_YAGO3-10_0409_1','TransH_YAGO3-10_0409_2','TransH_YAGO3-10_transh_17','TransH_YAGO3-10_transh_22']
model_list = ['PairREYAGO3-10_base_01','PairREYAGO3-10_paire_new_base_01','PairREYAGO3-10_0408_1'
              ,'PairREYAGO3-10_0408_4','PairREYAGO3-10_0409_2','PairREYAGO3-10_paire_14']

model_list = ['RotatE_YAGO3-10_up_gamma','RotatE_YAGO3-10_exp6_gam_1e4_alpha5e3']
model_list = ['TransH_YAGO3-10_transh_rep_new1']
file_name = "test_case.json"

entities_file = '/home/skl/stw/data/YAGO3-10/entities.dict'
relation_file = '/home/skl/stw/data/YAGO3-10/relations.dict'



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



def read_ent2type(file,id2name):
    with open(file, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            h,r,t = line.strip().split('\t')
            id2name[h] =t
    return id2name


id2entity,entity2id = read_dic(entities_file)
id2relation, relation2id = read_dic(relation_file)
entity2type = {}

type_file = '/home/skl/yl/data/YAGO3-1668k/yago_type_new/train.txt'
entity2type = read_ent2type(type_file,entity2type)
type_file = '/home/skl/yl/data/YAGO3-1668k/yago_type_new/valid.txt'
entity2type = read_ent2type(type_file,entity2type)
type_file = '/home/skl/yl/data/YAGO3-1668k/yago_type_new/test.txt'
entity2type = read_ent2type(type_file,entity2type)


def score_any(file_name,id2entity, id2relation):
    with open(os.path.join(file_name),'r',encoding='utf-8') as f:
        line  = f.readline()
        json_dic = json.loads(line,encoding='utf-8')
        test_cases = json_dic
        pre_top_mean_score = []
        pre_truth_mean_score = []

        nex_top_mean_score = []
        nex_truth_mean_score = []

        ne_top_max = []
        ne_top_min = []
        pre_top_max = []
        pre_top_min = []
        pre_rank = []
        ne_rank = []
        diff_h_and_t = []

        pre_pre_dif = []
        pre_nex_dif = []
        ne_pre_dif = []
        ne_nex_dif = []


        pre_total_min = []
        pre_total_max = []
        pre_total_mean = []

        ne_total_min = []
        ne_total_max = []
        ne_total_mean = []

        for case in test_cases:
            # 这里是每个case:
            # 统计top 10 的 max-value ，min-value，方差
            # 统计正确样本的得分
            # pre_top_mean_score.extend(case['pre_Topscore'])
            # pre_truth_mean_score.append(case['pre_trueScore'])
            diff_h_and_t.append(np.abs(case['pre_trueScore'] - case['ne_trueScore']))

            h,r,t = case['pre_truth']

            triple = id2entity[int(h)] +"\t"+id2relation[int(r)] + "\t"+id2entity[int(t)]

            pre_rank.append(case['pre_trueRank'])
            ne_rank.append(case['ne_trueRank'])
            pre_top_mean_score.extend(case['pre_Topscore'])
            pre_truth_mean_score.append(case['pre_trueScore'])

            pre_top_mean_score.extend(case['pre_Topscore'])
            pre_truth_mean_score.append(case['pre_trueScore'])
        
            pre_top_max.append(np.max(case['pre_Topscore']))
            pre_top_min.append(np.min(case['pre_Topscore']))

            nex_top_mean_score.extend(case['ne_Topscore'])
            nex_truth_mean_score.append(case['ne_trueScore'])
        
            ne_top_max.append(np.max(case['ne_Topscore']))
            ne_top_min.append(np.min(case['ne_Topscore']))

            pre_pre_dif.append(case['pre_gt_pre_score'])
            pre_nex_dif.append(case['pre_gt_ne_score'])
            ne_pre_dif.append(case['ne_gt_pre_score'])
            ne_nex_dif.append(case['ne_gt_ne_score'])

            pre_total_min.append(case['pre_min'])
            pre_total_max.append(case['pre_max'])
            pre_total_mean.append(case['pre_mean'])

            ne_total_min.append(case['ne_min'])
            ne_total_max.append(case['ne_max'])
            ne_total_mean.append(case['ne_mean'])


        # pre_top_score = np.mean(np.array(pre_top_mean_score))
        # print(pre_top_score)
        # pre_top_score = np.mean(np.array(pre_truth_mean_score))
        # print(pre_top_score)  
        # print(np.mean(diff_h_and_t))
        # print(np.mean(np.array(pre_top_max)),'\t',np.mean(np.array(pre_top_min)),'\t',
        #       np.mean(pre_top_mean_score),'\t',np.mean(pre_truth_mean_score),'\t',np.mean(pre_rank),'\t',
        #     np.mean(np.array(ne_top_max)),'\t',np.mean(np.array(ne_top_min)),'\t',
            
        #       np.mean(nex_top_mean_score),'\t',np.mean(nex_truth_mean_score),'\t',np.mean(ne_rank))
        
        print(np.mean(np.array(pre_total_max)),'\t',np.mean(np.array(pre_total_min)),'\t',np.mean(np.array(pre_total_mean)),'\t',
                np.mean(pre_truth_mean_score),'\t', np.mean(np.array(pre_pre_dif)),'\t',np.mean(np.array(pre_nex_dif)),'\t',np.mean(pre_rank),'\t',
              np.mean(np.array(ne_total_max)),'\t',np.mean(np.array(ne_total_min)),'\t',np.mean(np.array(ne_total_mean)),'\t',
                np.mean(nex_truth_mean_score),'\t', np.mean(np.array(ne_pre_dif)),'\t',np.mean(np.array(ne_nex_dif)),'\t',np.mean(ne_rank)
           )
for model in model_list:
    score_any(os.path.join(root_dic,model,file_name), id2entity, id2relation)