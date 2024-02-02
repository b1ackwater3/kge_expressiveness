

import os
import re

root = "/home/ai/yl/models/expression_final"

def read_from_log(model_name):
    

    file = os.path.join(root, model_name, "train.log")

    mrr = []
    hit1 = []
    hit10 = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # find mrr or hit valid step value
            match = re.search(r"Valid MRR at step (\d+): ([\d.]+)", line)  
            if match:  
                step = int(match.group(1))  # 提取step数字  
                value = float(match.group(2))  # 提取对应的值  
                mrr.append((step, value))

            match_111 = re.search(r"Valid HITS@1 at step (\d+): ([\d.]+)", line)  
            if match_111:  
                step = int(match_111.group(1))  # 提取step数字  
                value = float(match_111.group(2))  # 提取对应的值  
                hit1.append((step, value))

            match = re.search(r"Valid HITS@10 at step (\d+): ([\d.]+)", line)  

            if match:  
                step = int(match.group(1))  # 提取step数字  
                value = float(match.group(2))  # 提取对应的值  
                hit10.append((step, value))
    return mrr, hit1, hit10

model_name = "PairRE_FB15k-237_base_0715"
model_name_aug = "PairRE_FB15k-237_sin_0822_5"
# model_name_aug = "PairRE_FB15k-237_sin_0824_6"

# model_name = "PairRE_wn18rr_base_0715 "
# model_name_aug = "PairRE_wn18rr_sin_0823_5 "

# model_name = "PairRE_YAGO3-10_raw"
# model_name_aug = "PairRE_YAGO3-10_sys_base"

# model_name = "RotatE_FB15k-237_base_0715"
# model_name_aug = "RotatE_FB15k-237_exp8_0730_11_sys"

# model_name = "RotatE_wn18rr_base_0715"
# model_name_aug = "RotatE_wn18rr_exp8_0730_03"

# model_name = "Rotate_YAGO3-10_raw"
# model_name_aug = "RotatE_YAGO3-10_exp_sin_082407"



# model_name = "TransH_FB15k-237_base_0715"
# model_name_aug = "TransH_FB15k-237_exp8_0720_5"
# model_name_aug = "TransH_FB15k-237_exp8_trans"

# model_name = "TransH_wn18rr_base_0715"
# model_name_aug = "TransH_wn18rr_exp8_0821_13"

# model_name = "TransH_YAGO3-10_mrl_base_01"
# model_name_aug = "TransH_YAGO3-10_exp_sin_081404"

base_mrr, base_1, base_10 = read_from_log(model_name_aug)
# aug_mrr, aug_1, aug_10 = read_from_log(model_name_aug)



for i in range(len(base_mrr)):
    print(f"%d\t%s\t%s\t%s" % (base_mrr[i][0],base_mrr[i][1], base_1[i][1], base_10[i][1]))