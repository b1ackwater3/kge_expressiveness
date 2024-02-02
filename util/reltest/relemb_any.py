# 关系的表示分析
import numpy as np
import os
# load npy
import torch
base_path = '/home/ai/yl/kge_tool/util/'


models_name = "RotatE"
dataset = "_wn18rr_base_0715"

# sys_ids = [114, 178, 192, 77, 175, 152, 3, 164, 141, 97, 188, 58, 53, 123, 196, 119, 8, 45]
# trans_ids = [152, 9, 64, 75]
sys_ids = [10,3]
trans_ids = []

print(len(sys_ids))
# dataset = "_wn18rr_base_0715"

# select rel embedding

def trans_rotate(rel_emb):
    pi = 3.14159265358979323846
    rot_phase = rel_emb/(8/500/pi)
    a0 = rot_phase + pi  # phase to -pi
    a0 = np.abs(a0)
    a_pi = rot_phase - pi    # phase to pi
    a_pi = np.abs(a_pi)
    final = []
    for i in range(len(a0)):
        rel1 = []
        for j in range(len(a0[0])):
            result = min(a0[i,j],a_pi[i,j])
            rel1.append(result)
        final.append(rel1)
    
    return np.array(final)

def trans_transh(rel_emb):
    rel_emb = np.abs(rel_emb)
    return rel_emb

def trans_paire(rel_emb):
    PI = 3.14159265358979323846
    r_h,r_t = np.split(rel_emb,2,axis=-1)
    sub = np.abs(r_h - r_t) #
    add =  np.abs(r_h + r_t) #
    final = []
    for i in range(len(sub)):
        rel1 = []
        for j in range(len(sub[0])):
            result = min(sub[i,j],add[i,j])
            rel1.append(result)
        final.append(rel1)
    return np.array(final)


rel_emb = np.load(os.path.join(base_path, models_name + dataset + ".npy"))
# rel_emb = trans_rotate(rel_emb)

def select_by_max(rel_emb):
    rel_mean = np.max(rel_emb, axis= 1)
    thread = 2
    is_less = rel_mean < thread
    print(np.sum( is_less))
    print(np.sum(rel_mean[sys_ids]<thread))
    print(np.sum(rel_mean[trans_ids]<thread))

def select_by_mean(rel_emb):
    rel_mean = np.mean(rel_emb, axis= 1)
    thread = 1
    is_less = rel_mean < thread
    print(np.sum( is_less))
    print(np.sum(rel_mean[sys_ids]<thread))
    print(np.sum(rel_mean[trans_ids]<thread))

def select_by_norm(rel_emb):
    rel_mean = np.mean(rel_emb, axis= 1)
    thread = 1
    is_less = rel_mean < thread
    print(np.sum( is_less))
    print(np.sum(rel_mean[sys_ids]<thread))
    print(np.sum(rel_mean[trans_ids]<thread))

def select_by_dim(rel_emb):
    print(rel_emb.shape)
    thread = 0.1
    rel_less = rel_emb < thread
    rel_mean = np.sum(rel_less, axis= 1)
    print(np.sum(rel_mean))
    print(np.sum(rel_mean[sys_ids]))
    print(np.sum(rel_mean[trans_ids]))
    print(np.sum(rel_mean<20))
# 
print(rel_emb[sys_ids])
select_by_max(rel_emb)