import numpy as np
import torch
import os
import matplotlib.pyplot as plt

root_path = "/home/skl/stw/models/"
# models_names ="PairREYAGO3-10_paire_03_debug"
models_names ="PairREYAGO3-10_paire_08"

rel_num = 29

max_step = 100000
gamma_m = 0.01

steps = []
haddt_result = []
hsubt_result = []
habst_result = []

for step in range(0,max_step,1000):
    steps.append(step)
    file_name = os.path.join(root_path,models_names, models_names+"_"+str(step)+".npy")
    raw_emb = np.load(file_name)
    raw_emb = torch.tensor(raw_emb)
    sys_emb = raw_emb[rel_num]
    r_h,r_t = torch.chunk(sys_emb,2,dim=-1)

    haddt =torch.abs(r_h + r_t)
    hsubt =torch.abs(r_h - r_t)
    habst = torch.abs( torch.abs(r_h) - torch.abs(r_t))

    haddt_num = (haddt < gamma_m).sum()
    hsubt_num = (hsubt < gamma_m).sum()
    habst_num = (habst < gamma_m).sum()

    haddt_result.append(haddt_num.item())
    hsubt_result.append(hsubt_num.item())
    habst_result.append(habst_num.item())

# print(haddt_result)
print(len(steps))
print(len(haddt_result))

plt.figure(figsize=(10,10))
l1 = plt.plot(steps,haddt_result,marker='o',label='h add t')
l2 = plt.plot(steps,hsubt_result,marker='*',label='h sub t')
l3 = plt.plot(steps,habst_result,marker='+',label='abs sub abs')
plt.title(models_names)
plt.xlabel("step")
plt.ylabel("Num of less than gamma_m")
plt.legend()
plt.savefig("/home/skl/stw/kge_tool/figs/"+models_names+".png")
