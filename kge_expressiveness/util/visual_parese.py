import numpy as np
import torch 
import pandas as pd
import os

root_path = "/home/skl/stw/kge_tool/util/"
# files  = [#"TransH_YAGO3-10_0409_1.npy","TransH_YAGO3-10_0409_2.npy"
#           "TransH_YAGO3-10_mrl_base_01.npy","TransH_YAGO3-10_0508-1.npy","TransH_YAGO3-10_0508-1.npy",
#           "TransH_YAGO3-10_0508-2.npy","TransH_YAGO3-10_0508-3.npy", "TransH_YAGO3-10_0508-4.npy","TransH_YAGO3-10_0508-5.npy"
# , "TransH_YAGO3-10_0508-6.npy","TransH_YAGO3-10_0508-7.npy"
#         #     'transh_6.npy','transh_7.npy','transh_8.npy','transh_9.npy','transh_10.npy','transh_11.npy'
#         #   ,'transh_12.npy','transh_13.npy','transh_14.npy','transh_15.npy','transh_16.npy','transh_17.npy'
#           ]
files  = [#"TransH_YAGO3-10_0409_1.npy","TransH_YAGO3-10_0409_2.npy"
        "TransH_YAGO3-10_0509-1.npy","TransH_YAGO3-10_0509-2.npy",
          ]
def handle_file(file_name,from_0,from_pi,from_2pi,from_01,from_pi1,from_2pi1):
    raw =  np.load(file_name)
    raw_emb = torch.tensor(raw)
    # def normal(raw_emb):
    #     pi = 3.14159265358979323846
    #     return raw_emb/(26)*400*pi + pi

    # raw_emb =normal(raw_emb)
    is_marry = raw_emb[29]
    is_location = raw_emb[9]
    pi = 3.14159265358979323846

    # gamas = [0.001,0.005,0.01,0.05,0.1,0.5,1]
    # gamas = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01]
    gamas = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]


    for gama in gamas:
        a = torch.sum(torch.abs(is_marry) < gama)
        # b=  torch.sum(torch.abs(is_marry-pi) < gama)
        # c = torch.sum(torch.abs(is_marry-2*pi) < gama)
        from_0.append(a.item())
        # from_pi.append(b)
        # from_2pi.append(c)
        a = torch.sum(torch.abs(is_location) < gama)
        # b=  torch.sum(torch.abs(is_location-pi) < gama)
        # c = torch.sum(torch.abs(is_location-2*pi) < gama)
        from_01.append(a.item())
        # from_pi1.append(b)
        # from_2pi1.append(c)


## 
def handle_paire_file(file_name,sys_from0,trans_from0,sys_h_from0,sys_t_from0,trans_h_from0,trans_t_from0,sys_hplust_from0,sys_hsubt_from0,trans_hplust_from0,trans_hsubt_from0):

    raw =  np.load(file_name)
    raw_emb = torch.tensor(raw)

    r_h ,r_t = torch.chunk(raw_emb,2,dim=-1)

    raw_emb = torch.abs(torch.abs(r_h) - torch.abs(r_t))
    raw_trans = torch.abs((r_h) - (r_t))

    is_marry = raw_emb[29]

    is_location = raw_trans[9]

    hplust = r_h + r_t 
    hsubt = r_h - r_t


    r_h = torch.abs(r_h)
    r_t = torch.abs(r_t)

    gamas = [0.001,0.005,0.01,0.05,0.1,0.5,1]
    # gamas = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01]
    for gama in gamas:
        a = torch.sum(torch.abs(is_marry) < gama)
        sys_from0.append(a.item())
        a = torch.sum(torch.abs(is_location) < gama)
        trans_from0.append(a.item())

        a = torch.sum(torch.abs(r_h[29]) < gama)
        sys_h_from0.append(a.item())
        a = torch.sum(torch.abs(r_h[9]) < gama)
        trans_h_from0.append(a.item())

        a = torch.sum(torch.abs(r_t[29]) < gama)
        sys_t_from0.append(a.item())
        a = torch.sum(torch.abs(r_t[9]) < gama)
        trans_t_from0.append(a.item())  

        a = torch.sum(torch.abs(hplust[29]) < gama)
        sys_hsubt_from0.append(a.item())
        a = torch.sum(torch.abs(hplust[9]) < gama)
        trans_hplust_from0.append(a.item())     

        a = torch.sum(torch.abs(hsubt[29]) < gama)
        sys_hplust_from0.append(a.item())
        a = torch.sum(torch.abs(hsubt[9]) < gama)
        trans_hsubt_from0.append(a.item())  


from_0,from_pi,from_2pi,from_01,from_pi1,from_2pi1 = [],[],[],[],[],[]
s_hplust, s_hsubt, t_hplust, t_hsubt = [],[],[],[]
sys_from0,trans_from0,sys_h_from0,sys_t_from0,trans_h_from0,trans_t_from0 =  [],[],[],[],[],[]

is_paire = False
if is_paire:
    for file in files:
        handle_paire_file(os.path.join(root_path,file),sys_from0,trans_from0,sys_h_from0,sys_t_from0,trans_h_from0,trans_t_from0, s_hplust, s_hsubt, t_hplust, t_hsubt)
    data_fram = {
        '0':sys_from0,
        "1":s_hplust,
        "2":s_hsubt,
        "3":sys_h_from0,
        "4":sys_t_from0,
        "5":trans_from0,
        "6":t_hplust,
        "7":t_hsubt,
        "8":trans_h_from0,
        "9":trans_t_from0,
    }
else:
    for file in files:
        handle_file(os.path.join(root_path,file),from_0,from_pi,from_2pi,from_01,from_pi1,from_2pi1)

    data_fram = {
        '0':from_0,
        # "1":from_2pi,
        "2":from_01,
        # "3":from_pi,
        # "4":from_pi1,
        # "5":from_2pi1,
    }

df = pd.DataFrame(data_fram)
df.to_csv("/home/skl/stw/kge_tool/util/transh.csv")

