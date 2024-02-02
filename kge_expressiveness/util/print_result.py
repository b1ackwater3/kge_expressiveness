
import os
import re

modles = [
'/home/ai/yl/models/TransH_YAGO3-10_exp_sin_081401/',
'/home/ai/yl/models/TransH_YAGO3-10_exp_sin_081402/',
'/home/ai/yl/models/TransH_YAGO3-10_exp_sin_081403/',
'/home/ai/yl/models/TransH_YAGO3-10_exp_sin_081404/',
'/home/ai/yl/models/TransH_YAGO3-10_exp_sin_081405/',
'/home/ai/yl/models/TransH_YAGO3-10_exp_sin_081406/',
'/home/ai/yl/models/TransH_YAGO3-10_exp_sin_081407/',
'/home/ai/yl/models/TransH_YAGO3-10_exp_sin_081408/',
]



result_list = []

for model in modles:
    result_a =[]
    file_name = os.path.join(model,'temp.log') 
    print(file_name)
    with open(file_name,'r') as f:
        lines = f.readlines()
        for line in lines:
            result = re.search("^.+Test +HITS@\d.+: (\d+\\.\d+)",line)
            # result = re.search("^.+Ndcg (\d+\\.\d+)",line)
            if result != None:
                result_a.append(result.group(1))
            else:
                result = re.search("^.+Test +MR.+: (\d{1,}\\.\d+)",line)
                if result != None:
                    result_a.append(result.group(1))
    result_list.append(result_a)

i = 0
for model in modles:
    aa = result_list[i]
    print("\t".join(aa))
    i+=1

    