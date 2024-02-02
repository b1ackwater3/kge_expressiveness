# 输入一个日志列表

import linecache
import os
import re

pre = "/home/skl/yl/models"

fileList = ['PaiRE_FB15k_loss_300_01','PaiRE_FB15k_loss_300_02']


def get_file_last(fileName, num):
    with open(fileName,"rb") as f:
        offset = -100
        file_size = os.path.getsize(fileName)
        while -offset < file_size:
            f.seek(offset,2)
            lines = f.readlines()
            if len(lines) < num:
                offset*=2
            else:
                result = []
                for line in lines[-num:]:
                    result.append(line.decode('utf-8'))
                return result
        print("Not enough lines")

# 通常是5-6行
def get_a_part(lines):
    result = []
    for line in lines:
        line = line.strip()
        num =  re.sub("^.+: ", '', line)
        result.append(num)
    return result


def split_head_tail(lines):
    title = [lines[0]]
    head = lines[2:7]
    tail = lines[9:14]

    title.extend(get_a_part(head))
    title.extend(get_a_part(tail))
    return title


with open(os.path.join(pre,'join_result.txt'),'w',encoding='utf-8') as fout:

    for models in fileList:
        fileName = os.path.join(pre,models,'init_train.log')
        lines = get_file_last(fileName,60)
        
        data11 = lines[0:15]
        data1n = lines[15:30]
        datan1 = lines[30:45]
        datann = lines[45:60]
        result = []
        result.extend(split_head_tail(data11))
        result.extend(split_head_tail(data1n))
        result.extend(split_head_tail(datan1))
        result.extend(split_head_tail(datann))

        fout.write("Mode File: %s" % fileName)
        for data in result:
            fout.write("%s\n" % data)

fout.close()



