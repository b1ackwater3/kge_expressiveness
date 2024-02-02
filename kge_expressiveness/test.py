import random
import math

# 在半径为raduis 的圆内随机生成num个坐标点
def random_position(raduis, num):
    points = []
    raduis = raduis**2
    for i in range(num):
        point = {}
        angle = random.random()*2*math.pi
        r_p2 = random.random()*raduis
        r =  math.sqrt(r_p2)
        point['x'] = math.cos(angle)*r
        point['y'] = math.sin(angle)*r
        points.append(point)
    return points


raduis = 150
small_raduis = 70
ue_raduis = 17
sbs_count = 0
ue_count = 0


def is_number(x):
    x = x.strip()
    if x.isdigit():
        return True
    if x.count('.') != 1:
        return False
    left = x.split('.')[0]
    right = x.split('.')[1]
    if left.isdigit() and right.isdigit():
        return True
    return False


data_file = 'C:\\Users\\Administrator\\Desktop\\data.txt'

with open(data_file,'r',encoding='utf-8') as fin:
    lines = fin.readlines()
    count = 0
    result = ['0','0','0']
    for line in lines:
        if is_number(line):
            result[count ] = str(line.strip())
            
            count += 1
            if count == 3:
                count = 0
                
                print(result[0],'\t',result[1],'\t',result[2])


def build_network():
    with open("D:\\2021_project\\KG_learn\\KGE_tool\\position.txt",'w',encoding='utf-8') as fin:
        print("Here")
        for i in range(0, 3):
            angle = i * 2*math.pi / 3
            center_x = math.cos(angle)*raduis
            center_y = math.sin(angle)*raduis

            fin.write("MyNet.SBSs[{}].myId = {}\n".format(sbs_count,sbs_count))
            fin.write("MyNet.SBSs[{}].x = {}\n".format(sbs_count, center_x))
            fin.write("MyNet.SBSs[{}].y = {}\n".format(sbs_count, center_y))
            sbs_count = sbs_count + 1

            for k in range(0, 2):
                ue_angle = k * 2 * math.pi/2 
                ue_temple = ue_raduis * random.random()
                ue_temple = ue_raduis

                ue_x = ue_temple * math.cos(ue_angle)+ center_x
                ue_y = ue_temple * math.sin(ue_angle)+ center_y
                fin.write("MyNet.node[{}].myId = {}\n".format(ue_count, ue_count))
                fin.write("MyNet.node[{}].x = {}\n".format(ue_count, ue_x ))
                fin.write("MyNet.node[{}].y = {}\n".format(ue_count, ue_y ))
                ue_count = ue_count + 1


            for j in range(0, 8):
                small_angle = j * 2* math.pi/8
                small_x = math.cos(small_angle)*small_raduis + center_x
                small_y = math.sin(small_angle)*small_raduis + center_y
                fin.write("MyNet.SBSs[{}].myId = {}\n".format(sbs_count,sbs_count))
                fin.write("MyNet.SBSs[{}].x = {}\n".format(sbs_count, small_x))
                fin.write("MyNet.SBSs[{}].y = {}\n".format(sbs_count, small_y))
                sbs_count = sbs_count + 1
                for k in range(0, 2):
                    ue_angle = k * 2 * math.pi/2 
                    ue_temple = ue_raduis * random.random()
                    ue_temple = ue_raduis
                    ue_x = ue_temple * math.cos(ue_angle) + small_x
                    ue_y = ue_temple * math.sin(ue_angle) + small_y
                    fin.write("MyNet.node[{}].myId = {}\n".format(ue_count, ue_count))
                    fin.write("MyNet.node[{}].x = {}\n".format(ue_count, ue_x ))
                    fin.write("MyNet.node[{}].y = {}\n".format(ue_count, ue_y ))
                    ue_count = ue_count + 1
        print("Over")

