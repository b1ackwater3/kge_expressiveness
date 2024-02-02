#%%
import re

file = "/home/ai/yl/models/FB15k-237/PairRE_FB15k-237_test_loss_1/train.log"

def handl_loss(file):
    step2loss = {}
    step2regu = {}
    step2vaild = {}

    reguare = ".+Training average regularization at step (\d+): (\d+.\d+)"
    loss = ".+Training average loss at step (\d+): (\d+.\d+)"

    #%%
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            match = re.match(reguare, line)
            if match:
                step2regu[match.group(1)] = match.group(2)

            match = re.match(loss, line)
            if match:
                step2loss[match.group(1)] = match.group(2)


    step2RelLoss = {}
    for key in step2regu.keys():
        step2RelLoss[key] = float(step2loss[key]) - float(step2regu[key])
    return step2RelLoss

# %%

file1 = "/home/ai/yl/models/FB15k-237/TransH_FB15k-237_test_loss_1/train.log"
file2 = "/home/ai/yl/models/FB15k-237/TransH_FB15k-237_test_loss_2/train.log"
file3 = "/home/ai/yl/models/FB15k-237/TransH_FB15k-237_test_loss_3/train.log"
file4 = "/home/ai/yl/models/FB15k-237/TransH_FB15k-237_test_loss_4/train.log"
file5 = "/home/ai/yl/models/FB15k-237/TransH_FB15k-237_test_loss_5/train.log"
file6 = "/home/ai/yl/models/FB15k-237/TransH_FB15k-237_test_loss_6/train.log"
file7 = "/home/ai/yl/models/FB15k-237/TransH_FB15k-237_test_loss_7/train.log"
file8 = "/home/ai/yl/models/FB15k-237/TransH_FB15k-237_test_loss_8/train.log"
file8 = "/home/ai/yl/models/FB15k-237/TransH_FB15k-237_test_loss_8/train.log"

file9 = "/home/ai/yl//models/expression_final/RotatE_FB15k-237_base_0715/train.log"






loss1 = handl_loss(file1)
loss2 = handl_loss(file2)
loss3 = handl_loss(file3)
loss4 = handl_loss(file4)
loss5 = handl_loss(file5)
loss6 = handl_loss(file6)
loss7 = handl_loss(file7)
loss8 = handl_loss(file8)
loss9 = handl_loss(file9)


with open("./transh_fb_loss.txt",'w+') as f:
    print("OK")
    for key in loss9.keys():
        f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (key, loss1[key],loss2[key],loss3[key],loss4[key],loss5[key],loss6[key],loss7[key],loss8[key]))
        # f.write("%s\t%s\n" % (key,loss9[key]))


