#!/bin/sh
base="/home/skl/stw/models/"$2"_"$3"_"
savePath=$base$4
dataPath="/home/skl/stw/data/"$3
CUDA_VISIBLE_DEVICES=$1  python -u run_all.py --do_train --do_test --do_test_class \
--cuda \
--model $2 \
--data_path $dataPath \
-save $savePath \
-b $5 -n $6 --max_steps $7 \
-lr $8 --lr_step $9 --decay ${10} \
-g ${11} \
--train_type ${12} \
-d ${13}\
${15} ${16} ${17} ${18} 
