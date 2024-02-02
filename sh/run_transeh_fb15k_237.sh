#!/bin/sh
base="/home/conda/yl/models/FB15k-237/TransH_"$2"_"
savePath=$base$3
dataPath="/home/ai/yl/"$2
CUDA_VISIBLE_DEVICES=$1  python -u run_all.py --do_train --do_test  --do_test_class \
--cuda \
--model "TransH" \
--data_path $dataPath \
-save $savePath \
--do_rel_constrain \
-b 1024 -n 400 --max_steps 150001 \
-lr 0.0005  --lr_step 50000 --decay 0.1 \
-g 10 \
--loss_function "MRL" \
--train_type "NagativeSample" \
-d 400 \
$4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20}

