#!/bin/sh
base="/home/conda/yl/models/FB15k-237/RotatE_"$2"_"
savePath=$base$3
dataPath="/home/conda/yl/"$2
CUDA_VISIBLE_DEVICES=$1  python -u run_all.py --do_train --do_test --do_test_class \
--cuda \
--model "RotatE" \
--data_path $dataPath \
-save $savePath \
-b 1024 -n 256 --max_steps 100000 \
-lr 0.00005 --lr_step 50000 --decay 0.1 \
-g 9.0 \
--train_type "NagativeSample" \
--do_rel_constrain \
-d 1000 \
$4 ${5} ${6} ${7} ${8}  ${9} ${10} ${11} ${12} ${13} ${14} ${15}
