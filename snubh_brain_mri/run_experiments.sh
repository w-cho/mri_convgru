mkdir -p exp2
mkdir -p exp2/2d_conv_gru

mkdir -p exp2/2d_conv_gru/split_3
mkdir -p exp2/2d_conv_gru/split_4
mkdir -p exp2/2d_conv_gru/split_5
mkdir -p exp2/2d_conv_gru/split_6
mkdir -p exp2/2d_conv_gru/split_9

nohup python3 -u train.py --split 3 --alpha 0.3 --in_dim 2 --classifier gru --exp exp2/2d_conv_gru/split_3 --gpu_id 0 > exp2/2d_conv_gru/split_3/log.txt &
nohup python3 -u train.py --split 4 --alpha 0.4 --in_dim 2 --classifier gru --exp exp2/2d_conv_gru/split_4 --gpu_id 1 > exp2/2d_conv_gru/split_4/log.txt &
nohup python3 -u train.py --split 5 --alpha 0.4 --in_dim 2 --classifier gru --exp exp2/2d_conv_gru/split_5 --gpu_id 2 > exp2/2d_conv_gru/split_5/log.txt &
nohup python3 -u train.py --split 6 --alpha 0.3 --in_dim 2 --classifier gru --exp exp2/2d_conv_gru/split_6 --gpu_id 3 > exp2/2d_conv_gru/split_6/log.txt &
nohup python3 -u train.py --split 9 --alpha 0.3 --in_dim 2 --classifier gru --exp exp2/2d_conv_gru/split_9 --gpu_id 3 > exp2/2d_conv_gru/split_9/log.txt &
