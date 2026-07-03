export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=29501 script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_longbench.json
