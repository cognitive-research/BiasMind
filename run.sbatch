#!/bin/bash
#SBATCH -J 666_hgt_s0.1
#SBATCH -p A800
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --qos qmultiple


cog_names=("后见之明偏差" "群体内偏爱" "对比效应" "过度自信效应" "损失厌恶" "结果偏差" "确认偏差" "可用性启发式" "刻板印象" "光环效应" "权威偏见" "框架效应" "从众效应" )


for cog_name in "${cog_names[@]}"; do
    echo "Running for cognitive bias: $cog_name"
    
    srun singularity exec --nv \
    --env NCCL_DEBUG=INFO \
    --env NCCL_SOCKET_IFNAME=ens \
    --env NCCL_P2P_LEVEL=NVL \
    --env TORCH_DISTRIBUTED_DETAIL=DEBUG \
    --env NCCL_SOCKET_FAMILY=AF_NET \
    --bind /path:/path \
    /path/tellerapp_torch251.sif \
    torchrun --nnode=$SLURM_NNODES --nproc_per_node=1 \
    --rdzv_id=$RANDOM --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME:$((RANDOM + 10000)) \
    /path/code/TELLER_hgt/drive_dummy.py  --n_out 2 --batchsize 128 --lr 1e-3 --n_epoch 20 \
    --eval_mode "logics" --graph_merge "tanh_adjustment" --type_of_loss "focal" --focal_alpha "alpha1" --lamda2 1 \
    --device "cuda" --save_path "Qwen_081818_results1.json" --dataset_name "cognitive" \
    --type_of_logic_model "hgt_logic" \
    --dataset_name "cognitive" \
    --cog_name "$cog_name" \
    --graph_flag \
    > log.log 2>&1
    
    echo "Completed for cognitive bias: $cog_name"
done