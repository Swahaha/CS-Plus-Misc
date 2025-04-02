#!/bin/bash
#SBATCH --job-name=convnettraining
#SBATCH --output=convnet_training%j.log
#SBATCH --error=convnettraining%j.err
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00

module load python/3.6.15
source /home/users/sm961/CS+/DeepLabV3Plus-Pytorch/venv/bin/activate

cd /home/users/sm961/CS+/DeepLabV3Plus-Pytorch


srun python3 main.py --dataset map --data_root FullData --model deeplabv3plus_resnet50 --output_stride 16 --batch_size 16 --val_batch_size 4 --total_itrs 30000 --lr 0.01 --gpu_id 0
