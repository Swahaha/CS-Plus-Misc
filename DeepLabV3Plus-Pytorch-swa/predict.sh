#!/bin/bash
#SBATCH --job-name=deeplabv3_predict
#SBATCH --output=predict%j.log
#SBATCH --error=predict%j.err
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

module load python/3.6.15
source /home/users/sm961/CS+/DeepLabV3Plus-Pytorch/venv/bin/activate

cd /home/users/sm961/CS+/DeepLabV3Plus-Pytorch

srun python3 predict.py --input FullData/test/images --dataset map --model deeplabv3plus_resnet50 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_resnet50_map_os16.pth --save_val_results_to results/centropy --gpu_id 0

deactivate
