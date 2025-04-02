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
source /home/users/sm961/CS+/LoFTR/venv/bin/activate

cd /home/users/sm961/CS+/Loftr

srun python3 train.py data_config.py main_config.py --exp_name custom --gpus=1 --num_nodes=1 --accelerator="gpu" --batch_size=4 --num_workers=4 --pin_memory=True --check_val_every_n_epoch=1 --log_every_n_steps=10 --limit_val_batches=1.0 --num_sanity_val_steps=1 --benchmark=True --max_epochs=30 --parallel_load_data
