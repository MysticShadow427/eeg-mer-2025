#!/bin/bash
#SBATCH --job-name=eeg_train_subject_identification        # Job name
#SBATCH --output=/scratch/dkayande/eeg-mer/slurm/eeg_train_subject_identification.%j.out    # Standard output log
#SBATCH --error=/scratch/dkayande/eeg-mer/slurm/eeg_train_subject_identification.%j.err     # Standard error log

#SBATCH --partition=gpu-v100                    # Partition (queue)
#SBATCH --time=04:00:00                        # Runtime limit (hh:mm:ss)
#SBATCH --ntasks=1                             # Number of tasks (1 because it's a single job)
#SBATCH --cpus-per-task=1                     # Number of CPUs per task
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=32G                       # Memory per CPU (32GB total for 8 CPUs)

export WANDB_API_KEY=""

module use /apps/generic/modulefiles 
module load miniconda3                   
conda activate /home/dkayande/.conda/envs/emer                  

cd /scratch/dkayande/eeg-mer

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

srun python train.py --task subject_identification --model eegconformer --lr 0.001 --epochs 100

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
