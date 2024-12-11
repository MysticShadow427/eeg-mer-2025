#!/bin/bash
#SBATCH --job-name=submission       # Job name
#SBATCH --output=/scratch/dkayande/eeg-mer/slurm/eeg_submission.%j.out    # Standard output log
#SBATCH --error=/scratch/dkayande/eeg-mer/slurm/eeg_submission.%j.err     # Standard error log

#SBATCH --partition=gpu-v100                    # Partition (queue)
#SBATCH --time=00:30:00                        # Runtime limit (hh:mm:ss)
#SBATCH --ntasks=1                             # Number of tasks (1 because it's a single job)
#SBATCH --cpus-per-task=1                     # Number of CPUs per task
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=32G                       # Memory per CPU (32GB total for 8 CPUs)

export WANDB_API_KEY=""

module use /apps/generic/modulefiles 
module load miniconda3                   
conda activate /home/dkayande/.conda/envs/emer                  

cd /scratch/dkayande/eeg-mer

srun python inference.py --task subject_identification --model eegnet --voting_strategy mean --resume exps/subject_identification/eegnet/baseline_2024-08-29_16-17-27























