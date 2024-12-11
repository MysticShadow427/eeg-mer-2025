#!/bin/bash
#SBATCH --job-name=preprocess      # Job name
#SBATCH --output=/scratch/dkayande/eeg-mer/slurm/eeg_preprocess.%j.out    # Standard output log
#SBATCH --error=/scratch/dkayande/eeg-mer/slurm/eeg_preprocess.%j.err     # Standard error log

#SBATCH --partition=compute                    # Partition (queue)
#SBATCH --time=00:15:00                        # Runtime limit (hh:mm:ss)
#SBATCH --ntasks=1                             # Number of tasks (1 because it's a single job)
#SBATCH --cpus-per-task=1                     # Number of CPUs per task
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=16G                    

export WANDB_API_KEY=""

module use /apps/generic/modulefiles 
module load miniconda3                   
conda activate /home/dkayande/.conda/envs/emer                  

cd /scratch/dkayande/eeg-mer

srun python preprocess.py --split_dir data/splits
