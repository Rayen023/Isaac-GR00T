#!/bin/bash
#SBATCH --account=def-selouani
#SBATCH --gres=gpu:1      # Request GPU "generic resources"
#SBATCH --mem=80G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-23:00
#SBATCH --output=output/%N-%j.out

module load cuda
source .venv/bin/activate
python run_finetune.py