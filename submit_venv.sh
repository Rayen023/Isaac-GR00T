#!/bin/bash
#SBATCH --account=def-selouani
#SBATCH --gres=gpu:2       # Request GPU "generic resources"
#SBATCH --mem=80G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-23:00
#SBATCH --output=output/%N-%j.out

module load cuda
source .venv/bin/activate

#BATCH_SIZE=8; MAX_STEPS=216000; SAVE_STEPS=20000
#BATCH_SIZE=16; MAX_STEPS=102000; SAVE_STEPS=10000
#BATCH_SIZE=64; MAX_STEPS=147000; SAVE_STEPS=15000
# H100-1g.10gb : --gpus=h100_1g.10gb:1
# H100-2g.20gb : --gpus=h100_2g.20gb:1
# H100-3g.40gb : --gpus=h100_3g.40gb:1
#--gres=gpu:1 

BATCH_SIZE=220; MAX_STEPS=16000; SAVE_STEPS=2000; LEARNING_RATE=0.0002 # 160 uses 80gb vram, 120 uses 74gb vram
#srun --jobid 3626776 --pty tmux new-session -d 'htop -u $USER' \; split-window -h 'watch nvidia-smi' \; attach
#srun --jobid 3636672 --pty tmux new-session -d 'htop -u $USER' \; split-window -h 'watch nvidia-smi' \; split-window -v 'tail -f output/*-3636672.out' \; attach

python run_finetune.py \
    --batch-size $BATCH_SIZE \
    --max-steps $MAX_STEPS \
    --save-steps $SAVE_STEPS \
    --learning-rate $LEARNING_RATE \
