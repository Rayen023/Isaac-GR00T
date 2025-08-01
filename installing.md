module load python/3.10.13
    "kornia==0.7.4",
module load arrow/14.0.1
module load cuda # must before installing flash_attn
curl -LsSf https://astral.sh/uv/install.sh | sh

uv pip install --no-build-isolation flash-attn==2.7.1.post4 
uv pip install -e .[base]

python scripts/load_dataset.py --dataset-path ./combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30 --plot-state-action --video-backend torchvision_av
python scripts/gr00t_finetune.py --dataset-path ./combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30 --num-gpus 1 --output-dir ./so101-checkpoints --max-steps 10000 --data-config so100_dualcam --video-backend torchvision_av