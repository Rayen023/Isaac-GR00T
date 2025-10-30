#!/usr/bin/env python3
"""
Script to run GR00T finetuning with configurable arguments and timestamped output directory.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_BATCH_SIZE = 120
DEFAULT_MAX_STEPS = 150000
DEFAULT_SAVE_STEPS = 10000
DEFAULT_LEARNING_RATE = 0.0002

DATASET_PATH = "datasets/combined_cleaned_32710frames"
NUM_GPUS = 2
DATA_CONFIG = "so100_dualcam"
VIDEO_BACKEND = "torchvision_av"
REPORT_TO = "tensorboard"

def parse_args():
    parser = argparse.ArgumentParser(description="Run GR00T finetuning with configurable hyperparameters")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Maximum training steps (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS,
                        help=f"Save checkpoint every N steps (default: {DEFAULT_SAVE_STEPS})")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})")
    return parser.parse_args()

def create_timestamped_output_dir(max_steps, batch_size, learning_rate):
    """Create output directory with timestamp and all arguments for uniqueness."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract meaningful parts from dataset path for directory name
    dataset_name = Path(DATASET_PATH).name
    
    output_dir = (
        "runs/finetune_"
        f"{dataset_name}_"
        f"steps-{max_steps}_"
        f"bs-{batch_size}_"
        f"lr-{learning_rate}_"
        f"{timestamp}"
    )
    
    return output_dir

def run_finetune(args):
    """Run the GR00T finetuning script with configured arguments."""
    output_dir = create_timestamped_output_dir(args.max_steps, args.batch_size, args.learning_rate)
    
    # Build the command with f-strings
    cmd = [
        "python",
        "scripts/gr00t_finetune.py",
        "--dataset-path", DATASET_PATH,
        "--num-gpus", str(NUM_GPUS),
        "--output-dir", output_dir,
        "--max-steps", str(args.max_steps),
        "--data-config", DATA_CONFIG,
        "--video-backend", VIDEO_BACKEND,
        "--report-to", REPORT_TO,
        "--batch-size", str(args.batch_size),
        "--save-steps", str(args.save_steps),
        "--learning-rate", str(args.learning_rate),
    ]
        
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"Training completed successfully!")
        return result.returncode
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    args = parse_args()
    exit_code = run_finetune(args)
    sys.exit(exit_code)
