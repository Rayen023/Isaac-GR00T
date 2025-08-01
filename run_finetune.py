#!/usr/bin/env python3
"""
Script to run GR00T finetuning with configurable arguments and timestamped output directory.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Global configuration variables
DATASET_PATH = "./combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30"
NUM_GPUS = 1
BASE_OUTPUT_DIR = "./so101-checkpoints"
MAX_STEPS = 2000000
DATA_CONFIG = "so100_dualcam"
VIDEO_BACKEND = "torchvision_av"
REPORT_TO = "tensorboard"
BATCH_SIZE = 16

def create_timestamped_output_dir():
    """Create output directory with timestamp and all arguments for uniqueness."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract meaningful parts from dataset path for directory name
    dataset_name = Path(DATASET_PATH).name
    
    output_dir = (
        "runs/finetune_"
        f"{BASE_OUTPUT_DIR}_"
        f"dataset-{dataset_name}_"
        f"gpus-{NUM_GPUS}_"
        f"steps-{MAX_STEPS}_"
        f"config-{DATA_CONFIG}_"
        f"backend-{VIDEO_BACKEND}_"
        f"batch-{BATCH_SIZE}_"
        f"{timestamp}"
    )
    
    return output_dir

def run_finetune():
    """Run the GR00T finetuning script with configured arguments."""
    output_dir = create_timestamped_output_dir()
    
    # Build the command with f-strings
    cmd = [
        "python",
        "scripts/gr00t_finetune.py",
        "--dataset-path", f"{DATASET_PATH}",
        "--num-gpus", f"{NUM_GPUS}",
        "--output-dir", f"{output_dir}",
        "--max-steps", f"{MAX_STEPS}",
        "--data-config", f"{DATA_CONFIG}",
        "--video-backend", f"{VIDEO_BACKEND}",
        "--report-to", f"{REPORT_TO}",
        "--batch-size", f"{BATCH_SIZE}",
        "--save-steps", "25000",
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Run the subprocess
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_finetune()
    sys.exit(exit_code)
