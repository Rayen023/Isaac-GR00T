#!/usr/bin/env python3
"""
Simplified direct GR00T inference script.
Uses the same pattern as eval_lerobot.py but without subprocesses.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Imports following the same pattern as eval_lerobot.py
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

# GR00T imports following inference_service.py pattern
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

# Good ones 
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights2/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-2000000_config-so100_dualcam_backend-torchvision_av_batch-8_20250724_103347/checkpoint-183000"
MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights2/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-2000000_config-so100_dualcam_backend-torchvision_av_batch-16_20250724_103010/checkpoint-75000" #best so far, yup best
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-10000_config-so100_dualcam_backend-torchvision_av_batch-120_20250718_110428/checkpoint-1000" # why tf is this good
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-10000_config-so100_dualcam_backend-torchvision_av_batch-120_20250718_114056/checkpoint-2000" # very good too


# Configuration - same as in your original files
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights2/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-2000000_config-so100_dualcam_backend-torchvision_av_batch-64_20250724_094842/checkpoint-30000"  # MEH
# MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights2/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-2000000_config-so100_dualcam_backend-torchvision_av_batch-64_20250724_094842/checkpoint-12000" # worse
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights2/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-2000000_config-so100_dualcam_backend-torchvision_av_batch-16_20250724_103010/checkpoint-96000" #bad
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights2/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-2000000_config-so100_dualcam_backend-torchvision_av_batch-8_20250724_103347/checkpoint-204000" # overfitted so hard
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights2/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-2000000_config-so100_dualcam_backend-torchvision_av_batch-16_20250724_161637/checkpoint-100000" #unfuncitonal
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights2/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-2000000_config-so100_dualcam_backend-torchvision_av_batch-16_20250724_161637/checkpoint-50000" #MEH

# First group
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-12000_config-so100_dualcam_backend-torchvision_av_batch-32_20250718_055529/checkpoint-5000" # MEH, bad
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-12000_config-so100_dualcam_backend-torchvision_av_batch-64_20250718_024255/checkpoint-8000"
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-10000_config-so100_dualcam_backend-torchvision_av_batch-120_20250718_163926/checkpoint-10000" # good
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-16000_config-so100_dualcam_backend-torchvision_av_batch-64_20250718_153314/checkpoint-13000" #MEH

# FG 120 bs
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-10000_config-so100_dualcam_backend-torchvision_av_batch-120_20250718_163926/checkpoint-3000"
#MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-10000_config-so100_dualcam_backend-torchvision_av_batch-120_20250718_163926/checkpoint-5000" #good meh


EMBODIMENT_TAG = "new_embodiment"
DATA_CONFIG = "so100_dualcam"

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_calibrated_follower_arm8"
ROBOT_TYPE = "so101_follower"
TASK_DESCRIPTION = "Put the red lego block in the black cup"

def view_img(img_dict):
    """Save camera images like in eval_lerobot.py"""
    if isinstance(img_dict, dict):
        # Stack images horizontally
        img = np.concatenate([img_dict[k] for k in img_dict], axis=1)
    else:
        img = img_dict
    
    # Create output directory if it doesn't exist
    output_dir = "camera_feed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save image with timestamp
    timestamp = int(time.time() * 1000)
    filename = f"{output_dir}/camera_feed_{timestamp}.png"
    
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.title("Camera View - Real-time Feed")
    plt.axis("off")
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()
    
    print(f"Camera feed saved to: {filename}")

class DirectGr00tInference:
    """Direct inference client combining robot and policy"""
    
    def __init__(self, robot_config, model_path, embodiment_tag, data_config, denoising_steps=4):
        # Initialize robot
        self.robot = SO101Follower(robot_config)
        
        # Get keys from robot
        self.camera_keys = list(robot_config.cameras.keys())
        self.robot_state_keys = list(self.robot._motors_ft.keys())
        self.modality_keys = ["single_arm", "gripper"]
        
        print(f"Camera keys: {self.camera_keys}")
        print(f"Robot state keys: {self.robot_state_keys}")
        
        # Initialize GR00T policy
        data_config_obj = DATA_CONFIG_MAP[data_config]
        modality_config = data_config_obj.modality_config()
        modality_transform = data_config_obj.transform()
        
        self.policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
        )
        
    def connect(self):
        """Connect to robot"""
        self.robot.connect()
        print("Robot connected successfully!")
        
    def disconnect(self):
        """Disconnect from robot"""
        self.robot.disconnect()
        print("Robot disconnected!")
        
    def get_action(self, observation_dict, lang_instruction):
        """Get action from policy - same logic as Gr00tRobotInferenceClient"""
        # Prepare observation for policy
        obs_dict = {f"video.{key}": observation_dict[key] for key in self.camera_keys}
        
        # Show/save images
        view_img(obs_dict)
        
        # Prepare robot state
        state = np.array([observation_dict[k] for k in self.robot_state_keys])
        obs_dict["state.single_arm"] = state[:5].astype(np.float64)
        obs_dict["state.gripper"] = state[5:6].astype(np.float64)
        obs_dict["annotation.human.task_description"] = lang_instruction
        
        # Add batch dimension (history=1)
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]
        
        # Get action chunk from policy
        action_chunk = self.policy.get_action(obs_dict)
        
        # Convert to lerobot action format (same as eval_lerobot.py)
        lerobot_actions = []
        action_horizon = action_chunk[f"action.{self.modality_keys[0]}"].shape[0]
        for i in range(action_horizon):
            concat_action = np.concatenate(
                [np.atleast_1d(action_chunk[f"action.{key}"][i]) for key in self.modality_keys],
                axis=0,
            )
            action_dict = {key: concat_action[j] for j, key in enumerate(self.robot_state_keys)}
            lerobot_actions.append(action_dict)
        
        return lerobot_actions

def main():
    print("Starting Direct GR00T Inference...")
    
    # Setup robot configuration (same pattern as eval_lerobot.py)
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=0, 
                width=640, 
                height=480, 
                fps=30,
                color_mode=ColorMode.RGB,
                rotation=Cv2Rotation.ROTATE_180
            ),
            "front": OpenCVCameraConfig(
                index_or_path=2, 
                width=640, 
                height=480, 
                fps=30,
                color_mode=ColorMode.RGB,
                
                
            )
        }
    )
    
    # Initialize direct inference client
    policy_client = DirectGr00tInference(
        robot_config=robot_config,
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        data_config=DATA_CONFIG,
    )
    
    try:
        # Connect to robot
        policy_client.connect()
        
        print(f"Starting inference with task: {TASK_DESCRIPTION}")
        
        # Main eval loop (same as eval_lerobot.py)
        while True:
            # Get observation from robot
            observation_dict = policy_client.robot.get_observation()
            print("Got observation from robot")
            
            # Get action chunk from policy
            action_chunk = policy_client.get_action(observation_dict, TASK_DESCRIPTION)
            
            # Execute actions (same as eval_lerobot.py)
            action_horizon = min(8, len(action_chunk))  # Execute up to 8 actions
            for i in range(action_horizon):
                action_dict = action_chunk[i]
                print(f"Executing action {i+1}/{action_horizon}")
                policy_client.robot.send_action(action_dict)
                time.sleep(0.02)  # Same timing as eval_lerobot.py
    
    except KeyboardInterrupt:
        print("\nStopping inference...")
    
    finally:
        policy_client.disconnect()

if __name__ == "__main__":
    main()
