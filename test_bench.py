from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from utils.direct_inference import DirectGr00tInference
import time
import csv
from datetime import datetime
from utils.view_saved_positions_matplotlib import view_position
import os
import numpy as np
from PIL import Image
import json
import select
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
model = YOLO("best.pt")

SEPARATOR = "\n" + "-"*50 + "\n"

def is_enter_pressed():
    """Check if Enter key has been pressed without blocking."""
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False

def save_args(args_file, model_path, task_description, denoising_steps, max_chunk_len, sleep_between_actions, final_step):
    """Save current run arguments to JSON file."""
    args = {
        "model_name": os.path.basename(model_path),
        "TASK_DESCRIPTION": task_description,
        "DENOISING_STEPS": denoising_steps,
        "MAX_CHUNK_LEN": max_chunk_len,
        "SLEEP_BETWEEN_ACTIONS": sleep_between_actions,
        "FINAL_STEP_OF_CURRENT_RUN": final_step
    }
    with open(args_file, 'w') as f:
        json.dump(args, f, indent=2)

def can_continue_run(args_file, model_path, task_description, denoising_steps, max_chunk_len, sleep_between_actions, total_positions):
    """Check if we can continue from a previous run with matching parameters."""
    if not os.path.exists(args_file):
        return False, 0
    
    try:
        with open(args_file, 'r') as f:
            saved_args = json.load(f)
        
        # Check if all parameters match
        if (saved_args["model_name"] == os.path.basename(model_path) and
            saved_args["TASK_DESCRIPTION"] == task_description and
            saved_args["DENOISING_STEPS"] == denoising_steps and
            saved_args["MAX_CHUNK_LEN"] == max_chunk_len and
            saved_args["SLEEP_BETWEEN_ACTIONS"] == sleep_between_actions and
            saved_args["FINAL_STEP_OF_CURRENT_RUN"] < total_positions):
            return True, saved_args["FINAL_STEP_OF_CURRENT_RUN"]
        else:
            return False, 0
    except Exception as e:
        print(f"Error reading args file: {e}")
        return False, 0

def find_latest_test_dir():
    """Find the most recent test_results directory."""
    test_dirs = [d for d in os.listdir('.') if d.startswith('test_results_') and os.path.isdir(d)]
    if not test_dirs:
        return None
    return sorted(test_dirs)[-1]

def process_step(observation_dict, action_dict, timestamp, frame_index, episode_index, index, task_index, wrist_write_path, front_write_path):
    
    
    wrist_image = observation_dict['wrist']
    front_image = observation_dict['front']
    
    # Store paths and arrays for later saving
    image_buffer = {
        'wrist': (wrist_image, wrist_write_path),
        'front': (front_image, front_write_path)
    }
    
    action = [
        float(action_dict['shoulder_pan.pos']),
        float(action_dict['shoulder_lift.pos']),
        float(action_dict['elbow_flex.pos']),
        float(action_dict['wrist_flex.pos']),
        float(action_dict['wrist_roll.pos']),
        float(action_dict['gripper.pos'])
    ]
    
    observation_state = [
        float(observation_dict['shoulder_pan.pos']),
        float(observation_dict['shoulder_lift.pos']),
        float(observation_dict['elbow_flex.pos']),
        float(observation_dict['wrist_flex.pos']),
        float(observation_dict['wrist_roll.pos']),
        float(observation_dict['gripper.pos'])
    ]
    
    data_dict = {
        "action": action,
        "observation.state": observation_state,
        "timestamp": float(timestamp),
        "frame_index": int(frame_index),
        "episode_index": int(episode_index),
        "index": int(index),
        "task_index": int(task_index)
    }

    
    return data_dict, image_buffer

MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-10000_config-so100_dualcam_backend-torchvision_av_batch-120_20250718_114056/checkpoint-2000"
EMBODIMENT_TAG = "new_embodiment"
DATA_CONFIG = "so100_dualcam"

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_calibrated_follower_arm8"
TASK_DESCRIPTION = "Put the red lego block in the black cup"

DENOISING_STEPS = 8
MAX_CHUNK_LEN = 16
SLEEP_BETWEEN_ACTIONS = 0.0 #0.033  # 30 Hz

RESET_POSITION = {"shoulder_pan.pos": -0.5882352941176521,
"shoulder_lift.pos": -98.38983050847457,
"elbow_flex.pos": 99.45627548708654,
"wrist_flex.pos": 74.40347071583514,
"wrist_roll.pos": 3.3943833943834107,
"gripper.pos": 1.0575016523463316}

# Read total positions
with open("object_positions.csv", 'r') as f:
    total_positions = len(list(csv.DictReader(f)))

# Check if we can continue from a previous run
latest_dir = find_latest_test_dir()
start_episode = 0
continuing_run = False

if latest_dir:
    args_file = f"{latest_dir}/args.json"
    can_continue, last_step = can_continue_run(
        args_file, MODEL_PATH, TASK_DESCRIPTION, 
        DENOISING_STEPS, MAX_CHUNK_LEN, SLEEP_BETWEEN_ACTIONS, 
        total_positions
    )
    
    if can_continue:
        print(f"Found previous run in {latest_dir}")
        print(f"Last completed episode: {last_step}")
        response = input("Continue from previous run? (y/n): ").strip().lower()
        
        if response == 'y':
            eval_dir = latest_dir
            start_episode = last_step + 1
            continuing_run = True
            print(f"Continuing from episode {start_episode}")
        else:
            eval_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Starting new run in {eval_dir}")
    else:
        eval_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
else:
    eval_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Create directories
os.makedirs(eval_dir, exist_ok=True)
images_dir = f"{eval_dir}/images/"
os.makedirs(images_dir, exist_ok=True)
data_dir = f"{eval_dir}/data/"
os.makedirs(data_dir, exist_ok=True)

results_file = f"{eval_dir}/results.csv"
args_file = f"{eval_dir}/args.json"

# Initialize results file if new run
if not continuing_run:
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['position_num', 'success', 'inference_time', 'failure_reason', 'yolo_predicted_class', 'yolo_confidence'])
    # Save initial args
    save_args(args_file, MODEL_PATH, TASK_DESCRIPTION, DENOISING_STEPS, 
              MAX_CHUNK_LEN, SLEEP_BETWEEN_ACTIONS, -1)

print(f"Starting test bench with {total_positions} positions (starting from {start_episode})")
print(f"Results will be saved to: {results_file}")
print(SEPARATOR)

# List to track background save threads
save_threads = []

index = -1
is_connected = False

try:
    for episode_index in range(start_episode, total_positions):
        
        # Create directories for wrist and front camera images
        os.makedirs(f"{images_dir}/observation.images.wrist_view/episode_{episode_index:06d}", exist_ok=True)
        os.makedirs(f"{images_dir}/observation.images.front_view/episode_{episode_index:06d}", exist_ok=True)
        
        # Initialize list to store all frames for this episode
        episode_data = []
        # Initialize buffer for images to save later
        images_to_save = []
        
        print(f"\nTest {episode_index}/{total_positions}")
        print("Showing target position. Press 'q', 'escape', or 'Enter' to continue...")
        
        view_position(episode_index + 1)
        
        # Create robot config and policy client for this episode
        # This ensures cameras are properly reinitialized each iteration
        robot_config = SO101FollowerConfig(
            port=ROBOT_PORT,
            id=ROBOT_ID,
            cameras={
                "wrist": OpenCVCameraConfig(
                    index_or_path=1, ## why did it become 1 not 0 ???
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

        policy_client = DirectGr00tInference(
            robot_config=robot_config,
            model_path=MODEL_PATH,
            embodiment_tag=EMBODIMENT_TAG,
            data_config=DATA_CONFIG,
            denoising_steps=DENOISING_STEPS, 
        )
        
        print("Connecting to robot...")
        
        policy_client.connect()
        is_connected = True
        
        print("Starting inference. Press Enter to stop...")
        start_time = time.time()
        
        try:
            frame_index = -1
            stop_inference = False
            while not stop_inference:
                observation_dict = policy_client.robot.get_observation()
                action_chunk = policy_client.get_action(observation_dict, TASK_DESCRIPTION)
                action_horizon = min(MAX_CHUNK_LEN, len(action_chunk))
                for i in range(action_horizon):
                    action_dict = action_chunk[i]
                    policy_client.robot.send_action(action_dict)
                    time.sleep(SLEEP_BETWEEN_ACTIONS)
                    #processing_time = time.time()
                    observation_dict = policy_client.robot.get_observation()
                    frame_index += 1
                    index += 1
                    data_dict, image_buffer = process_step(
                        observation_dict=observation_dict,
                        action_dict=action_dict,
                        timestamp=time.time() - start_time,
                        frame_index=frame_index,
                        episode_index=episode_index,
                        index=index,
                        task_index=0,
                        wrist_write_path= f"{images_dir}/observation.images.wrist_view/episode_{episode_index:06d}/frame_{frame_index:06d}.png",
                        front_write_path=f"{images_dir}/observation.images.front_view/episode_{episode_index:06d}/frame_{frame_index:06d}.png"
                    )
                    episode_data.append(data_dict)
                    images_to_save.append(image_buffer)
                    
                    # Check if Enter was pressed
                    if is_enter_pressed():
                        print("\nStopping inference...")
                        stop_inference = True
                        break
                    
                    # processing_time = time.time() - processing_time
                    # print(f"Frame {frame_index}: Processing time {processing_time:.3f}s")

        except KeyboardInterrupt:
            print("\nInterrupted by user during inference...")
        
        
        inference_time = time.time() - start_time
        print("Stopping robot...")
        try:
            policy_client.robot.send_action(RESET_POSITION)
            time.sleep(1)
            policy_client.disconnect()
            # destroy policy client to free resources
            del policy_client
            time.sleep(1)
            
            is_connected = False
        except Exception as e:
            print(f"Error stopping robot or disconnecting: {e}")
        
        # Save all buffered images to disk in background
        print(f"Queueing {len(images_to_save)} frames for background saving...")
        
        def save_images_background(images_list, episode_num):
            save_start = time.time()
            
            def save_image_pair(image_buffer):
                wrist_array, wrist_path = image_buffer['wrist']
                front_array, front_path = image_buffer['front']
                Image.fromarray(wrist_array).save(wrist_path)
                Image.fromarray(front_array).save(front_path)
            
            # Save 8 images in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                executor.map(save_image_pair, images_list)
            
            save_time = time.time() - save_start
            print(f"\n[Background] Episode {episode_num}: Saved {len(images_list)} images in {save_time:.2f}s")
        
        # Start background thread - doesn't wait!
        save_thread = threading.Thread(
            target=save_images_background, 
            args=(images_to_save.copy(), episode_index),
            daemon=False
        )
        save_thread.start()
        save_threads.append(save_thread)
        
        print(f"Images queued for saving, continuing...")
        
        # Save all episode data to a single JSON file
        episode_file = f"{data_dir}/episode_{episode_index:06d}.json"
        try:
            with open(episode_file, 'w') as f:
                json.dump(episode_data, f, indent=2)
            print(f"Saved episode data to {episode_file} ({len(episode_data)} frames)")
        except Exception as e:
            print(f"Error saving episode data: {e}")

        # Run YOLO inference on the last front view frame
        predicted_class = -1
        confidence = 0.0
        if 'observation_dict' in locals() and 'front' in observation_dict:
            last_front_frame = observation_dict['front']
            yolo_result = model(last_front_frame, verbose=False)[0]
            predicted_class = yolo_result.probs.top1
            confidence = yolo_result.probs.top1conf.item()
            print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
        
        result = input("Result (s=success, f=failure): ").strip().lower()
        success = 1 if result == 's' else 0
        failure_reason = ""
        
        if result == 'f':
            failure_reason = input("Failure reason: ").strip()
        
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode_index, success, f"{inference_time:.2f}", failure_reason, predicted_class, f"{confidence:.4f}"])
        
        # Update args.json with the latest completed episode
        save_args(args_file, MODEL_PATH, TASK_DESCRIPTION, DENOISING_STEPS, 
                  MAX_CHUNK_LEN, SLEEP_BETWEEN_ACTIONS, episode_index)
        
        print(f"Recorded: Success={success}, Time={inference_time:.2f}s")
        
    print(SEPARATOR)
    print("Test bench complete!")
    print(f"Results saved to: {results_file}")

    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        results = list(reader)
        successes = sum(1 for r in results if r['success'] == '1')
        total_time = sum(float(r['inference_time']) for r in results)
        avg_time = total_time / len(results) if results else 0

    print(f"\nSummary:")
    print(f"Success Rate: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
    print(f"Average Inference Time: {avg_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")

except KeyboardInterrupt:
    print("\nInterrupted by user, stopping test bench...")
except Exception as e:
    print(f"\nError occurred: {e}")
    print("Stopping test bench...")
finally:
    print("Cleaning up...")
    
    # Wait for all background save operations to complete
    if save_threads:
        print(f"Waiting for {len(save_threads)} background save operations to complete...")
        for i, thread in enumerate(save_threads, 1):
            thread.join()
            print(f"  Completed {i}/{len(save_threads)} save operations")
    
    if is_connected:
        try:
            print("Resetting robot position...")
            if 'policy_client' in locals():
                policy_client.robot.send_action(RESET_POSITION)
                time.sleep(1)
        except Exception as e:
            print(f"Error resetting robot: {e}")
        try:
            print("Disconnecting from robot...")
            if 'policy_client' in locals():
                policy_client.disconnect()
        except Exception as e:
            print(f"Error disconnecting: {e}")
    print("Test bench exited.")
