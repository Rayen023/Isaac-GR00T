from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from utils.direct_inference import DirectGr00tInference
from utils.robot_gestures import say_hello, draw_letter_f   
import time

SEPARATOR = "\n" + "-"*50 + "\n"


MODEL_PATH = "/mnt/67202c8a-ad15-4297-8aba-aeafd1dd3341/Data2/Gr00t_weights/so101-checkpoints_dataset-combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30_gpus-1_steps-10000_config-so100_dualcam_backend-torchvision_av_batch-120_20250718_114056/checkpoint-2000" # very good too

EMBODIMENT_TAG = "new_embodiment"
DATA_CONFIG = "so100_dualcam"

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_calibrated_follower_arm8"
ROBOT_TYPE = "so101_follower"
TASK_DESCRIPTION = "Put the red lego block in the black cup"

DENOISING_STEPS = 8
MAX_CHUNK_LEN = 16

RESET_POSITION = {"shoulder_pan.pos": -0.5882352941176521,
"shoulder_lift.pos": -98.38983050847457,
"elbow_flex.pos": 99.45627548708654,
"wrist_flex.pos": 74.40347071583514,
"wrist_roll.pos": 3.3943833943834107,
"gripper.pos": 1.0575016523463316}

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

policy_client = DirectGr00tInference(
        robot_config=robot_config,
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        data_config=DATA_CONFIG,
        denoising_steps=DENOISING_STEPS, 
    )


policy_client.connect()

try: 
    while True:
        observation_dict = policy_client.robot.get_observation()
        action_chunk = policy_client.get_action(observation_dict, TASK_DESCRIPTION)
        action_horizon = min(MAX_CHUNK_LEN, len(action_chunk))
        for i in range(action_horizon):
            action_dict = action_chunk[i]
            policy_client.robot.send_action(action_dict)
            time.sleep(1/30)  

except KeyboardInterrupt:
    print("Interrupted by user, stopping...")
    policy_client.robot.send_action(RESET_POSITION)
    time.sleep(1)
finally:
    print("Disconnecting...")
    time.sleep(1)
    policy_client.disconnect()
