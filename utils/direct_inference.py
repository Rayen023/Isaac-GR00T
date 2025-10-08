import numpy as np
from utils.view_cam import view_img
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


class DirectGr00tInference:
    """Direct inference client combining robot and policy"""
    
    def __init__(self, robot_config, model_path, embodiment_tag, data_config, denoising_steps):
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
